#!/usr/bin/env python3
"""Prepare the native M4 environment, then invoke Mesa's deqp-runner suite.

Scheduling, filtering, retries, timeouts and result classification belong to
deqp-runner. This launcher only resolves machine paths and records provenance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex
import subprocess
import sys
import time
import tomllib


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["gl", "piglit"], nargs="?", default="gl")
    parser.add_argument("--config", type=Path, default=Path("/g16/tools/mesa-ci/config.json"))
    parser.add_argument("--driver", type=Path, help="Mesa installation prefix; defaults to /opt/mesa-g16")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-tests", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-fails", type=int, default=20)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout <= 0 or args.max_fails < 0:
        parser.error("jobs and timeout must be positive; max-fails must be nonnegative")

    config = json.loads(args.config.read_text())
    ci = Path(config["ci_dir"]).resolve(strict=True)
    template = ci / ("deqp-asahi-m4-gl33-gles30.toml" if args.suite == "gl" else "m4/piglit.toml")
    suite = template.read_text()
    for prefix, path in config["path_map"].items():
        # Replace TOML string prefixes, preserving escaping for local paths.
        replacement = json.dumps(str(Path(path).resolve(strict=True)))
        suite = suite.replace(json.dumps(prefix), replacement)
        suite = suite.replace(json.dumps(prefix)[:-1] + "/", replacement[:-1] + "/")
    parsed = tomllib.loads(suite)

    runner = Path(config["runner"]).resolve(strict=True)
    version = subprocess.check_output([str(runner), "--version"], text=True).strip()
    if version != "deqp-runner " + config["runner_version"]:
        raise RuntimeError(f"Unexpected runner version: {version}")
    driver = (args.driver or Path(config["driver"])).resolve(strict=True)
    # Accept future Mesa library versions without silently selecting two builds.
    libraries = list((driver / "lib").glob("libgallium-*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one libgallium in {driver / 'lib'}")
    library = libraries[0].resolve(strict=True)
    inputs = {str(template): digest(template), str(runner): digest(runner), str(library): digest(library)}
    for section in parsed.get("deqp", []):
        for name in [section["deqp"], *section["caselists"]]:
            inputs[name] = digest(name)
    expectations = []
    for kind, option in [("skips", "--skips"), ("flakes", "--flakes"), ("single-thread", "--single-thread")]:
        for prefix in ["all", "asahi", "asahi-agx2"]:
            path = ci / f"{prefix}-{kind}.txt"
            if path.is_file():
                inputs[str(path)] = digest(path)
                expectations.extend([option, str(path)])
    if not (ci / "all-skips.txt").is_file():
        raise RuntimeError("Missing upstream all-skips.txt")

    output = (args.output or Path(config["results_dir"]) / (time.strftime("%Y%m%d-%H%M%S") + "-" + args.suite)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "suite.toml").write_text(suite)
    baseline = output / "baseline.csv"
    baseline.write_text("".join((ci / f"{prefix}-fails.txt").read_text() + "\n"
                              for prefix in ["all", "asahi", "asahi-agx2"]
                              if (ci / f"{prefix}-fails.txt").is_file()))
    inputs[str(baseline)] = digest(baseline)

    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("MESA_", "ASAHI_", "AGX_", "GALLIUM_", "LIBGL_", "GBM_")) or key in ["LD_PRELOAD", "LD_LIBRARY_PATH", "EGL_PLATFORM"]:
            del env[key]
    env.update(config["environment"])
    env.update(LD_LIBRARY_PATH=str(driver / "lib") + ":" + config["piglit_library_dir"],
               LIBGL_DRIVERS_PATH=str(driver / "lib/dri"),
               GBM_BACKENDS_PATH=str(driver / "lib/gbm"),
               MESA_SHADER_CACHE_DIR=str(output / "shader-cache"))
    user = pwd.getpwnam(config["user"])
    if os.geteuid() == 0:
        # Plasma receives the live Xwayland DISPLAY and authentication path.
        # Read only display variables, never record the rest of its environment.
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                if proc.stat().st_uid != user.pw_uid or (proc / "comm").read_text().strip() != "plasmashell":
                    continue
                display = dict(item.split("=", 1) for item in (proc / "environ").read_bytes().decode().split("\0") if "=" in item)
                env.update({key: display[key] for key in ["DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"] if key in display})
                break
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        os.chown(output, user.pw_uid, user.pw_gid)
    elif os.geteuid() != user.pw_uid:
        raise RuntimeError(f"Run as root or {user.pw_name}")

    command = [str(runner), "suite", "--suite", str(output / "suite.toml"),
               "--output", str(output / "results"), "--baseline", str(baseline),
               "--jobs", str(args.jobs), "--timeout", str(args.timeout),
               "--max-fails", str(args.max_fails), "--save-xfail-logs", *expectations]
    for regex in args.include_tests:
        command.extend(["--include-tests", regex])
    metadata = dict(command=command, runner_version=version, inputs=inputs,
                    driver=str(driver), profile=str(template),
                    environment={key: env[key] for key in set(config["environment"]) | {
                        "LD_LIBRARY_PATH", "LIBGL_DRIVERS_PATH", "GBM_BACKENDS_PATH",
                        "MESA_SHADER_CACHE_DIR", "DISPLAY", "XAUTHORITY"} if key in env},
                    filtered=bool(args.include_tests), prepared_only=args.prepare_only,
                    prepared_at=time.time())
    provenance = ci / "provenance.json"
    if provenance.is_file():
        metadata["setup"] = json.loads(provenance.read_text())
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(shlex.join(command), flush=True)
    print(f"Run record: {output / 'run.json'}", flush=True)
    if args.prepare_only:
        return 0
    if os.geteuid() == 0:
        command = ["runuser", "-u", user.pw_name, "--", *command]
    result = subprocess.run(command, env=env)
    metadata.update(exit_code=result.returncode, finished_at=time.time())
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
