#!/usr/bin/env python3
"""Compare ordinary pipeline compilation using a non-executing recording DRM shim.

Requires the host probe with arguments:
mesh.spv fragment.spv halfz color.bin depth.bin dry-run q.bin
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "shim", "baseline-icd", "candidate-icd", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args = argparse.Namespace(**{k: v.resolve() for k, v in vars(args).items()})
    root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in os.environ.items()
            if not k.upper().startswith(("AGX_", "ASAHI_", "HK_", "VK_", "MESA_", "LD_"))}
    rows = []
    for threads in (1, 16, 128):
        for primitive in (0, 1):
            name = f"T{threads}-primitive{primitive}"
            shaders = args.output / name
            shaders.mkdir(exist_ok=True)
            for stage in ("mesh", "frag"):
                spv = shaders / ("arrays." + stage + ".spv")
                subprocess.run([
                    "glslangValidator", "-V", "--target-env", "vulkan1.3",
                    "-DTHREADS=" + str(threads), "-DPRIMITIVE=" + str(primitive),
                    str(root / ("native-mesh-output-arrays." + stage)), "-o", str(spv),
                ], check=True, capture_output=True)
                subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(spv)],
                               check=True, capture_output=True)
            for label in ("baseline", "candidate"):
                directory = shaders / label
                directory.mkdir(exist_ok=True)
                env = dict(base, LD_PRELOAD=str(args.shim),
                           VK_DRIVER_FILES=str(getattr(args, label + "_icd")),
                           HK_EXPERIMENTAL_APPLE9="1", HK_APPLE9_NATIVE_MESH="1",
                           HK_APPLE9_NATIVE_MESH_ATTRIBUTES="1",
                           HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING="1",
                           hk_enable_vertex_pipeline_stores_atomics="true",
                           MESA_SHADER_CACHE_DISABLE="true",
                           ASAHI_HOST_TRACE_DIR=str(directory))
                run = subprocess.run([
                    str(args.probe), str(shaders / "arrays.mesh.spv"),
                    str(shaders / "arrays.frag.spv"), "0",
                    str(directory / "color.bin"), str(directory / "depth.bin"),
                    "1", str(directory / "q.bin"),
                ], env=env, cwd=directory, capture_output=True, text=True, timeout=45)
                log = run.stdout + run.stderr
                (directory / "log.txt").write_text(log)
                if label == "baseline":
                    assert run.returncode != 0 and "PIPELINE_OK" not in log, log
                    assert "constant IO slots" in log or "slot offsets must be constant" in log, log
                else:
                    assert run.returncode == 0 and "PIPELINE_OK" in log, log
                    assert "submission=0 dry_run=1" in log, log
                    assert list(directory.glob("*.commands.bin"))
                    for file in ("color.bin", "depth.bin"):
                        assert (directory / file).read_bytes() == b"\xa5" * 16384
                    expected = bytearray(b"\xa5" * 16384)
                    for i in range(128):
                        expected[64*i:64*i+4] = bytes(4)
                    assert (directory / "q.bin").read_bytes() == expected
                # The shim records large, unused initializer BOs as well.
                for file in directory.glob("*.bin"):
                    if file.stat().st_size >= 0x4000000:
                        file.unlink()
                rows.append(dict(case=name, version=label, returncode=run.returncode,
                                 expected_rejection=label == "baseline",
                                 gpu_executed=False))
                print(rows[-1], flush=True)
    (args.output / "RESULTS.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
