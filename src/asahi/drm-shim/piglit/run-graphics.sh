#!/bin/bash
# SPDX-License-Identifier: MIT
# Reset and chainload m1n1 before invoking; one process owns the GPU.
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo "usage: $0 NEW_RESULTS_DIRECTORY [SHADER_TEST ...]" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mesa_root=$(cd -- "$script_dir/../../../.." && pwd)
workspace=$(dirname "$mesa_root")
mesa_build=${MESA_BUILD:-$workspace/mesa-m1n1-build}
export M1N1_SHIM_ROOT=${M1N1_SHIM_ROOT:-$workspace/m1n1-m4-agx}
export M1N1_SHIM_LIBRARY=$mesa_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
export ASAHI_GRAPHICS_BUILD=$mesa_build
export PIGLIT_ROOT=${PIGLIT_ROOT:-$workspace/piglit}
export PIGLIT_BUILD=${PIGLIT_BUILD:-$workspace/piglit-build}
export PIGLIT_LOCAL=${PIGLIT_LOCAL:-$workspace/piglit-local}
mkdir -p -- "$1"
results=$(cd -- "$1" && pwd)
shift
if [ -e "$results/run.log" ]; then
    echo "Refusing to replace $results/run.log" >&2
    exit 2
fi
cases=("$@")
if [ -n "${ASAHI_GRAPHICS_BINARY:-}" ]; then
    [ -x "$ASAHI_GRAPHICS_BINARY" ] || { echo "Missing executable: $ASAHI_GRAPHICS_BINARY" >&2; exit 2; }
    expected=${ASAHI_GRAPHICS_EXPECTED_RESULTS:-1}
else
    if [ "${#cases[@]}" -eq 0 ]; then
        while IFS= read -r entry; do
            case "$entry" in
                ''|'#'*) continue ;;
                mesa/*) cases+=("$mesa_root/${entry#mesa/}") ;;
                piglit/*) cases+=("$PIGLIT_ROOT/${entry#piglit/}") ;;
                *) echo "Unknown graphics manifest entry: $entry" >&2; exit 2 ;;
            esac
        done < "$script_dir/graphics-tests.txt"
    fi
    for test in "${cases[@]}"; do
        [ -f "$test" ] || { echo "Missing shader test: $test" >&2; exit 2; }
    done
    [ -x "$PIGLIT_BUILD/bin/shader_runner" ] || {
        echo "Build Piglit shader_runner with $script_dir/setup.sh first" >&2
        exit 2
    }
    expected=${#cases[@]}
fi
export G16G_RENDER_SOURCE=1 G16G_EXECUTE_MESA_RENDER=1
export G16G_MESA_RENDER_STAGE=direct AGX_APPLE9_DIRECT_RENDER=1
export G16G_RENDER_PRESENT=tiled-bgra8 G16G_RENDER_PRESENT_ON_SWAP=1
export G16G_NATIVE_EXECUTOR=1 EGL_PLATFORM=surfaceless
export PIGLIT_PLATFORM=surfaceless_egl PIGLIT_NO_WINDOW=1
# Test context only: this backend does not claim GL 2.1 conformance.
export MESA_GL_VERSION_OVERRIDE=2.1 MESA_GLSL_VERSION_OVERRIDE=120
set +e
timeout --kill-after=5 "${T8132_PIGLIT_TIMEOUT:-300}" \
    uv run --python "${M1N1_SHIM_PYTHON:-3.14}" \
    --with-requirements "$M1N1_SHIM_ROOT/requirements-agx.txt" sh -c '
    python_site=$(python -c '\''import sysconfig; print(sysconfig.get_paths()["purelib"])'\'')
    export PYTHONPATH="$M1N1_SHIM_ROOT/proxyclient:$python_site"
    export PYTHONUNBUFFERED=1 M1N1DEVICE=${M1N1DEVICE:-/dev/m1n1}
    export LD_PRELOAD="$M1N1_SHIM_LIBRARY"
    export LD_LIBRARY_PATH="$PIGLIT_LOCAL/lib:$PIGLIT_BUILD/lib:$ASAHI_GRAPHICS_BUILD/src/egl:$ASAHI_GRAPHICS_BUILD/src/gallium/targets/dri${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$ASAHI_GRAPHICS_BUILD/src/egl/50_mesa.json"
    export __GLX_VENDOR_LIBRARY_NAME=mesa MESA_LOADER_DRIVER_OVERRIDE=asahi
    export MESA_SHADER_CACHE_DISABLE=true ASAHI_MESA_DEBUG=nocompress
    if [ -n "${ASAHI_GRAPHICS_BINARY:-}" ]; then
        exec "$ASAHI_GRAPHICS_BINARY" "$@" -auto
    fi
    exec "$PIGLIT_BUILD/bin/shader_runner" "$@" -auto -report-subtests
' sh "${cases[@]}" > "$results/run.log" 2>&1
status=$?
set -e
python3 - "$results" "$status" "$expected" <<'PY'
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
status, expected = map(int, sys.argv[2:])
results = []
for line in (root / "run.log").read_text(errors="replace").splitlines():
    if not line.startswith("PIGLIT:"):
        continue
    report = json.loads(line.partition(":")[2])
    if "subtest" in report:
        results.extend({"test": name, "result": result}
                       for name, result in report["subtest"].items())
    elif "result" in report:
        results.append({"test": "single", "result": report["result"]})
counts = dict(collections.Counter(item["result"] for item in results))
summary = {"process_status": status, "expected_tests": expected,
           "counts": counts, "tests": results}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Graphics: {counts}; {len(results)}/{expected} results; process status {status}")
print(f"Log: {root / 'run.log'}")
sys.exit(0 if status == 0 and len(results) == expected and
         all(item["result"] == "pass" for item in results) else 1)
PY
