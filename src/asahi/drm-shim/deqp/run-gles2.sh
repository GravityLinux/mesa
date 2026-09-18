#!/bin/bash
# SPDX-License-Identifier: MIT
# Build and chainload m1n1 before each invocation. One process owns the GPU.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mesa_root=$(cd -- "$script_dir/../../../.." && pwd)
workspace=$(dirname "$mesa_root")
export M1N1_SHIM_ROOT=${M1N1_SHIM_ROOT:-$workspace/m1n1-m4-agx}
export DEQP_MESA_BUILD=${MESA_BUILD:-$workspace/mesa-m1n1-build}
export M1N1_SHIM_LIBRARY=$DEQP_MESA_BUILD/src/asahi/drm-shim/libasahi_noop_drm_shim.so
export G16G_RENDER_SOURCE=1 G16G_EXECUTE_MESA_RENDER=1
export G16G_MESA_RENDER_STAGE=direct
export G16G_RENDER_PRESENT=tiled-bgra8 G16G_RENDER_PRESENT_ON_SWAP=1
export G16G_NATIVE_EXECUTOR=1 EGL_PLATFORM=surfaceless
export DEQP_BINARY=${DEQP_BINARY:-$workspace/deqp-build/modules/gles2/deqp-gles2}
uv run --python "${M1N1_SHIM_PYTHON:-3.14}" --with-requirements "$M1N1_SHIM_ROOT/requirements-agx.txt" sh -c '
    python_site=$(python -c '\''import sysconfig; print(sysconfig.get_paths()["purelib"])'\'')
    export PYTHONPATH="$M1N1_SHIM_ROOT/proxyclient:$python_site"
    export PYTHONUNBUFFERED=1 M1N1DEVICE=${M1N1DEVICE:-/dev/m1n1}
    export LD_PRELOAD="$M1N1_SHIM_LIBRARY"
    export LD_LIBRARY_PATH="$DEQP_MESA_BUILD/src/egl:$DEQP_MESA_BUILD/src/gallium/targets/dri${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$DEQP_MESA_BUILD/src/egl/50_mesa.json"
    export MESA_LOADER_DRIVER_OVERRIDE=asahi
    export MESA_SHADER_CACHE_DISABLE=true ASAHI_MESA_DEBUG=nocompress
    exec "$DEQP_BINARY" --deqp-archive-dir="$(dirname "$DEQP_BINARY")" --deqp-surface-type=pbuffer --deqp-surface-width=256 --deqp-surface-height=256 --deqp-gl-config-name=rgba8888d24s8ms0 "$@"
' sh "$@"
