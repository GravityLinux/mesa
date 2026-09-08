#!/bin/sh
# SPDX-License-Identifier: MIT

# Run a real Mesa EGL/GLES draw through the m1n1-backed Asahi DRM shim.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mesa_root=$(CDPATH= cd -- "$script_dir/../../.." && pwd)
mesa_build=${MESA_BUILD:-/home/nsheth/Projects/asahi/mesa-m1n1-build}
m1n1_root=${M1N1_SHIM_ROOT:-/home/nsheth/Projects/asahi/m1n1-m4-agx}
output=${1:-/home/nsheth/Projects/asahi/logs/t8132_mesa_gles_triangle.ppm}
readback=${T8132_GLES_READBACK:-0}
if [ "$#" -gt 0 ]; then
    readback=1
fi
binary=$mesa_build/t8132_gles_triangle

ninja -C "$mesa_build" \
    src/asahi/drm-shim/libasahi_noop_drm_shim.so \
    src/gallium/targets/dri/libgallium-26.3.0-devel.so \
    src/egl/libEGL_mesa.so.0.0.0

cc -std=c11 -O2 -Wall -Wextra -Werror \
    "$script_dir/t8132_gles_triangle.c" -o "$binary" -lEGL -lGLESv2 -lm

export M1N1_SHIM_LIBRARY=$mesa_build/src/asahi/drm-shim/libasahi_noop_drm_shim.so
export M1N1_SHIM_ROOT=$m1n1_root
export M1N1_SHIM_BINARY=$binary
export M1N1_SHIM_OUTPUT=$output
export T8132_GLES_READBACK=$readback
export G16G_RENDER_SOURCE=1
export G16G_EXECUTE_MESA_RENDER=1
export G16G_MESA_RENDER_STAGE=direct
export AGX_APPLE9_DIRECT_RENDER=1
if [ -n "${T8132_GLES_FRAGMENT_MATRIX:-}" ]; then
    export G16G_MESA_FRAGMENT_MATRIX=1
fi
if [ -n "${T8132_GLES_FRAGMENT_MATRIX_INTERLEAVED:-}" ]; then
    export G16G_MESA_FRAGMENT_MATRIX_INTERLEAVED=1
fi

uv run --python 3.14 --with-requirements "$m1n1_root/requirements-agx.txt" sh -c '
    python_site=$(find "$VIRTUAL_ENV/lib" -maxdepth 2 -type d \
        -name site-packages -print -quit)
    export PYTHONPATH="$M1N1_SHIM_ROOT/proxyclient:$python_site"
    export PYTHONUNBUFFERED=1
    export M1N1DEVICE=${M1N1DEVICE:-/dev/m1n1}
    export LD_PRELOAD="$M1N1_SHIM_LIBRARY"
    export LD_LIBRARY_PATH="'$mesa_build'/src/egl:'$mesa_build'/src/gallium/targets/dri${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="'$mesa_build'/src/egl/50_mesa.json"
    export MESA_LOADER_DRIVER_OVERRIDE=asahi
    export MESA_SHADER_CACHE_DISABLE=true
    # The bounded G16G backend executes render commands today, not the Mesa
    # compute conversion/decompression command used by glReadPixels.
    export ASAHI_MESA_DEBUG=${ASAHI_MESA_DEBUG:-nocompress}
    # This development fixture exercises a selected GLES3 subset. The Apple9
    # backend does not yet meet every requirement for a conformant ES3 context.
    export MESA_GLES_VERSION_OVERRIDE=${MESA_GLES_VERSION_OVERRIDE:-3.0}
    if [ -n "${T8132_DESKTOP_GL:-}" ]; then
        export MESA_GL_VERSION_OVERRIDE=${MESA_GL_VERSION_OVERRIDE:-3.3COMPAT}
    fi
    if [ "${T8132_GLES_READBACK:-0}" = 1 ]; then
        exec "$M1N1_SHIM_BINARY" "$M1N1_SHIM_OUTPUT"
    fi
    exec "$M1N1_SHIM_BINARY"
'
