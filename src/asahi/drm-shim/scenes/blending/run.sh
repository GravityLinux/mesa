#!/bin/sh
# SPDX-License-Identifier: MIT
# Reset and chainload m1n1 before invoking.
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
scene_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$1"
output_dir=$(CDPATH= cd -- "$1" && pwd)
if [ -e "$output_dir/render-0000-attachment-0.bin" ]; then
    echo "output directory already contains a render capture" >&2
    exit 2
fi
unset T8132_GLES_BLEND_PROBE
export T8132_GLES_BLEND=1 T8132_GLES_FRAMES=8
export T8132_GLES_WIDTH=512 T8132_GLES_HEIGHT=512
export G16G_RENDER_ATTACHMENT_DUMP=$output_dir AGX_APPLE9_PACKAGE_TRACE=1
"$scene_dir/../../run_t8132_gles_triangle.sh" >"$output_dir/run.log" 2>&1
python3 "$scene_dir/check.py" "$output_dir"
