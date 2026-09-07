#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu
[ "$#" = 1 ] || exit 2
scene_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$1"; output_dir=$(CDPATH= cd -- "$1" && pwd)
[ ! -e "$output_dir/render-0000-attachment-0.bin" ] || exit 2
export T8132_GLES_VERTEX_INPUTS=1 T8132_GLES_WIDTH=256 T8132_GLES_HEIGHT=256
export G16G_RENDER_ATTACHMENT_DUMP=$output_dir
export AGX_APPLE9_PACKAGE_TRACE=1
"$scene_dir/../../run_t8132_gles_triangle.sh" > "$output_dir/run.log" 2>&1
python3 "$scene_dir/check.py" "$output_dir"
