#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu
[ "$#" -eq 1 ] || { echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2; exit 2; }
scene_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$1"
output=$(CDPATH= cd -- "$1" && pwd)
[ ! -e "$output/render-0000-attachment-0.bin" ] || exit 2
export T8132_GLES_BUFFERS=1 T8132_GLES_WIDTH=256 T8132_GLES_HEIGHT=256
export G16G_RENDER_ATTACHMENT_DUMP=$output
"$scene_dir/../../run_t8132_gles_triangle.sh" >"$output/run.log" 2>&1
python3 "$scene_dir/check.py" "$output"
