#!/bin/sh
# SPDX-License-Identifier: MIT
# Reset and chainload m1n1 before invoking.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 NEW_OUTPUT_DIRECTORY [mixed|arena|rollover|pin|wide|split]" >&2
    exit 2
fi
scene_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$1"
output_dir=$(CDPATH= cd -- "$1" && pwd)
if [ -e "$output_dir/render-0000-attachment-0.bin" ]; then
    echo "output directory already contains a render capture" >&2
    exit 2
fi
unset T8132_GLES_PIPELINE_SPLIT T8132_GLES_PIPELINE_WIDE T8132_GLES_PIPELINE_PIN T8132_GLES_PIPELINE_ROLLOVER AGX_APPLE9_RENDER_ARCHIVE_TEST_LIMIT
export T8132_GLES_DRAWS=6
case "${2:-mixed}" in
    mixed) ;;
    arena) export T8132_GLES_DRAWS=32 ;;
    rollover)
        export T8132_GLES_PIPELINE_ROLLOVER=1
        export AGX_APPLE9_RENDER_ARCHIVE_TEST_LIMIT=0xe240 ;;
    split)
        export T8132_GLES_PIPELINE_SPLIT=1 T8132_GLES_PIPELINE_WIDE=1
        export T8132_GLES_PIPELINE_PIN=1 T8132_GLES_DRAWS=21
        export AGX_APPLE9_RENDER_ARCHIVE_TEST_LIMIT=0x11000 ;;
    wide) export T8132_GLES_PIPELINE_WIDE=1 T8132_GLES_PIPELINE_PIN=1 T8132_GLES_DRAWS=21 ;;
    pin) export T8132_GLES_PIPELINE_PIN=1 T8132_GLES_DRAWS=21 ;;
    *) echo "unknown pipeline test: $2" >&2; exit 2 ;;
esac
export T8132_GLES_PIPELINES=1
export T8132_GLES_WIDTH=512 T8132_GLES_HEIGHT=512 T8132_GLES_FRAMES=4
export G16G_RENDER_ATTACHMENT_DUMP=$output_dir
export AGX_APPLE9_PACKAGE_TRACE=1
"$scene_dir/../../run_t8132_gles_triangle.sh" >"$output_dir/run.log" 2>&1
python3 "$scene_dir/check.py" "$output_dir"
