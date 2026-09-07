#!/bin/sh
# SPDX-License-Identifier: MIT
# Reset and chainload m1n1 before invoking.
set -eu
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { echo "usage: $0 NEW_OUTPUT_DIRECTORY [draw|clear-only|color-only|boundary]" >&2; exit 2; }
scene_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$1"
output_dir=$(CDPATH= cd -- "$1" && pwd)
[ ! -e "$output_dir/render-0000-attachment-0.bin" ] || exit 2
unset T8132_GLES_CLEAR_ONLY T8132_GLES_NO_DEPTH T8132_GLES_BATCH_BOUNDARY
check_mode=
case "${2:-draw}" in
 draw) ;;
 clear-only) export T8132_GLES_CLEAR_ONLY=1; check_mode=--clear-only ;;
 boundary) export T8132_GLES_BATCH_BOUNDARY=1 ;;
 color-only) export T8132_GLES_CLEAR_ONLY=1 T8132_GLES_NO_DEPTH=1; check_mode="--clear-only --no-depth" ;;
 *) exit 2 ;;
esac
export T8132_GLES_CANVASES=1 T8132_GLES_FRAMES=8
export T8132_GLES_WIDTH=${T8132_GLES_WIDTH:-512}
export T8132_GLES_HEIGHT=$T8132_GLES_WIDTH
export G16G_RENDER_ATTACHMENT_DUMP=$output_dir AGX_APPLE9_PACKAGE_TRACE=1
"$scene_dir/../../run_t8132_gles_triangle.sh" >"$output_dir/run.log" 2>&1
if [ -n "${T8132_GLES_BATCH_BOUNDARY:-}" ]; then
 python3 "$scene_dir/check-boundary.py" "$output_dir"
else
 python3 "$scene_dir/check.py" "$output_dir" --size "$T8132_GLES_WIDTH" $check_mode
fi
