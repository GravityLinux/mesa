#!/bin/sh
# SPDX-License-Identifier: MIT
# Reset and chainload m1n1 before invoking.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 NEW_OUTPUT_DIRECTORY [seven|eight|nine|twelve|16|24|32|linear|flat|mixed|mixed-clipped|integer|perspective|clipped|depth-clipped|procedural|zero]" >&2
    exit 2
fi
scene_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mode=${2:-nine}
vertex=vertex.glsl
fragment=fragment.glsl
components=9
perspective=
clipped=
interpolation=
case "$mode" in
    16|24|32) vertex=vertex-$mode.glsl; fragment=fragment-$mode.glsl; components=$mode; perspective=--perspective-varyings ;;
    linear|flat|mixed|integer) vertex=vertex-$mode.glsl; fragment=fragment-$mode.glsl; interpolation="--interpolation $mode" ;;
    mixed-clipped) vertex=vertex-mixed-clipped.glsl; fragment=fragment-mixed.glsl; interpolation="--interpolation mixed"; clipped=--clipped-varyings ;;
    nine) ;;
    seven) fragment=fragment-seven.glsl; components=7 ;;
    eight) fragment=fragment-eight.glsl; components=8 ;;
    twelve) vertex=vertex-twelve.glsl; fragment=fragment-twelve.glsl; components=12 ;;
    perspective) vertex=vertex-perspective.glsl; perspective=--perspective-varyings ;;
    clipped) vertex=vertex-clipped.glsl; perspective=--perspective-varyings; clipped=--clipped-varyings ;;
    depth-clipped) vertex=vertex-depth-clipped.glsl; perspective=--perspective-varyings; clipped=--depth-clipped-varyings ;;
    procedural) vertex=vertex-procedural.glsl; fragment=fragment-procedural.glsl; components=12 ;;
    zero) fragment=fragment-zero.glsl; components=0 ;;
    *) echo "unknown varying test: $mode" >&2; exit 2 ;;
esac
mkdir -p -- "$1"
output_dir=$(CDPATH= cd -- "$1" && pwd)
if [ -e "$output_dir/render-0000-attachment-0.bin" ]; then
    echo "output directory already contains a render capture" >&2
    exit 2
fi
if [ -n "$interpolation" ]; then
    export MESA_EXTENSION_OVERRIDE=GL_NV_shader_noperspective_interpolation
fi
export T8132_GLES_VERTEX_SOURCE=$scene_dir/$vertex
export T8132_GLES_FRAGMENT_SOURCE=$scene_dir/$fragment
export T8132_GLES_FRAMES=2
if [ "$mode" = procedural ]; then
    unset T8132_GLES_MESH
    export T8132_GLES_VERTICES=6 T8132_GLES_DRAWS=1
else
    export T8132_GLES_MESH=quad
fi
export T8132_GLES_WIDTH=512 T8132_GLES_HEIGHT=512
export G16G_RENDER_ATTACHMENT_DUMP=$output_dir
"$scene_dir/../../run_t8132_gles_triangle.sh" >"$output_dir/run.log" 2>&1
for frame in 0 1; do
    capture=$(printf 'render-%04d-attachment-0.bin' "$frame")
    python3 "$scene_dir/../mesh/readback.py" "$output_dir/$capture" \
        --mode quad --varyings "$components" $perspective $clipped $interpolation --frame "$frame" \
        --output "$output_dir/frame-$frame"
done
cmp "$output_dir/render-0000-attachment-0.bin" "$output_dir/render-0001-attachment-0.bin"
