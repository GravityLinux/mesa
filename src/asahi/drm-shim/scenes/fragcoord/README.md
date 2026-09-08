# Apple9 fragment screen coordinates

This fixture draws two triangles forming a quad, with both `gl_FragCoord.xy`
and a normal perspective varying. Three submissions check normalized XY,
exact half-pixel centers plus an asymmetric modulo pattern, and a viewport at
(37,53) smaller than the attachment. The independent CPU checker reads every
pixel and depth sample. Raw FBO rows have bottom-left origin; displayed images
must reverse those rows.

Reset and chainload m1n1, then run:

```
sh src/asahi/drm-shim/scenes/fragcoord/run.sh NEW_OUTPUT_DIRECTORY
```

Set `T8132_GLES_WIDTH` and `T8132_GLES_HEIGHT` to test another attachment size.
The offset case requires width > 101 and height > 127.

The compiler uses fragment SR 0xa0/0xa1 as integer pixel XY, as established by
EXP-0111-m4-fragment-semantics, FS-01, on M4. NIR `load_pixel_coord` exposes
16-bit coordinates; hardware reads the full word, whose value fits the supported
framebuffer dimensions. Conversion to float uses normal NIR/allocator paths.
Gallium advertises upper-left integer centers and mesa/st supplies the API
half-pixel and framebuffer-origin adjustment. No user varying slot is consumed.
The compiler accepts window-position IO and system-value NIR forms for all four
FP32 components. W uses coefficient zero (interpolated reciprocal clip W); Z
uses a separate rasterizer depth coefficient.

Per-draw viewport snapshots now preserve Gallium XY and depth transforms,
including utility clears with a different viewport. Attachment state remains
shared. Exact viewport and API scissoring is now implemented and tested separately in
`../scissor/`. This fixture confines its geometry to the viewport and primarily
tests coordinates. The earlier oversized-triangle negative control is retained
in tmp/apple9-fragcoord; the new scissor fixture validates that case.

Standalone GLSL compile tests do not apply Gallium's coordinate adjustment;
they test compilation only, not OpenGL window-coordinate semantics.

## Reciprocal clip W

`w-tests/*.shader_test` are Piglit shader_runner cases for constant and varying
clip W, clipping at the near plane, and simultaneous perspective varyings.
They need a 250x250 framebuffer (shader_runner default). Expected pixel values
are calculated from screen-space barycentric weights and vertex reciprocal W;
they do not use a captured shader or a reference image as the oracle. RGB stores
one quarter of W so values greater than one remain testable in UNORM output.
The constant cases have no user varyings. The near-plane case also probes an
area removed by clipping.

The API W value is a normal `load_frag_coord_w` intrinsic selected as an
interpolation read of coefficient zero. It needs neither a new instruction
encoding nor an extra user varying. Do not take an additional reciprocal: that
reciprocal is used when reconstructing perspective attributes, not for API W.

## Window depth Z

`load_frag_coord_z` reads a linear coefficient bound to the rasterizer's
`FRAGCOORD_Z` source, with source slot one and the output-select depth bit
enabled. User source slots shift by one when depth is requested, while user
coefficient indices stay unchanged. Depth is appended after the user
coefficients; it does not consume a user varying. With all 32 user components,
W occupies coefficient zero and Z occupies coefficient 33.

`depth.c` is a Piglit utility-based test of default, nondefault, and reversed
OpenGL depth ranges, an offset viewport, and near-plane clipping. Its fragment
shader reads Z, W, and a perspective varying together. An independent CPU
calculation checks both RGBA output and stored depth, including cleared pixels
outside the triangle. All five cases passed on T8132 and llvmpipe with zero
errors across 111,752 checked pixels. This validates window depth after the
viewport depth transform, rather than clip-space or normalized-device Z.

`z-tests/depth-with-32-varyings.shader_test` checks the maximum user-varying
layout together with both Z and W. Like the W fixtures, it uses shader_runner's
250x250 default framebuffer and analytically calculated expectations.

For the local Piglit checkout, build `depth.c` from the workspace root:

```sh
cc -DPIGLIT_USE_OPENGL -DPIGLIT_USE_WAFFLE \
  -I piglit/tests/util -I piglit-build/tests/util -I piglit/src \
  mesa-m1n1-shim/src/asahi/drm-shim/scenes/fragcoord/depth.c \
  -L piglit-build/lib -lpiglitutil_gl -lpiglitutil -lOpenGL -lm \
  -L piglit-local/lib -lwaffle-1 \
  -Wl,-rpath,"$PWD/piglit-build/lib" -o tmp/apple9-piglit-graphics/fragz-depth
```

After resetting and chainloading the standalone m1n1 image, the local hardware
harness can run it with:

```sh
PIGLIT_BINARY="$PWD/tmp/apple9-piglit-graphics/fragz-depth" \
  bash tmp/apple9-piglit-graphics/run.sh
```

Run the shader fixtures through the same harness without `PIGLIT_BINARY`, passing
`mesa-m1n1-shim/src/asahi/drm-shim/scenes/fragcoord/{z,w}-tests/*.shader_test`.
