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
The compiler accepts window-position IO and system-value NIR forms, but rejects
live Z/W reads. This is not complete `gl_FragCoord` support.

Per-draw viewport snapshots now preserve Gallium XY and depth transforms,
including utility clears with a different viewport. Attachment state remains
shared. Exact viewport and API scissoring is now implemented and tested separately in
`../scissor/`. This fixture confines its geometry to the viewport and primarily
tests coordinates. The earlier oversized-triangle negative control is retained
in tmp/apple9-fragcoord; the new scissor fixture validates that case.

Standalone GLSL compile tests do not apply Gallium's coordinate adjustment;
they test compilation only, not OpenGL window-coordinate semantics.
