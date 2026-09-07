# Apple9 viewport and scissor rectangles

Reset and chainload m1n1 before each GPU process, then run:

```
sh run.sh NEW_OUTPUT_DIRECTORY
```

The default gate uses a 512x512 RGBA8 + depth framebuffer and seven submissions:

- Oversized triangle into an offset viewport, exercising guard-band clipping.
- Pixel-exact, non-tile-aligned scissor rectangle.
- Scissor partly outside the framebuffer with negative API coordinates.
- Overlapping per-draw rectangles, empty/disjoint rectangles, and intersection
  with an offset viewport, all in one submission.
- Scissored color clear while the API viewport is elsewhere; glClear must ignore
  that viewport.
- Another scissored clear in a fresh submission, preserving previous contents.
- Empty and off-screen clears, with a masked draw to force unchanged readback.

Set T8132_GLES_WIDTH/HEIGHT to change attachment dimensions.
T8132_SCISSOR_COLOR_ONLY=1 removes depth. T8132_SCISSOR_BOUNDARY=1 adds 40 draws
with independent rectangles, splitting the clear plus draws into 32/9 packets
across two submissions. The checker validates both intermediate and final
attachments: nine captures for eight logical frames.

Implementation reuses Gallium's viewport/scissor intersection and SCISSOR
serializer, populating the batch array already uploaded through the UAPI.
Each draw's PPP record selects its scissor index and enables the fine pixel
test. Region clipping uses enclosing 32x32 tiles. Empty intersections normalize
to a zero-area rectangle. The normal batch flush owns the array and resets its
indices, including automatic draw-limit splitting. No shim changes are needed.

Mesa's scissored-clear VS uses VERT_ATTRIB_POS; the direct compiler now accepts
that semantic with ordinary compacted vertex-element fetch. No special shader
binary is substituted. The clear path continues to use Mesa utility geometry;
the driver does not advertise native scissored fast clears.

Scope is the existing single-viewport, single-layer, single-sample RGBA8 path.
The checker compares raw bottom-origin FBO rows with an independent integer
rectangle oracle and checks depth where attached. Color-only runs assert that
no depth attachment was captured.
