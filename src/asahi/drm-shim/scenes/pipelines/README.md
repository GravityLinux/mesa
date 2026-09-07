# Multiple graphics pipelines in one batch

This GLES fixture changes vertex and fragment programs between draws without
flushing. Each frame has one color/depth clear, several draws, then one
`glFinish`. The driver sees ordinary GL state, not the fixture selection.

Reset and chainload m1n1 before each invocation, then run:

```sh
src/asahi/drm-shim/scenes/pipelines/run.sh NEW_OUTPUT_DIRECTORY [mixed|arena|rollover|pin|wide|split]
```

- `mixed`: A → B → A → C → B → A, with distinct vertex/fragment mains,
  different varying layouts, VBO and procedural vertex generation, VS/FS
  uniforms, and overlapping triangles with distinct depths.
- `arena`: 32 application draws using those pipelines. With the utility clear this
  crosses the 32-slot boundary, exercising attachment preservation in the next batch.
- `rollover`: replace C with D halfway through the four-frame test. An
  intentionally reduced archive limit forces a rebuild between frames.
- `pin`: retain 20 different pipelines in one batch, exceeding the cache's
  soft 16-package limit, then reuse the first pipeline for draw 21.

- `wide`: retain 20 distinct procedural fragment shaders with spatial sine/cosine
  colors, then reuse the first. The resident archive crosses 64 KiB, and the
  checker requires an entry call using bit 17 as well as correct pixels.

- `split`: the same 20 procedural shaders, using indexed draws and a smaller
  archive limit. Admission splits each frame into two GPU submissions before
  capacity is exhausted. The checker counts each indexed draw once, verifies
  color/depth preservation and compares reversed-order frames byte-for-byte.

Every mode runs four hardware frames. Alternating frames reverse draw order.
The independent CPU rasterizer checks color within one byte, depth within
2e-6, and byte-identical attachments for each order pair. A 1/64-pixel band
around triangle edges excludes rasterization-boundary ambiguity. The checker
also requires four hardware publications for the ordinary modes and eight for
`arena`, with every expected VDM size checked. Final images are checked after
the last publication of each frame.

The checker uses the actual requested clear color `(0.75, 0.73, 1, 1)`. The utility clear adds
one indexed draw before the application draws.

The driver snapshots and retains each draw's shader package and resource
bindings. At submission it interns all referenced stage programs into one
resident archive, then publishes independent VS/FS launchers, coefficient
tables and PPP state. Archive rollover happens before any draw's calls are
published, after the previous fixed-USC owner has retired. No opaque helper
was disassembled or replaced for this feature.

Current limits remain 32 draws per batch, the existing compact archive
capacity (96 KiB total, including the fixed helper closure), one color attachment and the supported depth format. Pipeline
switching itself does not add shader operations. Subsequent blending and canvas
work added attachment reloads and automatic submission at the 32-draw limit.
Package retention still prevents eviction of a shader referenced by an earlier draw.

The direct renderer now preserves Gallium viewport orientation. Raw FBO rows
start at the lower edge; the CPU checker uses that convention. PNG export
(where present) reverses rows for display. Historical captures from the fixed
negative-Y viewport need the earlier checker or an explicit row reversal.
