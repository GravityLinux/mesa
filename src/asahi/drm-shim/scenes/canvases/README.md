# Apple9 canvas clears and preservation

The driver now honors RGBA clear colors, including alpha zero, and reloads
existing attachments on passes which do not clear. Partial background state
selects the existing reload entry point instead of the clear program. Clear
colors are ordinary float4 data in the attachment state graph; both published
views are updated per submission. The data provenance is EXP-M4-09's authored
MRT clear-color captures; the reload entry is also documented in the m1n1 T8132
Clear/Load/Store corpus.

A clear without an API draw currently uses Mesa's ordinary clear rectangle to
supply a complete graphics package. Fast-clear state remains initialized as well;
this is deliberately a correctness path, not an optimized zero-draw clear. The
native zero-draw envelope alone did not retire with inert API stage entries;
that failed experiment was removed. No opaque helper code was disassembled.
Triangle fans/strips are converted to supported indexed triangle lists with
Mesa's existing primitive converter. gl_FragColor is mapped to the single RT0.

After resetting and chainloading m1n1, run one mode per fresh GPU process:

```sh
./run.sh NEW_OUTPUT_DIRECTORY draw
./run.sh NEW_OUTPUT_DIRECTORY clear-only
./run.sh NEW_OUTPUT_DIRECTORY color-only
./run.sh NEW_OUTPUT_DIRECTORY boundary
```

The first three modes alternate two independently allocated canvases over eight
submissions: distinct initial RGBA clears, blended additions without clearing,
transparent-black clears, and more blended additions. `clear-only` omits the
application draw on clear passes; `color-only` also omits depth attachments.
`T8132_GLES_WIDTH=1024` selects 1024×1024 for those modes. Readback is checked
against independently rasterized color/depth, with one byte color tolerance and
2e-6 depth tolerance outside a 1/64-pixel triangle edge band. The publication and
command lengths are checked as well.

`boundary` uses 512×512, one opaque triangle followed by 64 additive triangles,
plus the utility clear. It must split into 32, 32 and 2 hardware draws. Each of
its three attachment snapshots is compared against the corresponding CPU prefix,
checking preservation at automatic batch boundaries rather than just glFinish.

These tests establish attachment clears, stores and reloads. Rendering to an
attachment and then sampling it is still blocked by fragment texture support;
this is not yet a complete LÖVE Canvas implementation. General scissor/viewport
state, additional render-target formats and presentation remain separate work.
Logs and readbacks are in `tmp/apple9-canvases/`.

The direct renderer now preserves Gallium viewport orientation. Raw FBO rows
start at the lower edge; the CPU checker uses that convention. PNG export
(where present) reverses rows for display. Historical captures from the fixed
negative-Y viewport need the earlier checker or an explicit row reversal.
