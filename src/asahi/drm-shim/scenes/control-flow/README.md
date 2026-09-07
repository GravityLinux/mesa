# Apple9 graphics control flow

Vertex and fragment stages now use the common structured execution-mask compiler:
branches, phi merges, loops, nested loops, break, and lowered continue constructs.
Continuation lowering runs before optimizations which require canonical loop bodies.
No captured control-flow sequence or fixed shader registers are introduced.
Graphics output limitations remain: a complete position export and one merged
RGBA8 fragment output. Texture sampling, window-position input, SSBOs and spilling
are separate compiler milestones.

After resetting and chainloading m1n1:

```sh
./run.sh NEW_OUTPUT_DIRECTORY
```

Eight 512×512 frames vary a uniform loop bound from zero through three, then
repeat with blending. The vertex loop has a different iteration count for each
vertex and a continue. Fragment loops have pixel-dependent bounds, a nested
loop, continue, and a conditional break. Interpolated vertex results are consumed
after the fragment loops. The checker independently evaluates the loop semantics,
interpolation, RGBA8 quantization and depth for every pixel (no edge exclusions).
All eight hardware frames match exactly, including the zero-iteration cases.

The compiler tests also cover explicit NIR continuation constructs, loop values
used by a subsequent if/else, merged outputs in both stages, and gl_FragColor/RT0
equivalence. Detailed logs are in `tmp/apple9-graphics-cf/` in the workspace.

The direct renderer now preserves Gallium viewport orientation. Raw FBO rows
start at the lower edge; the CPU checker uses that convention. PNG export
(where present) reverses rows for display. Historical captures from the fixed
negative-Y viewport need the earlier checker or an explicit row reversal.
