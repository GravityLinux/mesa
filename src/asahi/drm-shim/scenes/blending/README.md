# Apple9 blending

The RGBA8 fragment path supports ADD with ZERO, ONE, SRC_ALPHA and
ONE_MINUS_SRC_ALPHA factors, separate RGB/alpha factors, and channel write masks.
Mesa lowers blending to ordinary NIR arithmetic around a packed tile load.
Tile loads use the normal result-slot allocator and consumer dependency handling;
there are no fixed shader registers or captured blend sequences.
Each draw enables tile read/modify/write state when destination color is needed.

After resetting and chainloading m1n1, run:

```sh
mesa-m1n1-shim/src/asahi/drm-shim/scenes/blending/run.sh NEW_OUTPUT_DIRECTORY
```

Eight 512×512 frames test source-alpha and additive blending in opposite draw
orders, alpha zero/one, UNORM saturation, and disabled green/alpha writes. Seven application
draws plus the utility clear share one submission: opaque background, four overlays, a depth-rejected
overlay, and a final opaque triangle. Overlays depth-test without writing depth.
Changing blend state on the same GLES program exercises fragment shader variants.
The independent CPU rasterizer checks RGBA within one byte and depth within 2e-6,
excluding only a 1/64-pixel triangle edge band. The runner also checks one
publication per frame and the complete seven-draw command stream.

For tile-read diagnostics, manually set T8132_GLES_BLEND_PROBE=1 in the GLES
fixture and use check.py --probe. This selects ZERO/ONE destination passthrough.
The regular runner intentionally clears this diagnostic setting.

Hardware validation on T8132: all eight frames pass. Separate allocated-producer
experiments validate all six result slots. A native slot-6 read with the wrong
consumer dependency fails; setting the PPP read/modify/write bit alone is
insufficient. The relevant authored corpus sources are EXP-M4-01 f_blend and
EXP-M4-09 blend/write-mask command state. Detailed controls and readbacks are in
`tmp/apple9-blending/` in the workspace.

Other blend equations/factors, logic operations, alpha-to-coverage and MSAA are
not supported by this path. This is a single RGBA8 render-target implementation.
Transparent surfaces still require suitable application draw ordering.

The direct renderer now preserves Gallium viewport orientation. Raw FBO rows
start at the lower edge; the CPU checker uses that convention. PNG export
(where present) reverses rows for display. Historical captures from the fixed
negative-Y viewport need the earlier checker or an explicit row reversal.
