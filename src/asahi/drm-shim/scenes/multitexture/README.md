# Apple9 multiple texture bindings

The direct fragment sampling path supports sixteen live textures and sixteen
live samplers, selected independently. This is a hardware-validated capacity,
not a claim that sixteen is the hardware maximum. Sparse API bindings in slots
0–31 compact independently into the two tables. Dynamic indexing is unsupported.

After resetting and chainloading display-enabled m1n1, run from the Mesa tree:

```sh
src/asahi/drm-shim/scenes/multitexture/run.sh /absolute/path/to/new-output
```

The fixture samples all sixteen textures, displays individual results in sixteen
bands, and averages all samples in the lower half. Two frames each contain three
scissored draws. Texture and sampler bindings change between draws without an
intervening finish; the third draw binds one image through all sixteen slots with
different sampler states. The default alternates nearest and linear filtering.

For sixteen distinct sampler states, reset/chainload again and run:

```sh
T8132_MULTITEXTURE_LOD=1 src/asahi/drm-shim/scenes/multitexture/run.sh /absolute/path/to/another-new-output
```

This mode creates five explicit mip levels and sixteen samplers clamped to LODs
0, 0.25, …, 3.75, with linear mip filtering. Bindings rotate between draws and
frames. The quarter-step LODs are exactly representable, making this a precise
sampler-index test. `check.py` compares all RGBA channels of every pixel against
an independent CPU oracle and writes `validation.json`.

## Implementation and hardware evidence

Texture and sampler tables both use **32-byte slots**; sampler state occupies
only the first eight bytes. Each draw allocates tables sized to its live bindings
from the batch pool. Resource reads use the normal batch hazard tracking.
Compiler metadata carries separate binding masks, and VIR sample instructions
carry separate texture and sampler indices through scheduling and allocation.
The existing opaque 256-byte texture launcher is sufficient; no new helper or
captured shader main is introduced.

Controlled Metal probes on T8132 varied texture and sampler selectors separately,
including dependent chains through all sixteen slots. Inspection was limited to
our authored shader mains and caller resource data; proprietary launcher/helper
code was not disassembled. In the current 14-byte direct sample bundle:

| Selector | Encoding |
| --- | --- |
| Texture bit 0 | byte 8, bit 7 |
| Texture bits 1–3 | byte 1, bits 3–5 |
| Sampler bit 0 | byte 9, bit 0 |
| Sampler bits 1–3 | byte 4, bits 0–2 |

These observations describe this direct-table form, not all sampling encodings.
The low three bits of byte 1 continue to select the coordinate register pair.

Validation on 2026-09-07: both modes checked 131,072 pixels each with no failures.
Nearest/linear mode had maximum channel error 1/255; the sixteen-LOD mode matched
exactly. Existing mixed textured/untextured draws and render-to-texture with alpha
blending also matched exactly (131,072 pixels each). All 206 compiler tests passed,
including sparse, independently reversed indices and rejection of 17 live textures.
Local captures and results are under `tmp/apple9-multitexture/` in the workspace.
An earlier 0.2-step LOD probe had 40 pixels differing by 2/255 at a scissor edge;
the exact quarter-step probe isolates indexing from that unresolved precision effect.

Sampling otherwise retains the existing restrictions documented in
`../textures/README.md`: fragment-stage implicit-LOD FP32 2D sampling, supported
UNORM8 formats, and normalized clamp-to-edge coordinates. Vertex texture fetch,
explicit LOD/gradients, other texture types/formats and wrap modes remain future
work. Buffer arguments now use the address tables documented in `../buffers/README.md`.
Varying and per-batch draw limits are separate.
