# Linked vertex/fragment varyings

Apple9 supports up to **32 user components** (eight vec4s), in addition to the
four position components. FP32 inputs support smooth perspective, center
noperspective, and flat interpolation, including mixed modes in one interface.
Flat signed/unsigned 32-bit integers use bit-preserving publications. This is
the current compiler capacity, not a claimed hardware maximum.

The compiler records a component mask for each user location VAR0..VAR31 and
compacts the written components into scalar export slots. The FS is compiled
against that producer layout; its shader key includes the layout. Constant
array offsets, gaps between locations, and partial vectors are supported.
An input absent from the producer is rejected. Declaration order does not
define the cross-stage mapping.

The VS exports ordinary, unprojected values so homogeneous clipping can create
new vertices correctly. The driver requests native perspective coefficients
(shade 7), and the FS uses **coefficient-aware projective multiplication** with
reciprocal interpolated 1/W. This handles the rasterizer's primitive-constant
coefficient representation as well as varying values.

The earlier ordinary multiply incorrectly applied an extra W factor to constant
components. Its workaround—pre-dividing varyings in the VS and requesting linear
coefficients—passed unclipped and guard-band cases but corrupted attributes when
hardware clipping created vertices. The coefficient-aware multiply is reachable
from normal input lowering and uses ordinary register allocation; it replaces
that workaround. No captured shader sequence or fixed register assignment is
part of the compiler.

The driver generates the coefficient table and the related vertex/fragment
state from the linked count. In particular, bind0 +0x44 is the user scalar
count; it was previously hardcoded to three. The VDM output count includes
position, while coefficient counts include interpolated 1/W.

## Hardware checks

Reset and chainload m1n1 before **each** invocation, then run from the Asahi
workspace:

```sh
mesa-m1n1-shim/src/asahi/drm-shim/scenes/varyings/run.sh NEW_OUTPUT_DIRECTORY nine
```

Modes: `seven`, `eight`, `nine`, `twelve`, `perspective`, `clipped`, `zero`,
`depth-clipped`, `procedural`.
Each checks all pixels of two 512x512 hardware attachments against an
independent CPU reference and requires identical attachments. The indexed
quad reverses triangle submission order in the second frame. `perspective`
uses W=0.625 and 1.375 while preserving screen geometry. `clipped` also extends
the quad outside the viewport. `depth-clipped` crosses the near and far
clip planes with unequal W and both constant and nonconstant components, ensuring
actual hardware clipping rather than relying on the rasterizer guard band. `procedural` generates
the equivalent quad from vertex ID without VBOs or UBOs; its two frames repeat
the same draw. Cross-stage optimization may remove its constant outputs.

The nine-component test uses independently computed color, position-derived,
and quadratic vectors, declared in a different order in FS. Twelve adds a
fourth component to every vector. Seven/eight cover the coefficient allocation
boundary, and zero checks shaders with no user varyings. The compiler tests
also cover sparse locations, component holes, array offsets, missing producer
components, different producer layouts for the same FS, and the capacity
rejection.

## External runtime

The current opaque preload is outside Mesa:

`tmp/agx-apple9/render_buffers_varyings12_launch.bin`

SHA-256: `0d7c5eac10ecbd282c96f04785b3d7475274f4cdf8f2b26ed2ae1a4258fb3f7d`.
Its VS half is the complete 0xc0-byte launcher captured from our twelve-varying,
four-buffer Metal probe in EXP-M4-59. Its FS half is the unchanged EXP-M4-58
four-buffer launcher. The old VS launcher lost an output when all sixteen
publication registers were occupied. Neither launcher contains our API shader
main, and neither was disassembled or decompiled. Generated shader mains and
all linkage tables remain independently authored.

`tmp/agx-re/experiments/EXP-M4-59-varyings/extract_launcher.py` reproduces it
from the trusted `twelve.pkl.gz` capture and the retained
`tmp/agx-apple9/render_buffers_launch.bin`. The old blob is preserved.

Only pixel-center smooth FP32 interpolation is supported here. Flat/integer,
centroid/sample, and noperspective shader inputs remain unsupported. Graphics
control flow, textures, multiple render targets, and arbitrary raster state
are separate work. The former four-buffer budget is superseded by the shader-loaded address tables
documented in `../buffers/README.md`. Interfaces larger than four components
also use the draw arena when they have no buffers.

Validation on T8132, 2026-09-05: all eight cases pass both hardware frames with
no coverage or color failures (at most one channel byte of rounding error).
The indexed quad, depth-tested cube, 100-triangle scene and 2,476-triangle
island also pass: 24 hardware frames in total. All 190 compiler tests pass.
The complete reports are retained in EXP-M4-59 `VALIDATION.json`.

The clipping correction is validated by the nine-mode suite, including explicit
near/far-plane clipping, and by 16 sunset frames at 1024×1024 with enlarged
water triangles crossing the view volume. Source-correlated encoding evidence
and the before/after results are recorded in `tmp/apple9-render-bugs/RESULTS.md`.
The suite has 191 passing compiler tests after adding the projective-multiply
operand/encoding check.


## Larger interfaces and interpolation (2026-09-07)

New `run.sh` modes: `16`, `24`, `32`, `linear`, `flat`, `mixed-clipped`, and
`integer`. Each runs two frames with reversed primitive submission order and
checks the raw hardware attachment against the independent CPU rasterizer.
Larger interfaces use unequal W; the mixed case clips geometry while combining
all three interpolation modes. Integer shaders compare the complete 32-bit
payload against distinct expected values, including values with the sign bit
set and overflowing unsigned additions.

The center-linear GLSL fixtures explicitly enable the noperspective extension
for the development test; this does not advertise untested centroid/sample
interpolation. The compiler also accepts NIR-generated noperspective inputs.

Flat coefficient reads return an asynchronous three-word tuple. Its third word
is the unmodified provoking-vertex value. The compiler allocates a scoreboard
slot and consumes it through the common handoff path, keeping unused tuple lanes
live until completion. A synchronous interpretation produced intermittent
missing channels and failed integer values; merely changing the slot byte was
not a valid fix.

The runtime requires `render_buffers_varyings32_launch.bin` in the external
Apple9 blob directory (448 bytes, SHA256
`2cdfb0e078dfab8eb011ed0af9aff5ecdf81ead6db082dce40b81a57a79099cb`). It retains
an entire opaque 256-byte VS setup from our authored 32-scalar Metal probe and
the existing entire 192-byte FS setup. No helper code is reconstructed in Mesa.
The authored-main call moves to +0x54 in the VS setup; FS offsets are unchanged.

Export results still remain live until completion. Reusable publication
registers, more than 32 user scalars, and centroid/sample interpolation remain
future work. Detailed artifacts: `tmp/apple9-exports/RESULTS.md` in the Asahi
workspace.

Final validation: 18 hardware frames pass across the seven new varying modes
and the existing sampled-texture and buffer-table regressions. No coverage or
color errors; maximum interpolation rounding difference is one channel byte.
All 211 compiler tests pass. Raw reports are in
`tmp/apple9-exports/VALIDATION.json`.

## Export lifetime investigation (2026-09-07)

Later Metal and hardware tests distinguish publication contents from ordinary
GPR contents. Overwriting the same-number ordinary GPR **before** VARY_STORE
preserves the exact image; overwriting the publication destroys it. Metal also
writes ordinary GPRs while same-number publications remain live. Consequently,
the previous shared allocation/live-out model was unnecessarily conservative.
The allocator now assigns publication indices independently of ordinary GPRs.
FLOAT2_EXPORT and LOGIC_EXPORT define publication values; VARY_STORE consumes
those values. Namespace mismatches are rejected. Every publication retains a
unique index through completion, without occupying the same-number GPR.
Trace output distinguishes `pN` from `rN` and reports publication count.

The 32-component VS now peaks at 17 allocator-counted live GPRs, down from 42.
This is a compiler liveness improvement, not a measurement of physical hardware
occupancy. The existing 32-user-component interface limit is unchanged.
Allocator regression tests cover 64 simultaneous publications with one ordinary
GPR, publication exhaustion, overlapping publication slots, and namespace
mismatches. Validation results: `tmp/apple9-publication-allocator/RESULTS.md`.

Reusing publication storage immediately after VARY_STORE still fails. A delayed
overwrite can pass, which demonstrates asynchronous consumption rather than a
safe release rule. Fixed delays are not a valid workaround. Full evidence and
negative controls: `tmp/apple9-export-lifetime/RESULTS.md` in the Asahi workspace.
