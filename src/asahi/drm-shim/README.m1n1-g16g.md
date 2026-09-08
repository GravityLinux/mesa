# Apple9 compute through the m1n1 DRM shim

This development tree runs Mesa's Asahi Gallium driver against a real T8132
through the modern Asahi DRM UAPI. The preload library presents a render node,
embeds the m1n1 Python backend, and forwards device parameters, VM/GEM
operations, queues, synchronization, and submissions to the source-built G16
firmware backend.

Mesa now compiles both compute and a bounded vertex/fragment subset through
semantic Apple9 IR. The graphics path supports procedural vertex-ID geometry,
FP32 vertex elements, position and up to 32 user components across
multiple locations, smooth/linear/flat interpolation, integer flat inputs,
arithmetic, indexed triangles,
2D fragment sampling with filtering/mipmaps and structured control flow,
render-to-texture, and one packed RGBA8 color output.
See [sampled textures](scenes/textures/README.md) and [varying linkage](scenes/varyings/README.md). Unsupported graphics inputs fail compilation.

After the normal reset and m1n1 chainload, the hardware triangle gate is:

```sh
T8132_GLES_PROCEDURAL=1 T8132_GLES_PASSTHROUGH=1 \
src/asahi/drm-shim/run_t8132_gles_triangle.sh
```

Both shader mains are generated from GLSL/NIR; the original carrier mains are
replaced with STOP. Launchers, helpers, and fixed color load/store programs are
still opaque external assets from our own-source Metal captures. Their archive
offsets remain fixed, while generated mains are appended and selected through
explicit entry calls. Mesa supplies the full 4 MiB fixed-USC view with separate
executable/read-only and GPU-writable resource mappings; the shim does not
supply missing client pages.

The 512x512 standalone draw completed TA and fragment work for two frames.
Independent CPU analysis of the uncompressed attachment found exactly 82,620
covered pixels, no coverage mismatches, and at most one byte of color error.
Four additional Metal-hosted tests cover changed geometry, nontrivial fragment
arithmetic, and unequal vertex W, with both mains replaced in full.

This remains a graphics bring-up gate. Other interpolation modes, arbitrary
raster/attachment state remain limited. Structured graphics control flow and
RGBA clears/reloads are now implemented; see the later milestones below.
`glReadPixels` is not part of this gate. Readback validation used the shim's synchronized attachment bytes.

The [Tidal prism scene](scenes/tidal-prism/README.md) exercises a much larger
fragment program with sine, floor, smoothstep, procedural color, and unequal
vertex W. Its two hardware frames match an independent coverage/color oracle
within one channel byte. Shader files can be selected using
`T8132_GLES_VERTEX_SOURCE` and `T8132_GLES_FRAGMENT_SOURCE`; the optional
`G16G_RENDER_ATTACHMENT_DUMP` directory receives raw synchronized attachments
from the shim without altering the submission.


## G16G render synchronization

Normal render submissions compare BO contents with the last synchronized host
snapshot and send only changed bytes. Uploads larger than 64 KiB use m1n1's
compressed transfer helper. Unchanged atlases and GPU-written surfaces stay on
the target. Partial uploads invalidate CPU cache lines before writing, preserving
adjacent GPU results. Readbacks update the host snapshot so those GPU results
are not mistaken for CPU edits on the next draw.

Render readback defaults off, matching the original m1n1 shim. Set
`ASAHI_SHIM_PULL=1` to download the listed fragment attachments after completion;
`G16G_RENDER_ATTACHMENT_DUMP=DIR` also requests this attachment readback.
`ASAHI_SHIM_PULL=all` retains the slower diagnostic synchronization of every
writable binding. Timestamps and fences are synchronized independently. Compute
keeps its existing writable-buffer readback for CPU result consumers.

`G16G_SYNC_TRACE=1` logs upload bytes, write calls, compressed calls, upload time,
and the readback policy/byte count. Normal HDMI presentation no longer needs
`G16G_RENDER_PRESENT_ONLY=1` to avoid redundant transfers.

Content comparison is not mmap write-fault tracking: an unchanged host value
written over an unread GPU result cannot be detected. CPU consumers of render
results must request the appropriate readback (`all` when attachments do not
cover those results). This remains an experimental shim, not general shared
CPU/GPU memory coherence.

## Current compute model

- T8132 is exposed as G16G and T8140 as G17P. Apple9 has a distinct shader-ISA
  key and is not silently compiled as Apple8.
- G16 and G17 use the same fixed 1-TiB USC base, `0x10000000000`. The former
  relocatable G17 behavior was a shim mistake, not a hardware capability.
- GLSL compute shaders travel through Gallium and NIR into a semantic Apple9
  VIR, register allocator, scheduler, and machine encoder. Accepted shaders do
  not select a captured whole-main byte sequence.
- Scoreboard slots are allocated as compiler state. Pending asynchronous
  producers carry a slot; compatible consumers name that slot and release or
  retain it according to liveness. Ordinary ALU results are ordinary GPR
  values, not a second provenance system.
- The allocator supports the measured r0-r63 bank, scalar and adjacent tuple
  classes, copies around constrained instructions, and last-use release.
  Independent vector lanes are joined by a pre-allocation `COLLECT` pseudo,
  coalesced when already adjacent, and otherwise copied after allocation;
  vector stores never borrow an untracked scratch tuple after allocation.
  Spilling is not implemented and excess pressure fails compilation.
- Gallium owns the compute archive, compiler state, resource records, carrier
  installation, CDM command, BOs, VM bindings, and ordinary
  `drm_asahi_cmd_compute` submission.
- One eight-buffer superset carrier serves ordinary compute, while a second
  own-source eight-buffer carrier supplies the independently validated device-
  atomic launch contract. A compiled main may use any prefix of one through
  eight resources in either ABI; resource count does not select a launcher.
- The shared fixed-USC archive is append-only between installations. A
  screen-wide timeline serializes physical ownership changes, while each
  logical DRM VM keeps its own persistent root in the m1n1 backend.

The supported NIR surface is intentionally limited. It includes arbitrary
u32 constants; integer add/subtract, negate, multiply, AND/OR/XOR/NOT, shifts,
signed and unsigned min/max; core float arithmetic, FP32 reciprocal, reciprocal
square root, exp2, log2, floor, ceil, truncation, round-to-even, sqrt, sin, cos,
and FMA; comparisons, straight-line select, recursively nested structured
if/else, and general structured loops with loop-carried values, `break`,
`continue`, and nesting within the six encoded predicate scratch banks;
scalar/vector device loads and stores;
general constant, affine, and runtime buffer indexing; 8/16/32-bit memory
formats; and the measured system values and dense dispatch geometries used by
the fixtures. Scalar 32-bit SSBO atomics include integer add/subtract,
AND/OR/XOR, signed and unsigned min/max, exchange, compare-exchange, and the
NIR floating-add operation. Returned and discarded forms, dynamic addresses,
contention, and masked or looped atomics all use semantic VIR and allocated
registers. A final-use scalar device-load result may feed a returning atomic
directly through its allocated scoreboard slot. Live-after values, pending
indices, compare-exchange tuples, and discarded atomics remain on the
materialized path until their destructive-input contracts are proven.

The control-flow emitter recursively walks structured NIR and serializes its
implicit execution-mask stack. Every pending asynchronous load or atomic
return is copied to a durable GPR before its NIR block ends, so no pending
result crosses a nested push, else transition, or reconvergence pop. The load sequence's native
`HAS_NEXT` bit still follows linear issue order across those transitions, while
the formerly named `FIRST` bit is now modeled correctly as the raw `get_sr`
index address form. Arm-local stores use the native compare, execution-mask
push, else-transition, and pop sequence. Scalar and 1--4 component Boolean or
8/16/32-bit phis use the same compiler architecture as Apple8: register
allocation assigns one merge destination per component, and each predecessor
performs an ordinary bit-copy into that destination while its lane mask is
active. Merge destinations remain disjoint from every edge source, making each
edge's copy set parallel-copy safe without post-allocation scratch registers.
The merge pseudo itself emits no hardware instruction.
Native select formation remains a future optimization; keeping it disabled for
CFG phis makes this correctness-first slice exercise masked arm computation and
reconvergence directly.

Loops follow the public Apple8 compiler's structured architecture while using
Apple9's independently measured encodings. The emitter recursively walks the
complete NIR loop body; it does not recognize, clone, or rebuild a special
header/latch condition. Each structured break updates or unwinds the current
loop mask wherever it occurs, and one `JMP_EXEC_ANY` repeats the surviving
lanes. Semantic VIR operations represent loop-mask push/update/pop,
empty/active-lane branches, and nonlocal break unwinding. Branches name VIR
boundaries and are resolved to signed, start-relative byte displacements only
after all variable-sized pseudos have been emitted. Header and exit phis use
the same predecessor-edge merge architecture as `if` joins, and allocator
liveness is extended across every backedge. Source `continue` constructs are
lowered to structured masked latch paths; a `break` beneath nested divergent
scopes uses the native unwind record tied to its target loop's predicate level.
The exact hardware corpus covers zero-trip and bottom-tested loops, a break in
the middle of otherwise conditional loop work, compound conditions, general
and nested break, continue, three nested loop levels, vector recurrences, and
per-iteration device loads.

This early bring-up path deliberately does not implement robust buffer access.
The compiler does not infer resource access bounds, and dispatch does not
preflight a shader-derived byte span. Buffer indices are passed through to the
generated address calculation; callers are responsible for binding enough
storage for every access the shader can make.

Early return, general division/modulo, spilling, and unmeasured
package/resource forms reject rather than falling back to capture-assigned
registers or opaque native mains.

### FP32 math

The special functions use semantic VIR operations, allocated registers, and
the ordinary pending-producer dependency mask. Both pending and materialized
sources support retained and final uses. The SFU source-control byte (byte 6)
uses `0x90` for retain and `0xb0` for final use. Reciprocal has a distinct form
using `0x00` and `0x10`. Applying reciprocal's bit-4 rule to the ordinary SFU
selects `0xa0`, which reads a **bfloat16** source from the low 16 bits instead
of FP32. Those bits were zero in many early inputs. EXP-M4-54 confirms this
with matched native BF16/FP16/FP32 Metal shaders and packed-input mutations.
Ordinary-SFU source type occupies byte 6 bits [4:3], while reciprocal places
it in byte 7 bits [3:2]. The lifetime difference is part of distinct operand
layouts, independent of the scoreboard; no blanket source-copy workaround is
used. Ordinary-SFU source negation is byte 8 bit 0 and absolute value is byte 7
bit 7. Earlier positive-only rsqrt/log2 probes mistook negation for a NaN control.

`sqrt` uses the `0x2f` class-1 sqrt factor followed by a multiply. EXP-M4-55
establishes that this factor returns 1 for signed zero and positive infinity,
and the usual rsqrt factor for positive normal inputs. Multiplication by the
input supplies fast sqrt with signed-zero and infinity handling built in.
The tested arithmetic mode treats denormal SFU inputs and outputs as signed
zero. NaN results are checked by classification; payload propagation is not
promised. The source stays live across the factor until the multiply, using
the allocator's ordinary liveness handling.

Compute `sin` and `cos` use the `0x2f` class-3 factor, represented by the FP32 NIR
operation `fsin_factor_agx`: `sin(pi*x/2)/x` on [-1,1], with the pi/2 limit at
zero and NaN outside that interval. An independently constructed 256-bit
fixed-point expansion of 2/pi preserves quadrant information across the
finite FP32 range. The two-part reduced phase feeds factor-and-multiply
operations for sine and its complement for cosine. FMA corrections preserve
the phase residual and account for complement subtraction rounding. No Taylor
polynomial remains. This opcode evaluates reduced angles; it does not perform
range reduction or quadrant selection.

Tiny sine inputs and signed zero are preserved by bit selection; infinities
and NaNs return NaN. Full-range reduction still has substantial code and
register cost. Large combined expressions can exceed the current allocator's
no-spill register budget.

## External development inputs

The repository deliberately does not contain captured Metal package blobs.
During this bring-up phase, five recapturable inputs from two own-source
eight-buffer carriers are loaded from hardcoded paths under
`/home/nsheth/Projects/asahi/tmp/agx-apple9`:

```text
/home/nsheth/Projects/asahi/tmp/agx-apple9/carrier8/constant.bin
/home/nsheth/Projects/asahi/tmp/agx-apple9/carrier8/launch.bin
/home/nsheth/Projects/asahi/tmp/agx-apple9/carrier8/division.bin
/home/nsheth/Projects/asahi/tmp/agx-apple9/carrier8-atomic/constant.bin
/home/nsheth/Projects/asahi/tmp/agx-apple9/carrier8-atomic/launch.bin
```

The carrier resource record has three hidden entries followed by eight visible
buffer entries and a zero sentinel. For direct dispatch, q0 points to the
record's total-thread tuple and q1 to `{1,1,1}`. For indirect dispatch, q0 is
the caller's raw group-count pointer and q1 points to the record's local-size
scale tuple. The unchanged launch program derives the common execution
geometry from that tagged representation. The third hidden entry points to the
relocated 8-KiB integer-division helper table in `division.bin`.
Unused visible entries are padded with a valid mapped pointer that the compiled
main cannot reference. The full one-through-eight prefix has passed exact
hardware output and input/guard-preservation checks through ordinary GLSL,
Gallium, the DRM UAPI, and the G16 shim.

The atomic carrier was captured from the own-source eight-buffer workload in
`EXP-M4-48-atomic-package`. It publishes the eight caller buffers directly at
q0--q7, with no hidden geometry or division-table resources. Mesa replaces its
own-source stage main with the compiled main and patches the launch/archive and
resource pointers semantically. Direct dispatch, fixed or runtime local size,
and all ordinary compute system registers are supported. This capture does not
establish an indirect-dispatch launcher or a `gl_NumWorkGroups` hidden-resource
contract; those combinations reject instead of reading caller buffers as
metadata. The two carrier families are loaded independently, so missing atomic
development inputs do not disable ordinary compute and vice versa.

File lengths are taken from the files. The driver validates only the regions
it actually installs or patches; it does not require an exact whole-file size.

Two optional, currently non-gated render research inputs use the same
directory:

```text
/home/nsheth/Projects/asahi/tmp/agx-apple9/g16_render_package.bin.zst
/home/nsheth/Projects/asahi/tmp/agx-apple9/render_interleaved_vbo_launch.bin
```

They do not restore the removed render compiler. They are retained outside Git
only so future packaging work can reuse the current semantic relocation and
archive experiments without putting opaque Metal data in repository history.

The current local T8132 input hashes are:

```text
a3586e009bd675feb6b67d72b6f8b9500bde15487584c1a054925ac9af2d75ce  carrier8/constant.bin
62e85e9dd6cca4dd033cb101ad860da28934d8f92d7498d7bd11b42eff0957c3  carrier8/launch.bin
fbb72c3f6ffb8e4a2fd17c9155d5ae32d7e704eeb5c0c906bef6d315c7299e80  carrier8/division.bin
9baa760c5185b9e5645bd1299e5ec948674258d6cbb0dc68b1394f1e45f3fd27  carrier8-atomic/constant.bin
f712c5923161763e175403a715a66e4959239298e3905ce09d03bfe0e026d2ec  carrier8-atomic/launch.bin
9c7912148f4d4b48b59ba8e720e9dc94d0988f191294394af79849b21fb99cfe  g16_render_package.bin.zst
da8e9c9df75305fb8d11cd8d468e8cf0f35bb5172e7465777b6d258f243b145b  render_interleaved_vbo_launch.bin
```

The compute files are development inputs, not stable ABI. Recapture notes
should record their source workload, OS/build, package role, and hash outside
the Mesa repository until their remaining fields are replaced by semantic
builders.

## Build

The Asahi driver needs Mesa's generated internal CLC programs. On Arch-based
hosts the additional package is `spirv-llvm-translator`.

```sh
meson setup /home/nsheth/Projects/asahi/mesa-m1n1-build \
  --prefix /home/nsheth/Projects/asahi/mesa-m1n1-install/usr/local \
  -Dgallium-drivers=asahi \
  -Degl=enabled \
  -Dplatforms=[] \
  -Degl-native-platform=surfaceless \
  -Dgles1=disabled \
  -Dgles2=enabled \
  -Dglx=disabled \
  -Dgbm=disabled \
  -Dvulkan-drivers=[] \
  -Dllvm=enabled \
  -Dbuild-tests=true \
  -Dtools=drm-shim

ninja -C /home/nsheth/Projects/asahi/mesa-m1n1-build
```

Useful host-only gates are:

```sh
meson test -C /home/nsheth/Projects/asahi/mesa-m1n1-build \
  agx_tests \
  --print-errorlogs

/home/nsheth/Projects/asahi/mesa-m1n1-build/src/asahi/drm-shim/\
t8132_apple9_compute_runner --list
```

## Run on T8132

Build and chainload the matching `m1n1-m4-agx` tree first, leaving the target
at its proxy prompt with a cold GPU ASC. The native compute tests enter through
ordinary GLES 3.1, the normal Mesa driver, and the DRM UAPI; they do not
construct NIR or m1n1 work records directly.

Run the default supported compute regression set directly in one process and
EGL context:

```sh
src/asahi/drm-shim/piglit/run.sh direct
```

The default one-boot sequence omits `archive-cross-program-sequence`,
`program-lifecycle-stress`, the math matrix, and the 24-program
`two-source-comparisons` matrix
to reduce pressure on the temporary append-only 64 KiB shader archive. As the
ordinary compiler corpus grows, a complete default run can still reach that
temporary limit; run the named control-flow and lifecycle batches separately
after fresh chainloads when it does. Archive exhaustion is a packaging-harness
limit, not a shader result. The two lifecycle cases test program-publication
lifetime; the comparison matrix is the exhaustive predicate correctness gate.

Or record each named case as a Piglit subtest while retaining the same
single-process execution model:

```sh
src/asahi/drm-shim/piglit/run.sh piglit
```

Success means exact complete-buffer comparison, not merely command retirement.
The native cases also verify immutable inputs, poison-filled gaps, leading and
trailing guards, multiple bindings and dependent addresses. Dedicated cases
exercise repeated range dispatch, program publication/retirement pressure, and
ordered reuse of independently compiled programs.
`simple-divergent-if-else` checks unsigned, signed, and floating-point `<`,
`>=`, equality, inequality, and composed Boolean forms with mixed, all-true,
and all-false lane populations. Its floating-point inputs include NaNs and both
signed zeros, so `>=` is validated as an IEEE ordered comparison rather than a
naive inversion of `<`. Each of its 36 submissions verifies both predicated arm
stores and a post-reconvergence value merge across complete 16,384-word outputs,
while inactive outputs retain poison and surrounding guards remain unchanged.
`two-source-comparisons` independently computes both operands and covers every
canonical signed, unsigned, and floating-point equality/order relation,
including relations normalized by reversing operands. It runs mixed, all-true,
and all-false populations; the mixed floating-point population includes NaNs,
infinities, adjacent finite values, and both signed zeros. Additional short and
extended predicates keep neither, either, or both sources live after the
comparison. `compare-register-pressure` repeats a retained-source comparison
with 16 independently loaded values live across the region.
`single-region-shapes` covers empty, then-only, else-only, and two-sided arms,
with exact stores before, inside, and after the region. `multiple-phi-vectors`
checks a scalar phi and a four-component phi from ordinary GLSL through exact
scalar and vector-buffer outputs.
`branch-local-device-loads` uses seven SSBOs and three executions to cover
mixed, all-true, and all-false lane populations. It performs distinct affine
loads in both arms, fans each loaded value into an arm-local side effect and a
32-bit phi, then combines the merged value with a post-reconvergence load. All
three complete 16,384-word output buffers, inactive poison values, and guard
regions are checked exactly.
`nested-short-circuit-if-else` covers nested conditionals in both outer arms,
multi-level value reconvergence, sibling and asymmetric regions, and observable
right-hand-side stores for short-circuit AND and OR. Seven complete 16,384-word
outputs verify both final values and that skipped right-hand sides leave poison
unchanged. `nested-branch-local-loads` issues two distinct pending loads in
each of four nested leaf regions, resolves their values through both inner and
outer phis, and checks two complete guarded outputs. The two tests also pass
back-to-back in one process and persistent VM.
`reciprocal-direct`, `reciprocal-retain`, and `reciprocal-materialized` cover
pending-load handoff, source lifetime, and ordinary-GPR input paths with exact
FP32 output oracles. `reciprocal-denominators-1-1024` converts every integer
denominator in the permitted local-size domain with the compiler's ordinary
U2F and reciprocal path, then checks the returned FP32 values against
`abs(D * r - 1) <= 2^-18`. On T8132 the worst observed denominator is 981,
with residual `7.1362592279911041e-8`; no Newton refinement is required for
the ceiling-division lowering. `ceil-div-grid-domain` executes the compiler's
complete U2F/reciprocal/FMA/F2U/multiply/compare/correct sequence for eight
numerators at every denominator, including zero, remainder boundaries, and
the maximum legal `65535 * D`; all 8,192 exact integer outputs pass on T8132.

The separately runnable math cases are `sfu-rsqrt`, `sfu-sqrt`, `sfu-exp2`,
`sfu-log2`, `sfu-floor`, `sfu-ceil`, `sfu-trunc`, and `sfu-round-even`, plus a
`-materialized` variant of each. Every case checks 4,096 inputs: a retained
source, its result, a separate final-use pending input, and an ALU consumer
of the result, totaling 16,384 values plus immutable inputs and buffer guards.
Rounding functions require exact results; rsqrt/sqrt/exp2/log2 allow two ULPs
against host double-precision libm rounded to FP32. The ALU consumer is checked
against the independently validated rounded SFU result. Directed cases cover
signed zero, infinities, NaNs, denormals, exponent extremes, and rounding ties;
the remaining inputs are deterministic randomized values.

Run `math-sin`, `math-cos`, `math-sin-boundaries`, `math-cos-boundaries`,
`math-sin-reduced`, and `math-cos-reduced` as another batch after a fresh
chainload. They use the same buffer checks and
two-ULP numerical tolerance. Boundary cases sample neighboring FP32 values at
small multiples of pi/2 and at powers of two times pi/2 through exponent 127,
with both signs. Reduced cases densely sample [-pi,pi], exercising both factor
arguments and the small-input path. This is sampled numerical validation, not an exhaustive
two-ULP guarantee over all FP32 bit patterns. `--list` includes all math cases;
`--list-default` excludes them to keep archive-heavy batches separate.

`superset-1` through `superset-8` specifically compile distinct ordinary GLSL
programs and exercise every active-resource occupancy of the shared carrier.
`device-atomic-native-shape` reproduces the own-source eight-resource Metal
workload through the normal GLSL compiler and checks every target and four-word
result record. `device-atomics` checks discarded and returned forms, every
GLES-visible integer operation, compare-exchange success and failure, signed
and unsigned min/max, multiple retained results, two writable resources,
256-lane contention with a return-value permutation oracle, divergent loops
with discarded and returned atomics, 4,096 exact private-address recurrences,
and 32 consecutive 1,024-lane dispatches into the same counters. Every result
and final target is compared, rather than using command retirement as success.
`device-atomic-pending-load-forwarding` combines a directly forwarded final-use
load, a live-after load that must be materialized selectively, and an unrelated
pending load whose lifetime must not be drained by the atomic.

The indirect-dispatch geometry cases exercise the actual GLES API with CPU-
and GPU-authored records, nonzero offsets, zero dimensions, and asymmetric 2D
and 3D grids. `gl_NumWorkGroups` is compiled from the hidden q0/q1 package
contract and checked in every dimension for direct and indirect execution. It
does not consume a visible buffer binding or require CPU inspection of the
indirect record. Direct dispatch computes `ceil(global_threads / local_size)`
with the measured FP32 reciprocal sequence and one exact integer correction;
this covers short final workgroups and fixed or runtime local sizes. Indirect
dispatch returns the caller-provided group count directly.

## Bring-up boundary

This branch is an investigative compiler and packaging harness, not a complete
Mesa driver. The immediate productive direction is to keep expanding semantic
compute lowering and replace each external wrapper field as it becomes
understood. Render should resume from compiler-generated stage code and a
verified packaging model; the old capture-matched render compiler should not
be resurrected as a fallback.

### Graphics uniforms (EXP-M4-57)

The graphics compiler now supports one active UBO per stage, including Mesa's
ordinary default uniform block. The driver records each draw's VS/FS buffer
and range, retains the BOs, and publishes separate binding records while
sharing the shader binaries. The initial compatibility arena supports 32
uniform-bearing draws per batch. Multiple buffers in one stage and arena
exhaustion are rejected; this is not full graphics-resource support.

`scenes/uniforms/run.sh OUTPUT_DIRECTORY` tests matrix, tint, and time updates
through glUniform. Its `blocks` mode tests separate std140 UBOs at API bindings
4 and 7 with per-draw range rebinding. Both paths produce two exact hardware
frames with distinct transforms and colors; their output bytes match.

The additional opaque preload/data artifact remains outside Mesa:
`/home/nsheth/Projects/asahi/tmp/agx-apple9/render_uniform_launch.bin`
(SHA-256 `3840a61cb6cbaf17aae247766001a90663f493e2fa1e72791ca14c6e9d1ecafe`).
Its source, captures, and reproduction tool are under
`tmp/agx-re/experiments/EXP-M4-57-graphics-uniforms/` in the Asahi workspace.
It contains no Metal-generated API shader main.

### Vertex buffers, indices and depth (EXP-M4-58)

`scenes/mesh/` now exercises real FP32 vertex elements, u16/u32 index buffers,
VS transforms and FS tint uniforms, and Depth32Float testing/writes. Vertex
loads are normal compiler-generated device loads inside the API VS main;
there is no fixed-format Metal vertex-fetch main. Format, stride, and relative
attribute offsets are shader-key state; buffer addresses are retained per draw.
The compiler uses Gallium's compacted vertex-element indices independently of
GL attribute locations. `scenes/vertex-inputs/` additionally checks interleaved
FP32 positions, UNORM16 UVs, UNORM8 colors, constant attributes, and changing
VBOs. Multiple GL bindings of one resource share a hardware argument. Other
supported direct formats include signed normalized and signed/unsigned scaled
8/16-bit channels; unsupported formats use Mesa's vertex translation path.
Unused point-size writes are removed for this triangle-only path.

The graphics preload now publishes four pointers per stage. It lives outside
Mesa as `render_buffers_launch.bin` (SHA-256
`2c2c906598a3993ef31999a6d9dfd004f598eb660dd0c106be095f339baad67c`),
extracted as opaque records from the authored EXP-M4-58 Metal probe. This
supersedes the one-buffer EXP-M4-57 preload. Its known archive selector is at
+0x36; it does not use that older preload's separate constant-state reference.

Indexed command words use a 32-bit offset into the fixed USC aperture, so
index-buffer resources are allocated there. Draw-local PPP records supply
depth function/write state. GLES clip depth is converted to [0,w] in the VS;
caller depth attachment addresses, clears and stores already pass through the
existing render UAPI/shim path.

See `scenes/mesh/README.md` for reproduction and limits. The four-buffer u32
quad, depth-state controls and rotating shared-vertex cube pass pixel checks.
The earlier uniform and 100-triangle scenes retain their original hashes.

### Shader-loaded buffer address tables

The four-direct-buffer graphics limit is superseded by
[shader-loaded address tables](scenes/buffers/README.md). Each stage receives one
root pointer and supports up to 32 live table entries. Hardware tests exercise
32 VS resources and 16 FS resources together, including per-draw rebinding.
The existing opaque launchers are unchanged; generated mains perform the pointer
loads and indirect data accesses. API uniform-block limits remain separate.

### Generalized VS/FS linkage (EXP-M4-59)

See [the varying tests](scenes/varyings/README.md) for the scalar/component map,
coefficient-aware perspective interpolation contract, and reproduction. The
September 7 expansion supports 32 user scalars and mixed smooth, center-linear,
and flat inputs. Integer flat values use bit-preserving exports and asynchronous
coefficient reads integrated with the shared scoreboard allocator.

The runtime preload is `render_buffers_varyings32_launch.bin`: a complete opaque
256-byte VS launcher from the authored larger-interface probe, followed by the
existing 192-byte FS launcher. Earlier launchers remain historical artifacts.
Linkage, shader keys, and per-draw coefficient/state tables are generated from
the shader interface. See `tmp/apple9-exports/RESULTS.md` in the Asahi workspace
for clean-room provenance, before/after failures, and hardware reports.

### Multiple graphics pipelines in one batch

Shader binding no longer flushes an Apple9 render batch. Each draw retains
its source package and has independent launchers, buffer bindings, coefficient
layout and fixed-function state. Submission installs the complete shader set
before publishing those records; archive rollover cannot invalidate an earlier
draw in the same batch. The existing arena now holds up to 32 graphics draws,
including draws without uniforms.

`scenes/pipelines/run.sh OUTPUT_DIRECTORY` checks an A/B/A sequence extended
with another pipeline, changing both stages and varying counts while preserving
color and depth. Its `arena`, `rollover`, and `pin` modes test all 32 draw slots,
a constrained archive rebuild, and 20 simultaneously retained pipelines.
See [the fixture documentation](scenes/pipelines/README.md) for validation and
remaining limits.

### RGBA8 blending and write masks

The Apple9 fragment compiler now lowers ADD blending with ZERO, ONE, SRC_ALPHA
and ONE_MINUS_SRC_ALPHA factors, including separate RGB/alpha factors and channel
write masks. Destination reads use allocated scoreboard slots and the draw's
PPP tile read/modify/write state. Unsupported blend equations/factors fail
compilation. See [the blending fixture](scenes/blending/README.md) for pixel/depth
validation.

### Graphics control flow and 2D canvas groundwork

Both graphics stages now use the common branch/loop execution-mask compiler.
See [control-flow tests](scenes/control-flow/README.md) for divergent/nested loops,
zero iterations, break/continue and blending. ESSL gl_FragColor is accepted as RT0.

RGBA clear colors and attachment reloads now work. Clears without application
draws use the ordinary Mesa clear rectangle, with triangle fans/strips converted
to indexed lists. See [canvas tests](scenes/canvases/README.md) for clear-only,
color-only, two-target switching, 1024×1024 and batch-boundary coverage.

Fragment screen XY now uses native integer pixel system registers and Mesa's
API coordinate conversion. Per-draw viewport transforms follow Gallium, so raw
FBO readbacks now have bottom-left rows. See the [screen-coordinate tests](scenes/fragcoord/README.md).
Single-viewport scissoring now uses per-draw pixel rectangles, intersected
with the framebuffer and viewport; see [scissor tests](scenes/scissor/README.md).
FP32 `mix` lowers through NIR to supported arithmetic. Window Z/W remains
unsupported. Sampling/render-then-sample and expanded argument bindings are now
implemented.

Apple9 graphics frontends preserve structured loops instead of inheriting
Apple8's 32-iteration unroll threshold. This avoids expanding nested sampling
loops into large static instruction sequences. Standard scalar/algebraic NIR
cleanup before graphics-specific expansion also removes avoidable vector math.
Compute's frontend unroll policy is unchanged.

The compatibility render archive now uses the zero-filled reservation through
`+0x18000`, immediately before compiler state, instead of stopping at `+0x10000`.
The opaque helper closure stays at its original offsets. This increases dynamic
stage capacity from 14,016 to 46,784 bytes (including per-stage block overhead).
Compact calls preserve the validated next address bit; the `wide` pipeline test
checks shaders with entry points above 64 KiB against an independent pixel/depth
oracle. Batch planning checks both storage and call encodability before any
per-draw calls are published. Draw admission uses that same plan: when adding
another pipeline would exceed capacity, Mesa submits the existing draws and
retries the new draw in a fresh batch with attachment reloads. Shared stage
blocks count once; API draw statistics are not repeated on retry. The `split`
pipeline test covers indexed draws across this boundary. A single pipeline
must still fit in the bounded compatibility archive.

Graphics sine/cosine now uses compact FP32 phase reduction rather than the
full-range multiword reducer. See [graphics trig tests](scenes/trig/README.md)
for measured error versus input magnitude. Compute retains the full-range path
as a compatibility choice; it is not a general compute-stage requirement.

### Application-controlled HDMI presentation

The optional `G16G_RENDER_PRESENT_ON_SWAP=1` mode presents the actual surfaceless or
device-platform EGL pbuffer at an application swap, rather than attachment 0 after every GPU
submission. It uses the live Gallium resource and its current DRM-file-scoped
binding; offscreen rendering and archive splits do not produce display updates.
The [presentation fixture](scenes/present/README.md) documents the SDL/LÖVE
adapter, limits, and HDMI checks for resource identity, lifetime and resize.
Buffer upload behavior is unchanged by this presentation feature.
