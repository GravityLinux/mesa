# Apple9 native mesh compiler development path

`agx_compile_apple9_native_mesh(nir, enable_native_mesh, ...)` compiles a complete
mesh program through the ordinary NIR, VIR, allocation, scheduling and packing
pipeline. The gate is disabled unless explicitly enabled. It is a compiler
entry point, not a Vulkan native draw implementation or an advertised feature.
The existing compute-emulated Vulkan mesh path remains the default.

The accepted development class is a fixed 16x1x1 workgroup, triangle topology,
maxima V3/P1, one constant `SetMeshOutputs(3, 1)`, Position only, and no task
payload, user shared memory, culling or primitive attributes. Ordinary formation
handles the established finite, nondegenerate, inside-clip triangle record
class. This gate does not claim clipping or general mesh-shader support. The
hardware validation launch must use one workgroup while workgroup scope is
being qualified.

The normal IO and ordinary record formation prerequisites were adapted from
our authored compiler sources at `7779b8644dc355ee214cf64f4996254711e10c55`.
They contain no ordinary native helper code. The new completion lowering does
not call the research linker, append executable fragments, reserve physical
GPRs, or require a diagnostic descriptor. Source publication and the shared
output layout supply its operands. The object-grid export has separate
provenance: on 2026-10-14 the user explicitly approved decoding the complete
11-instruction **object** helper and said “implement our own equivalent”.
`agx_nir_apple9_object_grid_export` expresses that grid publication through
ordinary NIR. This approval does not cover ordinary mesh helper algorithms.

## Operations and allocation

There are two six-byte hardware families, represented by three compiler ops:

| Semantic operation | Inputs | Effects |
| --- | --- | --- |
| `mesh_control_apple9_agx` | primitive count and provisional B, each 16 bits; control42 | Reads and clears the selected count halfword; mesh/memory ordering |
| `mesh_submit_channel_apple9_agx` | channel selector and local plane byte offset, each 32 bits | Consumes an adjacent register pair; mesh/memory ordering |
| `mesh_submit_scalar_apple9_agx` | actual vertex count, 32 bits | Scalar mesh submission; mesh/memory ordering |

All are side effects and are neither eliminable nor reorderable NIR operations.
VIR models source uses and destructive operands explicitly. A count operand
owns a whole word conservatively, protecting any separately live companion
half by copying before a destructive use. B is also conservatively clobbered
until hardware qualifies preservation; the live vertex count gets a copy. The channel uses the existing tuple
and COLLECT machinery. RA chooses all physical operands. The packer derives
halfword descriptors as `2 * register + half`, and checks pair adjacency.
Completion scheduling drains pending operations before these side effects.

The known class issues control in all 16 lanes with A = the published primitive
count and B = the published vertex count. Channels 0..3 run in lanes below the
actual vertex count; their offsets come from the Position component map and
pitch. The scalar submission runs in lane 0. Device fences preserve the
qualified publication ordering. No marker or frame-padding instruction is
inserted.

A's primitive-count meaning and read-and-clear are qualified for 0, 1 and 2
primitives. B's general meaning and control42 remain unresolved. The known
class uses B = actual vertex count and control42 = 1, matching the successful
H1 combined-input arm. The channel release bit is set; allocation conservatively
clobbers the pair. Scalar release remains zero, but its source is also
conservatively clobbered because preservation is unqualified. Mode 1 (culling) is decoded/encoded for census coverage but
is not emitted by the development NIR path. General lane/workgroup scope,
culling and broader output families still need hardware evidence.

## Native argument and draw contract

The compiler reports `agx_apple9_mesh_body_info` and sets
`agx_shader_info.apple9_body_only`. This flag means a native launch contract is
still required, even though mesh completion is now present.
`hk_upload_shader` rejects this flag; generic compute or graphics submission
cannot silently run the program. No driver path selects the new API yet.
Object entry/completion, reservation and launch integration are the next step.

The private argument ABI is version 6:

| Uniform words | Meaning |
| --- | --- |
| 0..11 | Existing graphics roots |
| 12..13 | Mesh local-output root (selector 6) |
| 14..15 | Object grid root (selector 7) |
| 16..17 | Task payload pointer; unused by this class |
| 18..19 | Pointer to `agx_apple9_mesh_draw` |

`agx_apple9_mesh_draw` is an array of 23 little-endian 64-bit slots, in
`agx_apple9_mesh_formation_input` enum order. Address slots use all 64 bits;
other slots use the low 32 bits. It contains current geometry/list/root/scene
reservations, allocation IDs, fragment and raster state, viewport values, and
reservation readiness, and an optional device-u32 outside-class counter address
(slot 22, byte offset 176). Zero disables counting. The complete draw is 184 bytes.
The counter increments once per rejected triangle at the first invalid-index,
clip-domain, or screen-range predicate, by lane zero; readiness-zero and
zero-area rejection do not increment it. The clip predicate remains strict
`abs(x) < w && abs(y) < w`, so `x = ±1, w = 1` is a counted class miss.
Before any CTS subset, the runtime/harness must bind a zeroed owned counter,
read it after completion, and report class drops separately from native failures.
The runtime owns this read-only allocation through
completion. These are ordinary device loads. There are no baked GPU addresses,
kernel pool aliases, Q addresses or application descriptor assumptions.
The kernel UAPI is unchanged. The compiler preserves all 20 argument words
when choosing preamble publication space. When `HK_APPLE9_NATIVE_MESH` is enabled,
Honeykrisp hashes the ABI version, draw size and shared selector/layout key into
its compiler/device cache identity and mesh/task graphics-state cache key.
Stage-specific output maps and capacities follow source NIR, which already keys
shader compilation; layout algorithm changes require an ABI version bump.

Local layout reports the primitive count, compiler-owned vertex count,
Position planes and total output bytes. For this class the count offsets are
16 and 20, Position offsets are 32/48/64/80 and local output is 96 bytes. These
are results of layout construction, not physical-register assignments.

## Host checks

Build tools must come from this source tree: adding NIR intrinsics changes
serialized library NIR. Configure with `-Dllvm=enabled -Dmesa-clc=enabled
-Dprecomp-compiler=enabled -Dbuild-tests=true -Db_ndebug=false`, then build
`src/asahi/compiler/agx_tests` and `src/asahi/compiler/apple9-native-mesh`.

```
<build>/src/asahi/compiler/agx_tests --gtest_filter='Apple9Mesh*'
meson test -C <build> apple9-native-mesh --print-errorlogs
```

The five gtests cover all 58 singleton census encodings, field rejection,
low/high halfwords, adjacent tuples, live values across destructive uses,
0/1/2 primitive counts, allocation changes under pressure, actual spills and
count-only side effects with pending loads.

The full-program test executes 26 authored source/state fixtures at two
placements, plus readiness-zero, 24 live values across completion, and both
strict X clip boundaries with counting enabled/disabled and readiness-zero:
60 complete programs. The pressured program must have genuinely different
mesh unit register allocations. The independent DATA reader round-trips
positions to source-derived quantized coordinates, checks raster state and
record references, and compares software coverage masks. It accepts an
alternative all-raw encoding, so compression choices are not an oracle. It
rejects corrupted coordinates, references and padding. Each H1 source covers
512 software samples. The old byte-exact v0 DATA comparisons are retained only
as research evidence outside the build (and in commit `151c366e0b3`).

The wrapper reads the generic ASM listing at emitted instruction boundaries,
independently decodes all six units, follows allocated input definitions and
checks execution masks. It rejects reads of either U0 operand before
redefinition. Six deliberately misallocated or swapped-tuple mutations across
the normal and pressured programs must fail. The executable also checks
untouched sentinels, unit inputs/lane masks, a disabled gate and unsupported
count, workgroup, payload and capacity cases. Source provenance is recorded in
`test/mesh-source-fixtures.json`.
This is a compositional host comparison, not an emulator of the entire GPU
instruction set or a hardware rendering result. GPU record visibility,
retirement and pixels must still be checked against H1 on the mini. Compare
pixels to the authored-source reference and DATA through the decoder; do not
require v0's exact mode/base choices. The current formatter still needs a
looped bit writer or precompiled routine and explicit code-size/GPR budgets
before production enablement (roughly 44 KiB / 90 GPRs at this milestone).

For an external hardware harness, the test executable can export the complete
BASE/SHIFT_X binaries and their compiler metadata without modifying them:

```
mkdir -p <export-dir>
<build>/src/asahi/compiler/apple9-native-mesh --export <export-dir>
```

The exported files require a native launch. They are test artifacts, not an
asset-loader path in the driver. The metadata reports actual GPR/private-frame
usage; it does not insert H1's experimental 64-byte frame override.
