# Apple9 native mesh compiler development path

`agx_compile_apple9_native_mesh(nir, enable_native_mesh, ...)` compiles a complete
mesh program through the ordinary NIR, VIR, allocation, scheduling and packing
pipeline. The gate is disabled unless explicitly enabled. It is a compiler
entry point, not a Vulkan native draw implementation or an advertised feature.
The existing compute-emulated Vulkan mesh path remains the default.

The accepted development class is a fixed API workgroup with 1..16 invocations
(any 3D shape), launched at physical 16x1x1, triangle topology,
maxima V3/P1, one constant `SetMeshOutputs(3, 1)`, Position only, and no task
payload, user shared memory, culling or primitive attributes. Position retains
all four components, including non-unit W. Hardware clipping is unqualified;
the compiler does not perform a software clip-domain drop. The
hardware validation launch must use one workgroup while workgroup scope is
being qualified.

## API invocations and physical completion

The native entrypoints set `key.physical_threads = 16` and use the existing
`agx_nir_mask_apple9_mesh_body` NIR pass. The application region executes under
`local_invocation_index < product(declared_size)`. This excludes extra lanes
from application loads, stores, atomics, output writes and subgroup votes.
Local IDs are reconstructed using the declared dimensions, and WorkgroupSize
and NumSubgroups are frozen before shader metadata changes to physical T16.
SubgroupSize remains the architectural 32, with one partial subgroup; it is
neither the API invocation count nor the physical launch count. API barriers
in an expanded one-subgroup body shrink to subgroup scope. Compiler-owned
completion rendezvous remain outside the application mask.

`agx_apple9_mesh_body_info.logical_size` retains the API dimensions, while the
compiled part's `workgroup_size` describes the physical launch. NIR owns this
separation so ordinary control-flow selection, optimization and RA see it;
there is no special VIR lane filter or fixed register allocation. The same
pass now handles flattening 3D groups whose total is already 16.

The generated taskless object is ordinary T1 task NIR and goes through
`agx_nir_lower_apple9_native_object`. That bounded entrypoint also accepts
ordinary 1..16-invocation task NIR with one top-level grid launch and no payload
or user shared memory. Only API lane 0 writes the three grid dimensions. The
physical T16 workgroup then reconverges before its collective call, whose
hardware export still uses lanes 0/1/2 for X/Y/Z. Masking that callee to the T1
API lane set would lose two dimensions. The driver still does not expose or
upload these object programs.

Physical launches above 16, variable API sizes, and requested API subgroup
sizes other than 32 reject explicitly. T32 would still be a single
architectural subgroup, but the complete native object/mesh contract has not
been qualified at that launch width. This work does not promote the older
body-only experiments at wider physical widths.

The normal IO prerequisites were adapted from
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

The private argument ABI is version 7:

| Uniform words | Meaning |
| --- | --- |
| 0..11 | Existing graphics roots |
| 12..13 | Mesh local-output root (selector 6) |
| 14..15 | Object grid root (selector 7) |
| 16..17 | Task payload pointer; unused by this class |
| 18..19 | Pointer to three uint32 mesh grid counts, read by the object stage |

Mesh completion consumes the local output directly. It receives no scene,
cursor, output-buffer address, resource ID, viewport state or reservation
readiness value. The explicit DATA formatter and its descriptor were removed
once the combined-publication and eight-scene arms established implicit
reservation for this class. Their authored research sources remain in the
external research archive, not in the compiler build.

The kernel owns allocation, admission and retirement. No kernel UAPI change
is required. The compiler preserves all 20 argument words when choosing
preamble publication space. Honeykrisp includes the native ABI version and
selector/layout key in the gated compiler and graphics-state cache identities.

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

The full-program test interprets ordinary NIR and checks all local Position
components and primitive indices against authored inputs, including the clip
probes. It checks the six completion operations and their lane masks, with no
global record writes. A pressure variant keeps 24 independent values live
across completion. Its larger GPR high-water mark verifies that allocation
accounts for those live values.

The wrapper decodes six units at emitted instruction boundaries and follows
their allocated input definitions. It rejects reads of either U0 operand
before redefinition. Six misallocated/swapped-tuple mutations must fail. The
executable also checks the disabled gate and unsupported count, workgroup,
payload and capacity cases. `test/mesh-source-fixtures.json` records provenance.
These checks do not emulate hardware record formation, clipping or retirement.

For an external hardware harness, the test executable can export the complete
BASE/SHIFT_X binaries and their compiler metadata without modifying them:

```
mkdir -p <export-dir>
<build>/src/asahi/compiler/apple9-native-mesh --export <export-dir>
```

The exported files require a native launch. They are test artifacts, not an
asset-loader path in the driver. The metadata reports actual GPR/private-frame
usage; it does not insert H1's experimental 64-byte frame override.

## Object collective call

`agx_nir_apple9_taskless_object` generates a T1 API grid launch from runtime
counts. `agx_compile_apple9_taskless_object` uses native object lowering to
emit lane-zero grid stores, mask the API body, then append the semantic NIR
operation `launch_mesh_workgroups_apple9_agx` after a physical workgroup barrier. The operation
names a local root and byte offset for three uint32 grid counts; it has no
physical register arguments. After NIR motion/optimization, it expands through
`agx_nir_apple9_object_grid_export` between `begin_object_call_apple9_agx` and
`end_object_call_apple9_agx`. The validator rejects unclosed/nested regions,
loops, jumps, PHIs and any SSA crossing a region boundary (including caller
if conditions). The leaf obtains inputs through its own semantic loads.
Selection uses the normal system-value, predicate, shared-load and publication
paths. Local invocation index therefore uses the normal `GET_SR_ZEXT16` read,
including its zero-extension companion before 32-bit comparisons. System-value
caches are invalidated at both region boundaries so a closed callee cannot
reuse a caller's or another callee's read in the shared allocation block.
It emits OBJECT_CALL, the selected NIR body, OBJECT_RETURN and
OBJECT_RESUME in logical execution order. Structured callee masks carry
`local_control` and share one conservative allocation block; its entry/exit
NIR identities preserve the caller's incoming and outgoing CFG edges. OBJECT_GRID_EXPORT consumes a normally allocated
publication word. The scheduler treats call/return/resume as full resource
barriers and drains pending work at each boundary. RA sees the caller and
callee together, so all values live across the call interfere with callee
temporaries. No call-saved bank or fixed temporary is reserved.

After allocation, layout moves each callee out of line beyond the main STOP.
The caller keeps CALL followed by RESUME; relative CALL fixups use the final
callee block label. RETURN encodes convention-1 RET and RESUME encodes POP2.
All operations, including the main STOP, were present before RA. Layout adds
no instructions. No byte matching, fixed PC, padding,
opaque token or post-RA register-bearing instruction insertion is involved.
Leaf specialization deliberately avoids a general separate-compilation ABI.

This models the provisional token-free contract in the device runner's `1366b83afc8`:
one converged T16 object group, convention-1 CALL/RET, empty immediate-return
mask and lanes 0–15 restored by POP2. Divergent physical callers, other physical widths, nested
calls and scratch/spills are rejected. Hardware portability, hidden call GPR
clobbers, frame backing/minima and arbitrary grid values remain unqualified.
GPR-preservation promotion gate: exercise live values across CALL/RET throughout
the complete allocator-assigned GPR range. The observed r7 sentinels and r1-to-r3
rename do not establish that general guarantee.
The compiler emits one program but retains `apple9_body_only`, so ordinary
upload still rejects it. The observed object frame fields 16/16 must not be
inferred from its zero spill requirement or reduced to zero by a future
launch builder. No native runtime integration is enabled by this compiler work.

The driver does not call `agx_compile_apple9_taskless_object`; its
enable argument is false outside tests, and the disabled path leaves NIR and
output untouched. The host-only `apple9-native-object` and `Apple9Object*`
tests cover the compiler contract. Host interpretation checks our assumed
call model; it cannot qualify the hardware ABI. Runtime staging, native
binding and the reservation provider are separate (step-3) work.

`apple9-native-workgroup` checks all sizes 1..16 in both stages, with and
without store/atomic instrumentation, plus six 3D shapes per stage. Its NIR
interpreter checks active ballots, every local ID and size value, exact
side-effect counts, untouched inactive lanes, barriers, grid ownership, and
the absence of global record writes by plain mesh completion. Every case also
goes through normal selection and RA with physical T16 metadata and no spills.
Disabled, invalid, variable and wider sizes reject. `--export <directory>`
exports T1/T3/T16 plain and instrumented programs for hardware qualification;
these test files do not add a driver asset-loading path.
