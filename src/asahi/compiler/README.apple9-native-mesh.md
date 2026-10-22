# Apple9 native mesh compiler development path

`agx_compile_apple9_native_mesh(nir, enable_native_mesh, ...)` compiles a complete
mesh program through the ordinary NIR, VIR, allocation, scheduling and packing
pipeline. The gate is disabled unless explicitly enabled. It is a compiler
entry point used by Honeykrisp's opt-in `HK_APPLE9_NATIVE_MESH=1` runtime.
The existing compute-emulated Vulkan mesh path remains the default.

The accepted development class is a fixed API workgroup with 1..128 invocations
(any 3D shape), triangle topology,
maxima V3/P1, runtime `SetMeshOutputs` counts within those maxima, and no task
payload. Honeykrisp selects physical 16x1x1 for up to 16 logical invocations
and 32x1x1 otherwise. User shared storage is supported within the combined
32-KiB local allocation. The native class reserves room for the maximum current
output layout; the hybrid API retains emulation's public limits and falls back
when the combined allocation cannot fit. The opt-in native path admits up to
124 32-bit user scalars in generic varying slots, with smooth, flat and
noperspective interpolation selected by the fragment interface.
`HK_APPLE9_NATIVE_MESH_ATTRIBUTES=0` disables these for explicit comparisons.
The current hybrid candidate and its qualification status are tracked in
[the milestone log](#native-hybrid).

Mesh output writes/reads use component-major local planes in semantic order.
Completion publishes each plane with the existing U channel operation; user
channel numbers follow the ordinary VS smooth/flat/linear grouping. The compiler
exports that same varying layout to the fragment compiler and shared PPP builder.
User shared storage begins after the output planes at a 16-byte boundary.
Shared metadata records the logical size, offset and padded combined size;
output-state rows continue to describe only the output planes. The launch
allocation is the next power of two covering both regions, with a 128-byte
minimum. Shared accesses and atomics use the same native local root, and
barriers retain ordinary NIR semantics. No fixed application registers, new opaque code, or
capture-specific instruction sequences are introduced. P1 primitive attributes
use the qualified flat-varying broadcast lowering; set
`HK_APPLE9_NATIVE_MESH_PRIMITIVE_LOWERING=0` for explicit comparisons. General
primitive linking-plane/device-record publication is still unresolved.
Paired VS/native device arms live in `../vulkan/tests/native-attributes`.

Position retains
all four components, including non-unit W. Hardware clipping is unqualified;
the compiler does not perform a software clip-domain drop. The
hardware validation uses the opt-in paired kernel. Direct multi-workgroup child
grids are admitted for qualification; this is not a native-mesh default change.

Count publication uses the first API invocation's actual values. Uniform
branches may contain separate static `SetMeshOutputs` calls, provided at most
one executes per invocation as Vulkan requires. Both compiler-owned count words
start at zero, so a uniform early return that bypasses the call emits no output.
Returns reconverge before the physical completion barrier, including empty
workgroups. The generic body lowering still rejects constant counts above the
declared maxima. The host tests exercise dynamic counts, empty outputs, an early
return before the call and mutually exclusive calls with poisoned local memory;
hardware qualification of these additional paths remains pending.

## API invocations and physical completion

At physical T16, the entrypoints use the existing
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
API lane set would lose two dimensions. The gated runtime uploads this generated
object program alongside the mesh program. API task shaders select emulation
in the hybrid runtime.

For logical sizes above 32, ordinary `nir_lower_workgroup_size` serializes
logical subgroups on physical T32 between workgroup barriers. Logical IDs,
partial final-subgroup masks and live values across barriers are preserved.
Completion is appended after that transformation. Physical mesh widths other
than T16/T32, variable API sizes, and requested API subgroup sizes other than
32 reject explicitly. The generated object remains physical T16. T32 launch
and shared-memory runtime admission are host-tested development paths; they
still require a paired kernel accepting T32 mesh state and device qualification.

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
| `mesh_submit_scalar_apple9_agx` | actual vertex count and an unqualified second word, each 32 bits | Consumes an adjacent register pair; scalar mesh submission; mesh/memory ordering |

All are side effects and are neither eliminable nor reorderable NIR operations.
VIR models source uses and destructive operands explicitly. A count operand
owns a whole word conservatively, protecting any separately live companion
half by copying before a destructive use. Qualified U0 captures preserve the
selected B halfword, so B is an ordinary read and its later uses keep it. Both
submit forms use the existing tuple and COLLECT machinery. Scalar submit also
reads base+1: T8132 renders with 0 or 3 there and suppresses all geometry with
0x10..0x40, so native completion supplies an explicit zero. The word's meaning
is not established. RA chooses all physical operands. The packer derives
halfword descriptors as `2 * register + half`, and checks pair adjacency.
Completion scheduling drains pending operations before these side effects.

The known class issues control in all physical lanes with A = the published primitive
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
`hk_upload_shader` rejects this flag unless the gated native uploader verifies
the stage's complete compiler metadata. Generic compute or vertex submission
cannot run the program. Native mesh and generated object programs remain
separate owned shader BOs, retained by every referencing command buffer.

The private argument/metadata ABI is version 8:

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

`apple9-native-shared` checks shared allocation, cross-logical-subgroup exchange,
barriers, atomic return values, narrow accesses and guards through the NIR
executor, then audits allocated instruction roots. `hk-native-mesh` validates
combined launch allocations and rejects corrupted allocation/stage/width
metadata without modifying the launch output. `hk-native-mesh-options` exercises
the real driver compiler boundary for T1..T128, XYZ and mixed shapes, including
shared storage and invalid sizes. Scratch and returned-atomic frames still
reject at runtime; the maximum-attribute dynamic shared stress programs spill.

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
The compiler retains `apple9_body_only`; the gated driver validates the complete
native metadata before upload. Object frames are 16/16 and mesh frames 64/64,
independent of zero spill requirements. The runtime rejects scratch and
returned-atomic frames pending separate qualification. The generated object has
no caller values live after its call, so broader CALL/RET GPR preservation
remains a promotion gate for other object programs.

Honeykrisp generates the taskless NIR and calls
`agx_compile_apple9_native_object`. It constructs launches and mesh state from
compiler metadata and ordinary command-pool allocations, and uses the shared
fragment/PPP path. It accepts direct one-group draws. Conditional rendering
rejects under the native gate until zero-grid object completion is qualified.
The paired kernel requires one native draw in its own
stream, starting with native state, one layer and one sample; mixed/multiple
draws in that stream reject during command recording. API task shaders,
indirect/multigroup draws, multiview,
non-fill polygon modes and negative-one-to-one depth reject explicitly. The
last case still needs a native mesh equivalent of the vertex clip-halfz pass;
it is not silently ignored. The opt-in feature limits are below extension
minima and describe a development class, not conformant EXT_mesh_shader support.

The kernel must select its native setup by parsing hardware mesh packets, own
reservation/admission/retirement, and supply the native entry backing without
userspace resource aliases. Existing UAPI fields keep their usual meanings.
There is no native-mesh capability bit, so this opt-in requires an explicitly
paired mesh-enabled kernel. Host interpretation and submission recording do
not qualify GPU clipping, retirement or pixels.

`apple9-native-workgroup` checks all sizes 1..16 in both stages, with and
without store/atomic instrumentation, plus six 3D shapes per stage. Its NIR
interpreter checks active ballots, every local ID and size value, exact
side-effect counts, untouched inactive lanes, barriers, grid ownership, and
the absence of global record writes by plain mesh completion. Every case also
goes through normal selection and RA with physical T16 metadata and no spills.
Disabled, invalid, variable and wider sizes reject. `--export <directory>`
exports T1/T3/T16 plain and instrumented programs for hardware qualification;
these test files do not add a driver asset-loading path.

## native-coverage-plan

**2026-10-22; project: `mesh-shaders-next-big-project`. Proposal only; awaiting
the manager's go.** Start with emulation fallback and consistent API limits.
Generalize the native implementation behind that fallback, one qualified
capability at a time. Neither the current development limits nor the current
native compiler justify default-on EXT mesh support.

This section supersedes historical coverage/admission statements above only
where explicitly stated. Its failure analysis pins installed Mesa
`bcb0837f948a`, device runner report `a0b30cfd5b1`, and CTS
`f6a29701220f34dd1407513bfe80d74ca7b392ce`. Source line numbers below refer to
that Mesa revision, not the moving checkout. The
[full-run report](../../../../../../results/mesh-cts-full-native/RESULTS.md),
[case audit](research/native-coverage-20261022/CASE-AUDIT.tsv), and
[reproducible summary](research/native-coverage-20261022/SUMMARY.json) are the
evidence. The emulated comparison remains pending; no predicted pass gain below
is a measured result.

### Coverage and exact first rejections

The native inventory contains 28,012 cases: **15 pass, 625 fail, 14,626 skip,
2 aborted, 12,744 not run**. Only 15,266 completed. This is a partial full-list
campaign, not a completed full suite. There were no designated expected
failures, and repeat stability was not assessed. All 1,123 NV cases skipped.

The audit joins every failed case by its full name to the original
`CASE-RESULTS.json`. A host ARM/QEMU replay of the 545 feature-not-present
cases uses the **exact unmodified installed ICD**, SHA256
`907ced990a15920b38074b289c6fccafdc1cc2957529753e737564306c1aa5d5`, a no-op DRM
shim, and a proxy that adds a debug messenger. This exposes the compiler's
otherwise hidden rejection reason. It performs no GPU execution. The available
host CTS binary differs from the device binary; all **109 distinct GLSL inputs
and both assembly inputs** match original device shader-cache source keys.
This corroborates the source-based diagnosis; it does not establish identical
generated SPIR-V or new device outcomes. All 545 reproduce their Fail status.
[Audit method and inputs](research/native-coverage-20261022/README.md).

The following buckets are disjoint. Topology takes precedence within the
combined shape gate; an actual count below a declared maximum does not bypass
that gate.

| First rejection | Cases | Exact source condition at the pinned revision |
| --- | ---: | --- |
| Larger triangle output declarations | **293** | `eligible()`, `agx_nir_apple9_mesh_native.c:18–24`: `max_vertices_out != 3 || max_primitives_out != 1`; returns `native mesh development class requires V3 P1 triangles without payload` |
| Points | **29** | Same gate: `primitive_type != MESA_PRIM_TRIANGLES`; includes fifteen V1/P1 cases and fourteen larger declarations |
| Lines | **6** | Same topology gate; two V2/P1 and four larger declarations |
| Mesh payload declaration | **1** | Same gate: `nir->info.task_payload_size`; `misc.payload_not_accessed` is V3/P1 and has no API task stage |
| Per-primitive user attributes | **27** | `varying_layout()`, same file:260–262: `layout->primitive_components || layout->culling`; message `native mesh primitive attributes/culling are not implemented`. These 27 are user-attribute cases, not measured culling failures |
| Unsupported output builtins | **7** | `agx_nir_lower_apple9_mesh.c:229–233`: non-Position builtin below `VAR0`, except per-vertex PrimitiveID; message `Apple9 native output builtin is not implemented`. Layer: 4, authored PrimitiveID: 2, ViewportIndex: 1 |
| Vertex attribute gate | **1** | `varying_layout():263–264`: more than four vertex components with attributes disabled; `misc.multiple_outputs_vertices` |
| Indirect draws without count buffer | **61** | `hk_apple9_draw_mesh()`, `hk_cmd_draw.c:5388–5392`, `!direct`: 60 conditional-rendering cases plus `builtin.draw_index_in_mesh` |
| Indirect draws with count buffer | **60** | Same compound rejection; `!direct` is already true, and `count_addr` also applies. All 60 are conditional-rendering cases |
| Direct conditional draws | **60** | Same compound rejection, `hk_cond_render_applies(cmd)`; valid direct V3/P1 draws |
| **FEATURE_NOT_PRESENT subtotal** | **545** | Compiler reasons are surfaced by `hk_shader.c:5993`; draw rejection by `hk_cmd_draw.c:5392` |
| Output-component property assertion | **80** | `hk_physical_device.c:1877` advertises 4; CTS `vktMeshShaderInOutTestsEXT.cpp:628` fails `maxMeshOutputComponents too low`. No pipeline compiler rejection occurred |
| **All failures** | **625** | Exactly the original inventory's failure count |

The 293 triangle declarations are V96/P32:218, V4/P2:39, V256/P256:11,
V6/P4:9, V192/P64:8, V24/P8:3, V256/P254:3, and V3/P4:2. Thus the large API
draw group is already blocked at shader compilation: its 218 cases include
48 indirect, 160 indirect-count and 10 direct variants. Those 208 indirect
cases are **latent draw restrictions**, not additional first-rejection counts.
Similarly, conditional rendering appears in 180 failures, but 120 first hit
`!direct`. Fixing a first rejection alone does not guarantee a pass.

The original API-error tally reconciles exactly:

| Reported failure location | Compiler rejection | Draw rejection | Total |
| --- | ---: | ---: | ---: |
| `createGraphicsPipelines` | 265 shape + 27 primitive + 3 builtin + 1 attributes | 0 | **296** |
| Pipeline construction (`retcode`) | 19 shape + 4 builtin | 0 | **23** |
| `endCommandBuffer` | **45 shape**, compiled through deferred linking | **181** | **226** |

In particular, all 226 end-command-buffer failures must not be attributed to
draw support. The replay has 590 diagnostic messages because those 45 deferred
compiler errors also produce a command-buffer error; the audit counts cases.

Feature checks hide further coverage. Counts here use the **recorded first
skip reason**, not guessed capability combinations:

| Feature/category | FNP first rejections | Recorded skips | Exact current gate |
| --- | ---: | ---: | --- |
| API task shaders | 0; the separate mesh-payload failure above is 1 | **7,054** | `hk_physical_device.c:1749`, `taskShader = !native`; compiler additionally rejects any task stage at `hk_shader.c:6061–6067` |
| Mesh queries | 0 | **6,168** | `meshShaderQueries = !native`, `hk_physical_device.c:1751`. The one failing `query.no_queries.lines...` case is in the line bucket and has no query |
| Mesh multiview | 0 | **28** | `multiviewMeshShader = !native`, `hk_physical_device.c:1750`; `gfx->render.view_mask` also rejects at draw time |
| Other requirements | — | **1,376** | 1,123 NV; 120 float64; 80 16-bit storage IO; 24 device-address commands; 15 fragment-shading-rate requirements; 14 output-count/workgroup property requirements |
| **All skips** | — | **14,626** | Exact detailed reasons are retained in `SUMMARY.json` |

No rejection category is inferred for the 12,744 unstarted cases. Many have
multiple requirements; enabling task/query/multiview features can reveal new
failures and skips. These skip counts are opportunity sizes, not pass forecasts.

The two aborted property tests are separate from all these totals. The
48-byte output limit and four-component limit underflow CTS's unsigned
96-point output-size calculation: it logs 2,796,200 locations and
4,294,966,272 output bytes, then exhausts host memory. The
[OOM analysis](../../../../../../results/mesh-cts-full-native/OOM-ANALYSIS.json)
pins the arithmetic and source. This is an API-limit defect; neither a native
GPU fault nor an emulated pass was established.

### Evaluate fallback first

**Recommended first implementation: compile an emulated path for every valid
mesh pipeline, optionally add a native path, and choose a compatible path
before recording each draw.** Shader capability decides whether a native
variant can exist; draw state decides whether it can be used. Keeping only a
native variant for an eligible shader leaves the 181 draw failures unresolved.

This reuses working mechanisms already in Honeykrisp. The emulated path in
`hk_apple9_compile_mesh()` produces compute mesh plus rasterization programs;
`hk_apple9_draw_mesh()` handles GPU draw arguments, count buffers, predicates,
task work, per-view replay and query accounting. No new U encoding or native
task/primitive-record contract is needed to retain those capabilities.

Concrete work and constraints:

1. Introduce a typed capability result for native compilation, with a reason
   recorded in debug/performance diagnostics. Test the shader **and linked
   fragment interface**, shared/output allocation, subgroup and scratch needs.
   Use unmodified NIR or clones before path-specific preprocessing: native
   lowering mutates NIR before several rejection sites. Unsupported native
   capability selects emulation; invalid shaders, OOM and internal compiler
   errors retain their real error semantics.
2. Replace the variant-slot alias: `HK_MESH_VARIANT_OBJECT` currently aliases
   `HK_MESH_VARIANT_RAST` (`hk_shader.h:370–381`). A hybrid object needs distinct
   emulated main/raster and native mesh/object ownership, cleanup, serialization
   and BO retention. Compile the emulated path eagerly; compile the optional
   native path eagerly initially so late recording cannot discover a missing
   fallback. Measure compilation/memory cost before adding lazy optimization.
3. Make routing and fragment lowering follow the linked path, not environment
   globals. `hk_lower_per_primitive_input()` and preprocessing at
   `hk_shader.c:3991–3998` currently assume no emulated fallback under the native
   gate. Native flat channels and emulated record loads need separate compatible
   fragment/link variants. Include the path, interface and ABI in pipeline,
   library, shader-object, deferred-link and disk-cache identities. Preserve an
   unlowered source representation for separately linked stages.
4. Select the route after resolving the required shader capabilities but before
   path-specific preparation, heap work or command emission. Emulate API task
   stages, indirect/count draws, applicable predicates, active mesh queries,
   multiview, unsupported output/state, oversized grids and scratch needs.
   Handle secondary-command inheritance and dynamic state at their actual
   resolution point. Reading a GPU predicate or indirect buffer on the CPU is
   not an acceptable substitute.
5. Handle **stream transitions**, including ordinary VS draws and meta work.
   `hk_cmd_apple9.c:717–725` rejects a second draw after a native stream and
   native insertion after existing commands; native additionally requires one
   sample and one layer. Reuse the existing partial-render store/reload and
   `HK_CS_END_RENDER_SPLIT` machinery to isolate native work, with correct
   attachment, synchronization and query semantics. A later emulated draw
   cannot undo a native packet already recorded. Until splitting is qualified,
   any context without a proven safe transition must stay entirely emulated;
   shader-only fallback is not a default-on milestone.
6. Keep the native opt-in and explicitly paired kernel requirement while the
   hybrid route is validated. Kernel parsing/admission/retirement remain kernel
   responsibilities. This proposal adds no kernel UAPI field or userspace
   allocation replay. Emulation-only remains the control path.

Estimated first milestone: **6–10 engineering days for routing, variant/cache
ownership and stream integration, plus 2–4 days of focused device validation**.
That estimate assumes the existing partial-render machinery can preserve all
required transitions; a failing transition is a real dependency, not a reason
to suppress a feature. No new ISA discovery is expected. Stream/query/render
integration needs hardware validation. A compiler-only prototype is about
2–3 days but does not meet the coverage goal and should not be shipped as such.

The routing changes address the admission causes behind all 545 FNP failures;
consistent properties address the 80 limit failures and the OOM trigger.
They also allow task/query/multiview feature exposure to remain aligned with
emulation. **This is a coverage target, not a claim of 625 recovered passes.**
The pending baseline determines which tests already fail in emulation, and
restored properties change the shaders some CTS property tests generate.

### Advertised limits and features

Keep the existing emulated properties as the public contract of the hybrid
driver; native limits become internal routing predicates. Do not merely change
48 to 32,768 while continuing to reject valid larger shaders. The key proposed
values and required minima are:

| Property suffix (`maxMesh...` unless noted) | Current native gate | Vulkan minimum | Proposed hybrid |
| --- | ---: | ---: | ---: |
| `WorkGroupTotalCount` | 1,048,575 | 4,194,304 | 4,194,304 |
| `WorkGroupCount`, each dimension | 65,535 | 65,535 | 65,535 |
| `WorkGroupInvocations` / `WorkGroupSize`, each dimension | 128 / 128 | 128 / 128 | 128 / 128 |
| `SharedMemorySize` / `PayloadAndSharedMemorySize` | 28,672 / 28,672 | 28,672 / 28,672 | 32,768 / 32,768 |
| `OutputMemorySize` | 48 | 32,768 | 32,768 |
| `PayloadAndOutputMemorySize` | 48,128 | 48,128 | 48,128 |
| `OutputComponents` | 4 | 128 | 128 |
| `OutputVertices` / `OutputPrimitives` | 3 / 1 | 256 / 256 | 256 / 256 |
| `OutputLayers` | 1 | 8 | 8 |
| `MultiviewViewCount` | 1 | 1 | `min(core maxMultiviewViewCount, 16)` |
| `maxTaskPayloadSize` (task enabled) | 0 | 16,384 | 16,384 |
| Task shared / payload+shared (task enabled) | 32,768 / 32,768 | 32,768 / 32,768 | 32,768 / 32,768 |

Minimum column: [Khronos Vulkan limits table](https://docs.vulkan.org/spec/latest/chapters/limits.html#limits-minmax).
Current/proposed columns: `hk_physical_device.c:1816–1882` and
`HK_APPLE9_NATIVE_MAX_GROUP_TOTAL`. Task grid/workgroup bounds retain the
existing emulated values, equal to the mesh grid/workgroup minima above.
Keep existing output granularities at one and recheck all related properties
together. Independent maxima do not promise their unrestricted cross-product.

Retain `taskShader`, `meshShaderQueries` and `multiviewMeshShader` at the
emulation-supported values once every corresponding draw reliably routes to
emulation. Do not enable unrelated float64, 16-bit IO, NV mesh or shading-rate
features to reduce skip counts. Native output planes plus header/alignment and
user shared memory must fit the real native allocation; spills/unsupported
frames select emulation. A legal API output-memory size is not the same as
native local-allocation bytes. Public limits must be backed by an executable
path for every legal combination under those limits.

Until fallback is implemented and validated, keep native as explicit research
opt-in. It cannot satisfy the requested minima alone. There is no honest
property-only fix that makes the current native class conformant.

### Native generalization: hardware needs, research and effort

Estimates below are **one experienced engineer's implementation days after the
necessary hardware facts are known**, with a separate initial research budget.
Research budgets are bounded first investigations, not promises that unknown
contracts will be solved. Rows share work and should not be summed as a schedule.
Every expansion needs focused paired device tests after authorization.

The common invariant is the authored U model: U0 accepts count/control inputs;
U1–U4 publish Position channels for the small class; U5 publishes the scalar
vertex count with an explicit second word of zero. Larger interfaces require
more channel submissions, so “U0–U5” is a reference class, not a six-instruction
template. The [fixed producer model](README.apple9-u0-producer.md) keeps one U0
outside loops, canonical completion selector 1 with alternative selector 2,
and correct waits before consumers. Do not implement larger outputs by
repeating U0; even fully waited repetition timed out.

The completed [r8 report](../../../../../../results/mesh-u0-r8-v2/RESULTS.md)
finds bit44/bit42 coupling: no bit44-off form qualifies both N1 and N32, and
the canonical on form retains M00 failure and M01/M02 rescue. Keep the current
on-form producer/wait contract. Bit44 is not proven to be a universal output
enable, and token absence/multiplicity remains unresolved. Neither r8 nor the
existing V3/P1 tests establish arbitrary output or task contracts.

| Category / measured coverage | Required native work from existing evidence | More device research? | Implementation / initial research |
| --- | --- | --- | --- |
| **Triangle shapes: 293** | Generalize declaration eligibility, index/count layout, state and output allocation. Publish actual runtime counts; distribute channel work over all emitted vertices, including counts above T16/T32, without repeating U0. Verify primitive-index range, masks, empty counts and U5's second word at larger counts. | **Yes.** H1 capacity arms qualify V64/V128 and P32/P64 declarations but emit only V3/P1 through a research adapter. They do not qualify full-capacity native output. Start authored V4/P2, then shared vertices and boundary V/P counts, before V256/P256. | **5–8 d / 3–8 d** |
| **Vertex attributes: 1 direct gate failure; prerequisite for many shape cases** | Reuse existing component-plane layout, ordinary interpolation grouping, U channels, fragment linker and PPP; make capability automatic rather than a separate global attribute gate. Check capacity, sparse components, interpolation, provoking vertex and register pressure. | **Qualification, mainly.** Existing attribute/per-primitive paired evidence covers a small class, not all 128-component/large-output combinations. Combined output/shared limits and spills still need coverage. | **2–4 d / 1–3 d** |
| **Per-primitive data: 27; PrimitiveID: 2** | The qualified P1 lowering broadcasts primitive values through flat vertex channels. For arbitrary P, a shared vertex can belong to primitives with different values, so P1 broadcast cannot generalize unchanged. Establish primitive record stride/address publication and fragment coefficient link, or evaluate a semantic vertex-duplication lowering with explicit capacity fallback. | **Yes for general P.** DATA storage/stride and individual channel encodings are known, but the live local-linking-plane → coefficient → owned-record-address relation is not. Capture that relation for two strides and a nonzero primitive index. | **4–7 d / 4–10 d**; bounded P1 path audit **1–2 d** |
| **Layer: 4; ViewportIndex: 1; other output builtins latent** | Model per-primitive control fields in the local/header layout and publication; connect layer/viewport selection and multi-layer state. Add proper clip/cull-distance transport where required. | **Yes.** Own-MSL body stores establish packed control inputs, not completion consumption. Need distinguishing layer/viewport captures and exact pixels. | **3–5 d / 2–5 d** |
| **Points: 29** | Add point topology to local index/control layout and native state; qualify PointSize, clipping and one-index primitive publication. Current triangle lowering deliberately drops PointSize. | **Yes.** Triangle U0/U5 evidence and a generic body layout are insufficient to establish point submission. | **2–4 d / 2–4 d** |
| **Lines: 6** | Add two-index primitives, line state, widths/raster modes, clipping and primitive attributes; qualify output-count/control interpretation. | **Yes.** Start isolated one-line and two-line authored outputs with CPU references. | **2–4 d / 2–4 d** |
| **CullPrimitive: 0 first failures, latent under shape/feature gates** | Retain qualified P1 degenerate-index lowering only for filled, nonconservative triangles. General culling needs correct primitive compaction/index/count semantics, or established native mode-1 operand semantics; preserve user primitive data and query counts. | **Yes for general native cull.** Mode-1 encoding alone is not a semantic contract. The nine paired P1 arms, including untaken cull, passed; they do not qualify other topologies or general P. | **3–5 d / 3–6 d** |
| **Indirect/count: 121 first failures, 208 latent API cases** | Let authored object code read GPU grid/count arguments, preserve draw ID, count clamping/stride and memory dependencies. A zero child grid needs a qualified object completion/retirement path; repeated indirect draws also need stream isolation. U completion per child remains unchanged. | **Yes.** Nonzero direct child grids are qualified; zero-grid object launch, false count and multi-draw behavior are not. No CPU readback fallback. | **3–5 d / 2–4 d** |
| **Conditional: 60 first failures, 180 total affected cases** | Predicate GPU-side grid publication, including inverted/offset predicates and secondary inheritance. False predicate must skip task/mesh side effects and retire the object correctly. Share the zero-grid work above. | **Yes**, primarily the same zero-grid/retirement discriminator. | **1–2 d / 1–2 d**, incremental after indirect work |
| **Multiview: 28 skips** | Begin with existing per-view replay: correct ViewIndex, masks/layers, view-dependent mesh/task execution and fragment linkage, one qualified native stream at a time. U output publication is still per generated view. | **Yes**, especially layered state, attachment preservation and query semantics. Single-view child grids do not cover it. | **2–4 d / 1–3 d**, after layer/stream work |
| **Task/payload: 7,054 skips plus 1 payload rejection** | Replace generated taskless object with generalized task NIR; preserve API workgroup/shared/subgroup semantics, payload ownership and child visibility, launch counts/IDs and zero-child completion. Extend the collective-call ABI only with qualified liveness/clobber/frame behavior. | **Substantial.** Present object evidence is one T16 parent, limited callee and no API payload. Multiple parents, payload lifetimes, returned-atomic/scratch frames and calls with live values are unqualified. Use emulation until these are established. | **7–12 d / 5–15 d**, open-ended beyond first tranche |
| **Queries: 6,168 skips** | Initially route query-active draws to emulation. For native, add semantic task/mesh invocation and generated-primitive accounting with correct reset/availability, conditional/indirect/multiview behavior and ordering. Shader counters may reuse normal atomics after verifying exact API semantics; opaque hardware counters are unnecessary if software accounting is exact. | **Yes**, for counts/visibility and integration, usually not new U encoding. Do not infer generated-primitive count from visible pixels or local snapshot counts. | **2–4 d / 1–3 d** |
| **Limits: 80 failures, 2 aborted; grid/shared/state restrictions latent** | Hybrid routing supplies the public limits first. Native growth must separately validate allocation/header overhead, larger actual outputs, maximum legal grids, shared combinations, frame/scratch and attachment classes. The current native total-grid cap is also below minimum. | **Yes** for each expanded native resource class. Passing a declared-capacity arm is not a maximum-resource qualification. | **1–2 d property/routing audit** included in fallback; broader work in the rows above |

Evidence boundaries: the
[capacity result](../../../../H1-CAPACITY-HARDWARE-STATUS-20261019.md)
is explicitly declared-capacity-only; the
[per-primitive result](../../../../../../results/mesh-cts-second/RESULTS.md#per-primitive)
has nine passing paired arms and a closing control; the
[NumWorkGroups result](../../../../gate-lift-stack-20261022/device-results.md)
has 32 focused CTS passes plus authored N32/XYZ checks with one object parent.
These are different pinned payloads/campaigns. Their passes are not additions
to the 15 passes of the full inventory.

Additional native rejection sites have **zero observed first failures in this
campaign**, not proof of support: fixed subgroup/workgroup constraints;
nonconstant IO slots, non-32-bit outputs and unsupported output loads;
native combined-memory/frame admission; builtin clipping state;
non-fill/conservative cull state; multisampling/layer count; and mixed/multiple
draw streams. `hk_apple9_prepare_native_mesh()`, the native metadata validators,
`hk_cmd_apple9.c:567–575,717–725,1081`, and the lowerer's reason strings must all
feed the routing predicate. Capacity/unsupported-state fallback must happen
before native packets are emitted, rather than catching a poisoned command
buffer afterward.

### Recommended order and acceptance gates

1. **Restore API coverage with hybrid fallback and consistent properties.**
   Incorporate the pending emulated baseline; build the variant/link/cache and
   draw/stream routing above. This attacks all observed admission failures and
   the three large feature-skip groups without waiting on new native hardware
   contracts. Validate property arithmetic on the host before rerunning the
   formerly OOMing cases. Do not run another native full sweep with the current
   48-byte/four-component properties.
2. **Promote the already understood small native class.** Requalify it with
   the accepted producer model, attributes and bounded P1 primitive lowering,
   including native → emulated → native and ordinary graphics transitions.
   Leave unsupported combinations on emulation. This is a small native
   coverage gain but tests the hybrid architecture and reuses existing evidence.
3. **General triangle outputs plus vertex attributes, then primitive data.**
   V4/P2 reaches 39 currently blocked declarations; broader shapes reach the
   218 V96/P32 API shaders and other meshlet-sized interfaces. Attributes and
   per-primitive values are prerequisites, so a shape-only patch is not a
   promised gain of 293 passes. This is the strongest likely game-relevant
   native expansion; actual game benefit needs application measurements.
4. **Indirect/count plus conditional execution and query accounting.**
   Share zero-child and stream work. Emulation already supplies coverage after
   step 1; native promotion reduces fallback for GPU-driven dispatch. Query
   support is comparatively inexpensive if semantic counters suffice.
5. **Layer/viewport and per-view replay, points and lines.** Promote by measured
   workload demand. Points/lines cover 35 first failures and unblock property
   stress; they matter for correctness but are a smaller observed native
   coverage opportunity than larger triangle interfaces.
6. **General API task shaders and payloads.** Their 7,054 skips make fallback
   urgent, but not native task implementation the cheapest first project.
   Start native task research after child output/retirement and stream rules
   are stable. Prioritize it earlier only if application profiles show task
   emulation dominating runtime.

After go, use focused tests for each change: host property checks, capability
classification and NIR/cache round trips; then paired device tests for each
first-rejection bucket, direct/indirect/count/conditional routes, task payload,
queries, multiview and stream transitions. Include larger legal declarations
with small and maximum actual outputs, empty outputs, shared-memory pressure,
and attachment preservation. Verify route diagnostics so a passing emulated
draw is never reported as a native pass. Compare candidate and emulation using
identical CTS, kernel, libraries, properties and compression defaults.

The acceptance criterion is **no new failures against the measured emulated
baseline**, stable behavior of the promoted native subset, and no unbacked
advertised limits/features. Keep failures, skips, expected failures, mixed
outcomes and not-run cases separate. A full CTS run waits for an explicitly
authorized/overnight slot after focused coverage passes; game testing uses
native applications or the separately authorized native FEX setup, not muvm.
Default-on additionally requires ordinary production-kernel integration and
qualification; a successful hybrid research opt-in is not that decision.

This plan changed documentation and host analysis artifacts only. No fallback,
limit, feature, compiler, kernel or device changes were made for this request.

## native-hybrid

**2026-10-22, manager GO (18:0x).** Implementation branch
`apple9/native-mesh-hybrid` starts at integration-next `bcb0837f948a`; the
approved coverage plan is retained as `4327e79d9a7` (from `104134ce422`).
The goals are no mesh-group CTS regressions against the measured emulated
baseline and native execution whenever the shader and draw contract allow it.
Native remains default-off through the full-group comparison.

Milestones:

1. **Pipeline capability and path ownership.** Classify native incompatibility
   before deferred linking, keep fragment lowering paired with the mesh path,
   and preserve independently owned native and emulated programs. The 45 known
   deferred shape failures must be classified at pipeline creation and routed
   to emulation, rather than become command-buffer errors.
2. **Draw selection, stream transitions and public properties.** Select the
   usable path before emitting work, preserve attachments and ordering across
   native/ordinary streams, and retain emulation's limits and features.
3. **Coverage qualification.** Host route/link/cache checks and the full pinned
   rejection inventory precede device validation against the device runner's fixed
   full-group runner and emulated baseline. The manager schedules device slots;
   they use knob revision 2, selecting mode through debugfs.

Each compiler/driver milestone goes to compiler review. Harness-only
changes do not. Commit locally as work progresses; no installation or push is
part of these milestones. Device runs are requested only with a concrete,
reviewed candidate and host evidence.

Status: the compiler review approved M1 `91c5fddb86a` and M2
`64598c0abb5` for the device comparison, which the device runner is running. Native
remains default-off. The shape rejections select a valid emulated pipeline;
they are not converted into new pipeline-creation errors for valid shaders.

### Milestone 1 — candidate ready for compiler review

The base API shader now always owns the emulated realization. An optional
native leaf owns a separate mesh/object pair or its matching fragment program;
the raster/object variant alias never crosses these ownership boundaries.
Creation-time native capability checking runs the actual mesh lowering on a
clone, including for deferred libraries and shader objects. A negative result
survives serialization and prevents later native compilation. Supported
candidates can narrow further at link time (fragment interface, raster state,
backend/launch restrictions). Only explicit native capability failures discard
the optional realization; upload/allocation errors still propagate.

Per-primitive fragment inputs retain their semantic form until the realization
is selected for compilation. The emulated program reads mesh records, while a
qualified native P1 program uses flat varyings. The cache container versions and
owns both paths, and constant-data initialization checks both layouts before
using the API stage's root pointer. The native gate is still default-off; public
limits and draw restrictions are unchanged until milestone 2.

Host validation (2026-10-22):

- x86-64 and aarch64 Vulkan driver builds pass.
- `hk-native-mesh-options` checks source-NIR immutability for accepted V3/P1 and
  rejected inventory declarations, points/lines, payload, Layer and workgroup
  classes, in addition to its existing interpolation/shared/output-array checks.
- `hk-native-mesh-hybrid` compiles eager and deferred native/emulated pairs with
  per-primitive fragment inputs, round-trips both kinds of cache object, checks
  constant layouts, and injects failure at every BO upload allocation. All
  objects and BO references are released; upload OOM remains a Vulkan error.
- `hk-constant-data` passes. These three focused tests pass together.

The hybrid test replaces BO allocation with host memory, not shader compilation
or cache handling. It does not execute GPU commands. The complete set of 45
historical deferred cases and the full rejection inventory remain qualification
work for milestone 3, after routing and properties are ready. Compiler review
is pending; no device qualification is claimed.

### Milestone 2 — candidate ready for compiler review

Draw routing now resolves the pipeline before selecting a path. Native requires
paired programs, a supported nonempty direct grid, one sample/layer, fill mode,
no API task/geometry/tessellation stage, no multiview or predicate, and no active
mesh/statistics/occlusion query. P1 CullPrimitive additionally excludes
conservative rasterization. Other valid draws use the owned emulated programs;
no GPU-resident indirect argument or predicate is read on the CPU.

The common draw preparation ends a preceding native stream before any following
draw, including ordinary vertex and meta draws. Entering native also ends a
nonempty ordinary stream. The existing partial-render background/EOT/ZLS and
render-continuation synchronization preserve attachments and ordering. Each
native stream consequently has exactly one leading native draw; the packet
emitter retains its final contract check. Secondary command buffers use the
common command-queue replay, so they take the same selection/preparation path.

Mesh/task features and limits now remain exactly those backed by emulation.
Qualified attribute and P1 broadcast support are enabled within the explicit
native opt-in; their diagnostic overrides remain available. The global native
gate is still default-off and the matching kernel is still required.

Host validation (2026-10-22): x86-64 and aarch64 ICD builds pass. Eight focused
tests pass together: `hk-native-mesh-routing`, `hk-native-mesh-hybrid`,
`hk-native-mesh-options`, `hk-native-mesh`, `hk-mesh-query`, `hk-render-merge`,
`hk-pregfx-recording`, and `hk-constant-data`. The new routing test exercises the
real eligibility and stream-preparation functions, including fallback states,
empty-stream reuse, native/emulated/ordinary/meta transitions, partial attachment
registers and continuation events. Its representative stream bytes are not a
GPU rendering test; actual Vulkan recording and device parity remain milestone 3.
Compiler review is pending for both candidates.

### Milestone 3 — host qualification complete; review/device gates pending

The frozen aarch64 candidate is driver commit `64598c0abb5`, SHA-256
`331b16359c26a10504a0111f58d83d6099cb13539c3154f02be20e3df9fceee2`.
The harness sources and reproduction instructions are in
`src/asahi/vulkan/tests/native-mesh-hybrid/`. Compact evidence is in
`research/native-hybrid-20261022/`; full local captures are in the sibling
`../native-hybrid-20261022/` directory. No driver changes follow milestone 2.

- The 625-case rejection inventory completes compilation and command recording
  in both modes with **zero Vulkan API/debug errors**, including all **45**
  previously deferred shape failures. Both modes have identical harness
  outcomes: 622 expected missing-GPU image failures and 3 harness passes, with
  626 suppressed queue submissions each. These counts are **not device CTS
  results**. The comparison uses the same hybrid binary with native off/on;
  it is not a comparison to a rebuilt parent or the full mesh_shader group.
- Mesh properties/features are byte-identical across modes (200 bytes). The
  device reports task, mesh, multiview and query support, 256 output vertices
  and primitives, 128 components, 32768-byte output/shared limits, 16384-byte
  task payload, 4194304 total workgroups, 8 layers and 16 multiview views.
- A separate real Vulkan `NENVNMNN` probe records submissions without the
  suppressing proxy: eligible mesh, emulated mesh, eligible mesh, ordinary VS,
  eligible mesh, meta clear, and two eligible mesh draws. Native-on emits five
  separate native renders and three ordinary renders. Each native stream has
  one leading native packet, one direct draw and its terminator. Emulation's
  compute work precedes its raster draw; all commands retain the recorded
  previous-submit barriers. Continuations load the partial attachment program,
  intermediate renders store partial attachments, and the last render uses
  the final-store program. Native-off coalesces the eight draws into one
  ordinary render between the emulation and readback compute commands.
- The probe's independent CPU image oracle is ready for device execution.
  Host pixels remain the untouched sentinel as expected. Depth/stencil pixel
  preservation, query results and GPU synchronization remain device checks;
  the focused unit test covers the CPU ZLS transition state only.

The pinned host CTS binary has SHA-256
`83acc28c211c515668ddaf108cbd5a38fdaaa1c46a3dd141ee1ed0e6683310aa`;
its build differs from the original device inventory run. The earlier source
equivalence audit applies to the inventoried shader cases. It does not replace
the device runner's fixed device runner. All C harnesses build with warnings as errors;
the final binaries are byte-identical to those used for the recorded evidence.

Both compiler milestone reviews are still pending. Their concrete review
requests are `reviews/review-request-native-hybrid-m1-20261022.md` and
`reviews/review-request-native-hybrid-m2-20261022.md` at the workspace root.
After review, the manager must schedule paired focused transition/state checks
and the full mesh_shader comparison using knob revision 2 and debugfs mode.
Compare against the fixed emulated baseline, retaining failures, skips,
expected failures and flakes separately and checking the actual native route.
Zero full-group regressions and device correctness are not yet established.
Native remains default-off; no device was accessed, installed or reset here.

### Milestone 4 — review follow-ups in progress

Manager 19:1x relayed GO verdicts for milestones 1 and 2, recorded in
`reviews/native-hybrid-m1-review-20261022.md` and
`reviews/native-hybrid-m2-review-20261022.md` at the workspace root.
the device runner's full-group run uses the frozen milestone-2 driver; milestone 4 does
not replace that artifact while it is running.

The shared `agx_apple9_native_mesh_shape_eligible()` predicate now controls
both early driver output-array lowering and the native compiler's V3/P1 gate.
Late fragment primitive-input lowering runs only its targeted fix-up, DCE and
IO gathering, preserving the helper-write guards from the original preprocessing.
Native mesh/FS leaves carry a content-derived compile ID through independent
cache round trips. Draw selection requires matching IDs before native packets
can be emitted; a mismatch selects emulation.

Deferred variants now key the requested realization as well as shaders and
graphics state. Native requests compile only native mesh/object/FS programs;
unsupported bodies fall back using untouched source NIR. Ineligible draw state
requests emulation directly. Switching modes reuses or lazily compiles the other
cache entry, rather than always compiling both paths on first native use. Eager
pipelines retain their independently owned paired realizations. Binding a native
leaf uses that leaf's constant-data root. Cache container revision 2 records the
new metadata and rejects older layouts.

Initial host checks: x86-64 ICD build and four focused tests pass. The hybrid
test counts real backend calls: a deferred eligible native resolve performs
one native mesh compile, one FS compile and zero emulated mesh compiles. It also
checks path switching/cache reuse, pairing IDs through serialization, helper
guard depth, explicit native capability fallback and every BO upload failure
on paired/native-only/fallback compiles. The routing test rejects mismatched IDs
for both eager alternatives and standalone deferred leaves. Broader focused
checks, the aarch64 build and the constant-table/timestamp/merged-render probes
are still in progress; M4 compiler/driver review is pending.

**M2 acceptance condition:** if the device comparison finds any
native-fail/emulated-pass case using native attributes or P1 primitive lowering,
disable the corresponding opt-in default again. Neither default is qualified
by host recording alone. Native itself remains default-off regardless.
