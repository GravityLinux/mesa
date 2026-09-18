# Apple9 parameterized launchers

The T8132 driver generates complete VS, FS and CS setup programs from stage
parameters. Root literals, resource loads, allocation, entry
selection, publication/scratch frames, argument transfers and STOP are all
constructed from source. Coverage shaders use the same entry ABI as other
fragment shaders. No executable fragments, carrier files, render seed or
external division table are required at runtime.

`agx_apple9_launch.c` accepts the USC entry offset, resource addresses,
compiler publication requirements, independent frame extents, tile layout and
threadgroup-memory size. `agx_apple9.c` supplies those live parameters. The
fragment importer, library parser, external loader and fragment exporter have
been removed.

## Independent shader entry ABI (T8132)

Normal GLSL compilation produces complete VS, FS and CS bodies through NIR,
Apple9 virtual IR, shared register allocation, scheduling and encoding. Each
body starts at byte zero of its own executable, low-VA, writeback BO, sized
from the actual binary. Bodies are immutable and need no relocation when a
pipeline or submission changes. Internal branches remain relative to their
instruction address. Body size is bounded by allocation and compiler limits,
not a 64/96/128 KiB shared code arena.

The first 8 MiB of the USC heap holds a stable, GPU-read-only entry arena.
Its 0x340-byte helper directory is initialized once. Each shader body BO owns
one 0xc0-byte block, with a 0x40-byte header, a 0x40-byte constant reservation
and an entry at +0x80. Compute uses the generated terminating constant helper.
The arena remains mapped for graphics and compute; no submission replaces
its mapping. Launch and resource records are allocated in the batch USC pool.
Batch limits follow command and resource space, independently of body size.
There is no shader-body packing or code-size rollover path.

The generated target record selects the small entry through its compact
entry field. The entry uses the compiler's `JMP_EXEC_ANY` encoder to transfer active
lanes directly to the body. This ten-byte instruction carries a signed,
instruction-start-relative byte displacement in six bytes. The entry requires
even addresses within the same 4 GiB USC heap and bounds the displacement to
that heap's reach. An immediately following STOP terminates if no lanes are
active. The transfer creates no return address and consumes no GPR, resource
argument, publication or scratch slot. Bodies end through the compiler's
ordinary stage completion and STOP; they do not return to the entry.

The body resource contract is the stage ABI described below. Graphics roots
are texture, sampler and buffer-table arguments. Compute selects compact direct
buffers or six descriptor-table roots, as described below. These are argument-space
values, not fixed body GPR assignments. Ordinary body allocation uses 64
32-bit GPRs. The shared allocator reserves r0:r1 for control/wait and copy
support, extending the reservation through r3 when spill parallel-copy
support is needed. Entry setup's temporary GPR choices impose no additional
body reservation. The compiler reports publication high-water marks and
aligned scratch byte requirements to the launch builder. The current
per-invocation spill limit is 4096 bytes; changing the code allocation does
not change that limit or the independently described atomic-frame contract.

Graphics packages intern byte-identical bodies by stage and size, sharing
immutable BO ownership. The package cache evicts only inactive packages;
each in-flight batch pins its packages and independently references both
stage BOs. Compute shader objects own their body BOs, and each dispatch batch
references the BO before shader deletion can release it. CPU writes are
reported through the normal BO dirty-range interface. An entry is allocated
lazily under the device entry mutex and is immutable while its body BO has
references. The final body reference releases the entry before BO cache reuse.
Pending batches retain both the body BO and the arena, so shader deletion and
cache eviction cannot recycle an entry while the GPU is using it.

Submission follows the ordinary Gallium dependency and retirement model.
It does not wait on the CPU to install entries or switch graphics/compute
state. Per-draw launchers, resources, PPP records and varying state live in
batch pools; only the previous canonical draw record is retained on the CPU
for adjacent-state reuse. Obsolete global framebuffer/context publication,
the separate USC header alias and fixed render-context mapping are removed.
Color load/store helpers keep their own batch launch handles.

VDM commands are encoded directly into their final GPU allocations. The initial
encoder BO comes from the render-context heap; rollover allocations come from
the batch's context pool. Submission uses the initial BO's address without
relocating or copying command bytes. Each allocation retains the ordinary
0x800-byte mapped command-fetch tail, including after links and terminators.
The CPU cursor points at the native terminator, so appending a draw or a link
replaces it without stepping before a newly allocated continuation buffer.

Each render batch starts with a native VDM barrier, encoded as `0x20000013`
on T8132. Asynchronous depth/stencil blits and texture/sampler reuse need bits
0, 1 and 4 together. Clearing any of these bits from the tested `0xff` mask
reproduced failures; the reduced `0x13` mask passed. The older USC invalidate
bit 3 alone did not suffice. The individual cache domains are not yet known.
The packet belongs at batch initialization, before the first vertex envelope;
it does not add a CPU wait or a submission fence.

On T8132, a VDM stream link encodes a 32-bit offset from the render-context
base, with the old high-address field zero. An otherwise intact 10,002-draw
stream with 11 absolute-address links failed the tiling event; changing only
those links to context-relative addresses completed three replays. The normal
GLES path then passed rollover-boundary sweeps and 100,000-draw workloads for
both indexed and non-indexed draws.

The mapped tail is necessary because command fetch can overread. A T8132
Minecraft capture failed as its relocated stream reached within 0x400 bytes
of an unmapped page; mapping that following page made the unchanged frame
complete 100 times. The previous copy path lost the padding when allocating
only the command bytes. Direct allocation preserves the encoder's existing
guarantee by construction. These observations establish the overread hazard;
they do not establish the exact hardware fetch width.

T8132 placement experiments executed graphics and compute entries beyond the
former 18-bit target limit, including +0x7f8000 and +0x7f0000 respectively.
The target occupies all 24 bits of `2 * entry_offset + 0x2a`; the builder
rejects odd offsets and overflow. Removing the entire old 4 MiB header alias
and fixed render-context mapping still passed mixed compute/render readback,
multiple contexts and 4x MSAA with the generated tile pipeline.

Asynchronous validation held the first job behind a software fence and
verified its completion remained unsignaled: a 32-pair indirect workload
queued 96 jobs (64 compute and 32 render), then passed 8,396,800 readback
checks. A 64-pair direct workload reached 127 pending submissions before
normal batch pressure, and passed 16,793,600 checks. A three-context test
queued 240 render jobs while deleting the final round's programs and creating
96 replacements, with exact MRT readback across the original 864 draws and
192 pixels from the replacement programs. Diagnostic sources and logs are in the
workspace's `tmp/apple9-async-state-20260914/`; the gate is not driver code.

Compute emits the full `0x600fffff` post-dispatch CDM barrier, matching the
Apple8 path. GPU cache ordering remains separate from CPU submission lifetime.
Queued 64-pair direct and indirect tests also pass with the original
`0x60000160` tail; they do not independently reproduce the reported stale-SSBO
failure or prove that the narrower barrier is sufficient for other workloads.


Native M4 validation includes ordinary compiled bodies of 230942 bytes (VS),
246100 bytes (FS) and 203284 bytes (CS), with input-dependent control flow and
spills. Isolated placement controls separately executed compiler bodies at
2.25 and 3.75 GiB USC offsets and a backwards transfer. Those controls verify
branch reach; they are absent from the production driver. The encoding's
full signed 48-bit range is unit tested but is not a claim of hardware
execution outside the 4 GiB heap. Evidence and negative destination-STOP
controls are in the workspace's `tmp/entry191/` and `tmp/independent192/`.
The launch machinery is now source-generated as described below. This does
not imply that every control bit has a complete architectural interpretation.

## Resource ABI

Vertex and fragment stages always reserve texture, sampler, and buffer roots
in that order. A shader without textures uses the same layout, with unused
texture/sampler roots zero. The compiler's graphics buffer argument base is 2.
There is no separate buffer-only vertex or fragment launcher selection.
The six-slot entry table reserves three further pointer slots, currently zero
in normal draws. Each pointer is loaded from a GPR-addressed table and
published as two argument words. The first transfer waits for slot 2.
Independent authored VS, FS and coverage-FS readouts matched both halves of
all six pointers, including distinct values in the reserved slots, across
three draws with different table addresses.

Graphics chooses root pair r2:r3 and pending words 18..29. Tests also passed
with root pairs r4:r5 and r60:r61 and pending bases 8 and 24. These are entry
allocation choices; ordinary shader allocation remains independent. The
normal resource loads occupy 42 generated bytes after the 16-byte root setup;
48 generated bytes transfer the twelve words. Coverage uses the same loads,
transfers and control records; its old auxiliary spans are omitted. The entry
field is at byte 68 for all graphics. The original independent resource
readouts are recorded in `tmp/graphics-resources170/RESULTS.md`.

Buffer-only compute with up to 18 combined SSBO/UBO resources uses direct
arguments: one hidden root points to three group counts and visible buffer
arguments start at 1. Textures or larger binding sets select six roots:
texture descriptors, sampler descriptors, buffer addresses, group counts,
shared memory, and a reserved zero pointer. The indirect buffer table supports
32 combined resources. Both ABIs compact active API bindings, with writable
buffers last, and retain 32-bit ownership masks. The graphics sysval UBO and
sampler mapping follow API vertex/geometry shaders lowered to compute.
Texture publications contribute to the normal compiler-sized launch frame.

Each dispatch has a 0x100-byte resource record. Direct pointers occupy qwords
0..18; the table ABI publishes six pointers and uploads buffer addresses
separately. Inline group counts live at +0xc0. Keeping
geometry beyond the pointer window prevents larger binding sets from
corrupting it. Direct and indirect dispatch use the same resource layout and
CDM modes. The compute constant helper is generated from source using the
existing terminating program and padding form.

Returned 32-bit atomics require four bytes in frame extent A in the tested
path. Discard-result atomics do not request this storage. The compiler derives
that requirement from the final atomic instructions, after dead results have
been removed. The driver uses `max(spill_bytes, atomic_frame_bytes)` for A and
the existing spill requirement for B. These are independent fields; this does
not establish the role of B for every possible operation or wider atomics.

Follow-up T8132 compute tests isolate A as a byte capacity for the main's
word-indexed spill accesses: if the highest accessed slot is S, A = 4*(S+1)
passes, while one byte less fails. This boundary holds at four pressure levels,
also with returned atomics live among the spills. B = 0 passes those tests;
increasing B does not rescue undersized A. Returned 32-bit atomics without
spills fail at A = 0..3 and pass at A = 4. These are independent requirements
covered by the maximum, not evidence that atomic storage must be appended to
the spill range. The mixed shader can itself cause a different spill layout.
The driver retains conservative aligned scratch sizes and the existing B
setup until graphics stages and other call patterns are independently tested.
Detailed controls and generated-code slot traces are in
`tmp/launcher-opaque167/RESULTS.md` in the parent workspace.

## Stage parameters

Let C denote the encoded entry field. The builder derives C from the stage
and resource count; callers supply an entry offset and live stage parameters.

| Input | Established operation |
| --- | --- |
| Stage entry | USC byte offset, encoded in the tested 24-bit target field |
| Graphics resource root | Full GPU pointer in entry pair r2:r3 |
| Compute resource root | Full GPU pointer in the entry ABI pair r2:r3 |
| Publication count P | `ceil(P / 2) << 7` in the 16-bit word at C+11 |
| Frame extents A/B | Independent 16-bit words at C+13 and C+15 |
| Fragment tile bytes and samples | Generated 1×/2×/4× allocation before C |
| Compute threadgroup memory | Allocation word at C-6, `(bytes << 2) | 0x80` |

P is the compiler's actual 32-bit publication high-water mark, including
holes and temporary texture operands. It is not the number of user varyings.
Unused low bits of the publication word are zero. Graphics requires compiler
publication metadata; there is no external prolog or library-default fallback.

The allocation interface currently accepts zero or the four threadgroup-memory
sizes with execution evidence: 128, 256, 512, and 1024 bytes. Zero emits the
source-generated zero-size allocation form. The compiler lowers aligned 32-bit
shared accesses and workgroup barriers, rounding the shader's allocation to
one of those sizes. A shared resource occupies its own argument root with value
`0x80000000`; it has no external-buffer read/write hazard. Native probes cover
multiple arrays, one through three buffer bindings, and partial workgroups.

Internal libagx helpers also retain portable NIR before Apple8 preprocessing.
Apple9 compiles this NIR through its ordinary compute backend and binds the
argument block as UBO 0. The geometry prefix helper uses native integer scans,
uniform broadcasts, shared storage, and barriers through this path. Its body
is generated from the helper source; it is not a fixed binary sequence.

The compute builder chooses root pair r2:r3 and a pending-result
base of 18. Independent hardware tests moved the resource root to r4:r5 and
r60:r61, and pending results to bases 8 and 24. These are setup allocation
choices, not fixed register assignments in ordinary compiled shaders.

For N visible resources, R=N+1 pointers (the group-count pointer followed by
the resources) populate 2*R pending words using one- or two-pointer loads at
eight-byte offsets from the resource root. Generated transfers publish these words to the main's
argument window, with the first transfer waiting on dependency slot 2.
Authored main-entry readouts verify both halves of every pointer against the
CPU record, including independently allocated unused entries. Main code reads
qword i with uniform selectors 4*i and 4*i+2. The builder computes the main-call
position as C=26+14*ceil(R/2), with a generated frame and stage allocation record.

The four formerly published compute state words had no consumers. Their state
record, pointer literal, load and transfers have been removed, saving 46 bytes
per generated compute setup. Preamble storage starts immediately after the
actual 2*R root words rather than reserving the maximum resource footprint.

## Generated launch control

The stage program consists of the following source-generated operations:

| Operation | Bytes | Parameters |
| --- | ---: | --- |
| Root literals | 16 | Full resource pointer |
| Pending pointer loads | 14 each | Root pair, result base, offset, one/two pointers |
| Stage allocation | 8 | Threadgroup memory or tile/sample layout |
| Stage target | 10 | Entry offset in the USC heap |
| Publication/scratch frame | 10 | Publication pairs and independent A/B extents |
| Argument transfers | 4 per word | Destination argument, pending source, first wait |
| STOP and zero padding | 4 | Ends the setup invocation |

The target field at byte 2 is `2 * entry_offset + 0x2a`. The tested field
is 24 bits wide; the builder rejects overflowing and odd entry offsets.
Header `0x0177` and control bits at byte 7 bit 1 and byte 8 bit 7 were
constructed through live bit-forcing and cumulative clearing experiments
around authored, normally compiled shaders. The full generated record passed
three changing frames of the large spilling compute shader, including returned
atomic ticket permutation, counter and guard checks. Graphics uses the same
record and resource ABI. No proprietary executable was disassembled or
decompiled to derive this construction.

The frame encodes `ceil(publication_count / 2)` at bit 7 of its publication
word, plus separate 16-bit A/B extents. Unused low bits and the final byte are
zero. The old 48 bytes after that record are unnecessary. The first argument
transfer waits for the pending table loads. Setup then ends with ordinary
STOP. Appending a direct branch to the body can duplicate execution; simple
idempotent stores missed this error, while returned-atomic checks caught it.

The allocation record uses the supported compute-memory size or the actual
fragment tile/sample layout. Fragment modes are 0x4b for 1x/2x and 0x43 for
4x. Vertex and zero-shared-memory compute setup use the zero-size allocation
form. Coverage-specific auxiliary setup and endings are unnecessary for the
compiler's fragment completion model, including discard with spills.

## Unresolved hardware semantics

- The exact scheduling meaning of the target control bits remains unknown.
  Clearing byte 7 bit 1 times out; clearing byte 8 bit 7 can leave a later
  dispatch unexecuted. Both remain set. Header bit 3 and bits 9..15 passed
  independent zero/one controls; the combined source encoding leaves them zero.
- The names of individual allocation-mode bits are unresolved. The supported
  tile/sample combinations and compute allocation sizes are empirical bounds.
  Larger allocations and shared operations beyond aligned 32-bit loads/stores
  and workgroup barriers remain unsupported.
- Direct entry using roots, loads, frame, transfers and a branch alone timed
  out. The successful setup protocol is consistent with deferred stage
  invocation, but its internal context/scheduling mechanism is not established.
- The generated helper directory retains terminal tags whose complete hardware
  interpretation is still unknown. Their required layout remains modeled in
  the helper constructor; they are not loaded from an external executable.

The experiments and negative controls are retained in the parent workspace's
`tmp/launch195/`, `tmp/launch196/`, `tmp/launch197/`, `tmp/launch198/` and
`tmp/target200/`. EXP199's fragment schema was a host-only intermediate step;
EXP201 removes that last file dependency and the compatibility assembly API.
Older fragment files remain research controls and are not read by the driver.

## Compute group counts

Direct dispatch uploads three CPU-computed group counts at resource-record
+0xc0, rounding up for partial final groups. Indirect dispatch publishes the
original GPU indirect-buffer pointer. Both modes expose that pointer as
argument zero, so `load_num_workgroups` is an ordinary device load with no
shader division, reciprocal, local-size read, or mode selection. The division
table generator, upload and allocation constants have been removed. Ordinary
shader arithmetic lowering is independent of this builtin.

## Validation

The host launcher tests cover parameter bounds, unchanged output on invalid
input, guarded output allocations through the maximum resource count, entry
relocation, publication-pair boundaries, and allocation/frame independence.
They accept no external executable fixtures.
Build and run them with:

```sh
ninja -C ../mesa-m1n1-build src/gallium/drivers/asahi/agx_apple9_launcher_tests
../mesa-m1n1-build/src/gallium/drivers/asahi/agx_apple9_launcher_tests
```

Final source-only validation is recorded in `tmp/source201/` in the parent
workspace. A process-local file-access interposer denies the old development
input directory: the saved fragment-dependent control driver fails as expected,
while the new driver passes without attempting an external read. The native
checks include resource counts through 18, returned/discarded atomics, spills,
large VS/FS/CS bodies, discard at 1x/2x/4x, divergent loops with break/continue,
program deletion and queued lifetime, and partial-render pressure. The normal
35-case graphics suite uses CPU/pixel checks; the two legacy cube diagnostics
compare against the retained control's pixel output. No throughput improvement
is claimed.

The native `src/asahi/drm-shim/t8132_apple9_resource_smoke.c` probe takes total
buffer count, UBO count and a sparse-binding flag. For example, `18 2 1` tests
16 SSBOs plus 2 UBOs with API bindings through 31. It uses three fresh programs
and buffer sets, changing range offsets, a CPU hash reference, and full guard
checks. Build with `cc -O2 .../t8132_apple9_resource_smoke.c -lEGL -lGLESv2` and
run with the development Mesa library selected in `LD_LIBRARY_PATH`.

Historical state-load readout controls, placement variations, integration
checks and normal-driver regressions are recorded in
`tmp/state-load175/RESULTS.md` in the parent workspace. The observer belongs
only to the private research library; normal Mesa no longer publishes those
unused words and uses no readout hooks.

## Native tile export

The direct renderer's EOT helper emits `nir_image_store_block_agx` for each
stored attachment. Apple9 lowers it through a bulk-image-store VIR operation,
an allocated publication tuple, PBE image-table binding, and export completion.
Tile coordinates and per-attachment tile byte offsets are ordinary NIR values.
PBE descriptors encode tiled or linear destination layout, level/layer address,
bounds and format. This replaces the per-pixel loop for supported formats.
RGB565, RGB5_A1 and RGBA4 use half-float tile components; RGB10_A2 integer
uses 16-bit integer tile components. Their PBE descriptors convert to packed
memory formats. Native bulk modes also export all samples of 2x/4x MSAA
and select array layers. Layered mipmapped allocations use a generated GPU
pixel-store loop with explicit mip offsets and layer strides. The renderer
restores temporary image/buffer bindings after constructing either helper.
Both the background reload and EOT export helpers prepare fragment launch
records and retain their resources without emitting a rasterized draw. A
fullscreen reload triangle would mark every tile as non-empty, including tiles
the application never touches. Reloading an attachment also must not set its
clear bit: that would request processing empty tiles even after removing the
triangle. Real color clears use ordinary clear draws; depth/stencil fast clears
retain the empty-tile processing required to initialize the whole attachment.
Original-source probes, validation and
measurements are documented in the parent workspace's
`linux-m4-integration/tools/gpu/asahi/plasma-es3/block-export/README.md`.

## Compiler-generated uniform setup

Apple9 main shaders may now have a compiler-generated UBO/ALU preamble. The
compiler stores its complete body, ending in STOP, after main in the shader BO
and reports its byte offset and size. Both parts keep the original complete
resource map. Setup writes results into the remaining argument words through
255: starting at word 12 for graphics and descriptor-table compute, or
`align(2*(N+1), 4)` for N directly published compute resources.
The ordinary main compiler reads those words
through modeled uniform operands.

After frame and root initialization, a launch record optionally tail-branches
to the persistent setup body. It uses the shared signed relative-branch encoder,
with even addresses and the same validated 4 GiB USC reach as main entries.
Setup STOP completes argument initialization and triggers the already selected
main entry; no return address or additional main invocation is introduced.
The inactive-lane fallthrough is STOP. Launch allocations remain 256 bytes for
graphics and 1024 bytes for compute, independent of preamble size. All code stays
in the shader BO, with its existing compiled-object and batch references.

The launch builder reads parameters only, including launch and preamble GPU
addresses. It reads no caller-owned executable bytes and leaves the output
unchanged when parameter validation fails. The driver validates compiler
preamble bounds before passing its address. Host tests cover branch placement
after the final argument transfer, forward/backward targets, address bounds,
and failure atomicity. Native tests cover changing UBO contents and bindings
between dispatches/draws, exact integer transfer, VS/FS use, and uniform math.

An initial implementation that copied setup into each launch record was
replaced after a controlled workload showed additional launch overhead. The
persistent-body version returned that workload to approximately baseline time.
The numerical checks and counter/throughput observations are recorded in
`linux-m4-integration/tools/gpu/asahi/codegen-quality-fixes/REPORT.md` in the
parent workspace. These measurements do not establish GPU instruction-cycle
latencies or a T8132 occupancy table.


Sample shading is compiled into the monolithic Apple9 fragment variant. The
variant includes multisample rasterization state, independently of the physical
attachment sample count, so disabling multisampling preserves broadcast color
stores and the API's sample-count uniform. Per-sample interpolation, depth,
coverage and blending have native T8132 readback coverage at two and four
samples; single-sample fallback is checked separately.
