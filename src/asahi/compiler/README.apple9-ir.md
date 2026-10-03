# Apple9 backend control flow

The Apple9 backend uses the same architectural separation as the Apple8
backend: stable blocks and instructions, CFG liveness, SSA phi edges through
allocation, and final branch fixups after physical instruction expansion.
It retains Apple9's own instruction, register-class, publication, and scoreboard
contracts. It does not reuse Apple8 hardware encodings.

## Storage and branches

Blocks own linked lists of separately allocated instruction nodes. Their layout
links are independent of their allocation lifetime, so a forward target can be
created before its block is placed. Removing the last instruction leaves an
empty block; it does not invalidate branches to that block.

`program.instructions` is a derived pointer view for analysis. Its indices and
`block.start_index` are temporary positions, never branch identities. Moving an
instruction updates its owning list and regenerates this view. Do not reorder or
compact the view directly. `branch_target` always points to a block in the same
program.

Placement belongs to the transformation: a COLLECT is inserted in its memory
consumer's block; an immediate asynchronous-result handoff is inserted in its
producer's block. This distinction matters at loop headers. Neither insertion
requires editing branch targets.

A general unconditional break publishes a true predicate for the currently
active lanes before mask unwind. The unwind's final byte selects that predicate
bank, independently of the scope count in byte three. The predicate dies at
unwind, so all loop depths reuse bank one; ordinary conditional pushes use bank
zero. Publish after exit-phi source selection so comparisons in those expressions
cannot replace the break predicate. Omitting this write can lose break-edge
values or prevent loop termination. Native nested-loop probes through depth 16
confirm the shared-bank model, including breaks at each enclosing level.

The allocation CFG comes from the original NIR blocks and logical edges, not
from linear execution-mask layout. Both arms stay represented so edge copies
have the correct lane population. Synthetic loop-mask restoration blocks retain
their machine placement without becoming SSA predecessors. Branch byte offsets
are assigned after expansion.

An if/else arm whose estimated cost exceeds 19 (texture operations 50, other
intrinsics 10, ALU 1, the common backend's weights) is preceded by
`JMP_EXEC_NONE` to a synthetic block beginning with the following else or pop.
Without it every invocation issues both arms, such as each fragment shader's
pipeline-statistics block. An arm executed with no active lanes writes no lane,
so skipping it leaves the logical allocation CFG exact; the physical scheduler
follows the jump edge when merging pending results.

After allocation, an else immediately followed by its pop is removed: the pop
restores the mask its push saved, and native one-armed ifs are likewise a push
and a pop (2,000 pops but only 288 elses in 400 public pipelines). Earlier, an
empty arm may still receive allocation edge copies, which need the else mask.
On a 300-case Vulkan selection 57% of elses are removed. The physical program must be reindexed before
these fixups; stale block positions can turn a backedge into a shader-entry jump.

## Shared SSA allocation and spilling

A PHI declares one SSA value in its successor. PHI_SRC records identify incoming
values and their logical predecessor. `agx_apple9_ra.c` translates this dataflow
into the shared Asahi IR, retaining selected Apple9 operations as opaque
instructions with explicit SSA sources and destinations. Vector results use
normal shared COLLECT/SPLIT operations. Publications have their own namespace
and slot lifetime; they do not enter the GPR allocator.

The production compiler calls the existing `agx_spill` and `agx_ra` through this
adapter. It reuses shared CFG liveness, SSA spill insertion and repair, memory
slot allocation, phi handling, and parallel-copy lowering. There is no retry
spiller or per-phi scratch snapshot scheme. The bounded VIR allocator remains
for standalone encoding/IR fixtures; production NIR compilation does not call it.

`agx_ra_target` supplies target limits in 16-bit units without changing Apple7/8
zero-default behavior. Apple9 reserves two full registers (r0-r1), increasing to
five (r0-r4) when spill/parallel-copy scratch temporaries are required. Ordinary
values can occupy the remaining registers through r95. Instruction-local
constraints describe each operand's legal range, alignment, fixed register,
and destructive-input behavior. Register-demand calculation includes the
minimum physical extent of those constraints, so rematerialization cannot
shrink the register file below an operand’s required bank. The shared allocator
places operands in their
legal banks and relocates live occupants with a parallel copy when necessary;
there is no post-allocation bank-repair sequence.

Destructive inputs receive private SSA lifetimes. Move coalescing removes the
copy when the original value dies at that instruction; otherwise the copy
preserves the original. The currently encoded extended shift releases its
source, as do the modeled atomic inputs. Sources are otherwise killed after
their instruction, except that a scalar result may take the registers of a
source dying there (`agx_reg_constraint.reuse_killed`), as in `a = a * 3 + b`.
Synchronous instructions read their sources before publishing the result, and
device, private and shared loads read theirs at issue. The source keeps its
release bit: register-state probes on the same family show that when a
released source is also the destination, the new result wins. Encodings that
write a result in parts from several native instructions (`split_result`: the
paired half conversion and `pack_unorm_4x8`) keep their destination off their
sources. Shared SSA kill information supplies Apple9 source-release bits.

Constrained placement prefers a value's phi-web register, or else its slot in
the collect it feeds, beside an already placed source or in a free range that
holds the complete collect. That is the coalescing `pick_regs` applies to
unconstrained values; without it, every phi and every assembled register pair
needs copies. Pure opaque operations
can be removed when dead, and redundant identity copies are eliminated.

Most arithmetic, conversion, special-function, comparison, device-memory,
interpolation, tile, and export-source forms address r0..r95. Wide unary source
fields cross byte 5 into instruction bit 48; that register bit is independent
of source lifetime. Select destinations use bits 4..7, 22..23, and 60.
System-register reads and their narrow zero-extension companion instead have
six-bit destinations at bits 4..7 and 22..23. Atomic result records add bit 60
to that destination map. The shader allocator applies the result-record
constraint to the returning atomic definition.

Texture result tuples have a five-bit base at instruction bits 3..7 and must
fit through r31. Texture coordinates and depth/export publications use their
own namespaces and retain their separate limits. Perspective-coefficient
multiply still has a six-bit GPR model: its coefficient index occupies the
ordinary compact ALU high-register fields.

After allocation, Apple9 lowers physical MOV/SWAP and memory moves. No
register is reserved by default. Swaps use three XORs. A shader that spills
reserves r0..r4: a SAVE releases its data register, so a live allocated value
is copied to the r1 copy temporary first, large frames use r2 as a zero index,
and memory-to-memory swaps hold their values in r3/r4. The long immediate form
has six destination bits, so a literal rematerialized above r63 goes through
r1; allocation first runs without reservations and is retried with r0..r1
reserved only when such a literal appears. Setting bit 60 as a seventh
destination bit, as one M3 disassembler suggests, hung the GPU, and native
Metal never materializes a long immediate above r63. Unused texture channels
read into an allocated register to release their publication. FILL retains the scratch word for later uses and loop
iterations. The physical completion scheduler tracks pending operations and
places waits at consumers or hazards, folding a wait into a suitable consumer
when its operand lifetime permits. It also accounts for register overwrites,
scratch aliases, device-memory ordering, and borrowed publication inputs.
An instruction that only orders against a producer (it overwrites the result
or a borrowed source, or touches the same memory) waits in its own dependency
field. Pending producers flow forward across execution-mask changes and
forward jumps, as native Metal code keeps them; a block starts from the union
of its predecessors' producers. One wait in a consumer retires a slot whose
producers come from several paths when that fold is valid for each of them. A
wait inside a conditional region does not retire the slot for code after the
merge: native Metal waits again there (all 15 such loads in 4,000 public
pipelines). Loop control, back edges (machine or logical)
and entry to a loop header wait for everything outstanding.

Anything that cannot fold uses the standalone wait: byte 0 is 0x06 and bits
10..15 are a slot mask, bit i naming slot i + 1; the empty mask is a nop. It
reads and writes no register. A probe build replaced every standalone
completion with it and passed a 1,167-case CTS selection, while no wait, the
empty mask, and a mask of every slot but the pending one each failed and hung
the GPU. Native Metal emits it rarely (405 in 9.7 million instructions of a
public compute corpus), before spill drains, a few address overwrites and
execution-mask changes; ordinary instructions there name one slot.

Predicate compares carry a dependency slot index at bits 45..47 (six-byte
form) or 61..63 (ten-byte forms). The 0xc0 byte formerly treated as a constant
tail in native captures is slot 6. Coverage publications and dynamic-mask tile
stores carry the bits 12..17 mask: native captures show their byte 1 both with
and without the slot-1 bit (0x14/0x04 and 0x16/0x06), which waited for the
coverage read supplying their operand. Depth publications always show the
slot-1 bit and keep it.

Memory operations read their address, index and data registers at issue, so
those registers are free once the operation issues. In the public compute
corpus, Metal overwrites the address or index of a pending device load 3,744
times without waiting (for example, incrementing a loop index right after the
load). Probe builds that overwrote device-atomic, private-load and shared-load
inputs immediately after issue passed the CTS atomic, indexing, shared-memory,
scheduler-stress and stratified selections. A device atomic releases its inputs
(EXP-M4-47); that is a register-cache release, not a late write. Texture
parameter publications and scratch SAVE's destructive input remain borrowed
until completion. Atomic publication records stay adjacent to their producer.
Tags are retired before reuse.

A returning device atomic stores its old value in an invocation-frame word
named by its byte 3. The adjacent eight-byte result record is effectively a
scratch fill of that word into the destination register: bytes 1 and 4 carry
the frame word and bits 29..31 the completion slot minus one. Native Metal
captures of our own kernels with up to seven simultaneously pending atomics
put every result on exactly the slot those bits name. The record's bits
45..47/61..63 hold a one-hot tag rather than a slot, and that tag also appears
in the atomic's byte 9 bits 1..3 and in its input dependency. Metal gives each
simultaneously pending atomic its own tag while sharing one frame word. It
places that word past its spills and private arrays, also naming it in the
atomic, and reads word 0 when the frame is otherwise empty.

The earlier model wrote 0xa7 into byte 3 and so published every result to
slot 6, while varying the tag as if it were the slot. Waits on any other slot
then read stale registers. A spilling reproducer hit this when a pending
scratch store pushed the atomic to slot 1, and EXP-M4-49's six-slot test passed
only because its first consumer waited on slot 6. Encoding the scheduled slot
passed with slots 1, 5 and 6, so returning atomics again take the first free
slot. Every atomic uses tag 1. The scheduler already orders device atomics
against each other, so two returning atomics are never pending at once.

The atomic's store also explains the corrupted spill slot 0 in that
reproducer: Mesa names frame word 0. When a shader has a returning device
atomic, spills start 16 bytes into the frame, with private arrays after them,
and a frame holding neither stays empty. Pointing the atomic and its record
at a word past the spills, with frame extent A covering it, also passed. That
layout needs larger words than the atomic's known seven-bit field, so Mesa
keeps word 0. Changing the word in only one of the two lost every result. This
store was the latent problem behind wrong results in heavily spilling
tessellation kernels, which contain four returning atomics. Capping their
allocation at 52 or fewer GPRs failed before; it now passes down to 32.
The register-spill region is rounded to 16 bytes. Indexed private arrays follow
this region in the same frame; their combined allocation is bounded at 32 KiB,
as described below.

Completion applies to the full result tuple, so its first consumer need not
read every component. Compact float and extended logic consumers, including
their scalar export forms, can retain an operand while retiring its completion
slot; later reads use the ordinary GPR path. This holds for every asynchronous
producer except scratch fills: device loads, system values and textures were
validated natively (EXP-M4-34/35/63), and atomic, private-load, shared-load and
interpolation results by CTS probe builds. Float exports expose
a split six-bit input mask independently of their output completion tag. These
paths fold waits into useful instructions instead of forcing identity copies.
EXP-M4-62 records the initial native evidence and shared-RA regression tests.

Device loads also consume pending element indices and pointer pairs through
the mask at bits 12..17. The former raw-system-index flag and first-index-use
flag were individual bits of this mask, not separate address modes. Index and
pointer retention remain independent lifetime controls. Draw-ID and coverage
SR encodings are explicit asynchronous producers: their current four-byte forms
select output slot 1, which the scheduler retires before reuse and folds into
a capable consumer. Synchronous SR forms keep their ordinary behavior.

Compact float consumers can complete texture tuples through any used component,
retaining that component and its siblings for later GPR reads. Integer and
other consumers with a dependency field complete them the same way (CTS texture
and image probes); borrowed texture-parameter inputs remain live until
completion. EXP-M4-63 validates these address, SR and texture paths
with authored Metal kernels, matched producer/consumer retags across all six
load slots, source-retention controls, and native Linux shader regressions.

Independent texture operations use the same six completion tags as memory
traffic. The texture bundle's byte 5 carries matched zero-based tag fields at
bits 2..4 and 5..7. Controlled retags of authored T8132 Metal read and filtered
sample shaders validate tags 1–6, including multiple pending reads. The lower
field alone also selected the expected completion; the upper field's separate
role is not established, so the packer follows the matched native form.

The physical scheduler chooses a free tag and preserves the existing result,
coordinate-publication, register-reuse and control-flow hazards. This permits
independent reads to overlap in the selected stream; it does not hoist reads
across dependencies or change execution masks. Publication storage and physical
register pressure can still require earlier handoffs. EXP-M4-66 records the
native evidence and Linux validation. This is distinct from overlap between
separate copy/compute commands.

Set `AGX_APPLE9_STATS=1` to print packed code bytes (including the trailing stop),
physical instruction/copy/wait counts, SAVE/FILL counts, GPR usage, and scratch
bytes for each compiled stage. These are code-generation metrics, not timings.

The resulting VIR is physical: canonical operands identify GPRs or publication
slots, not SSA values. Do not run SSA use analysis or allocation on this output.

## Memory addresses

Buffer addresses enter selection as general wrapping 32-bit byte offsets.
`nir_lower_mem_access_bit_sizes` selects supported memory formats using access
size and alignment, splitting partial stores and reconstructing unaligned
loads. Load/store VIR represents a 64-bit base, a zero-extended 32-bit index,
an independent index shift, and a signed 16-bit byte displacement. Selection
folds `imul`, GLSL's `amul`, shifts and constant additions when bounds prove
that doing so preserves the original 32-bit arithmetic. A potentially wrapping
shift masks its source before widening; an unknown sum remains in the index.
Atomic offsets retain their element-index conversion.

Narrow integer SSA values occupy word registers and may retain unrelated upper
bits after truncation. Integer-to-float selection therefore zero-extends or
sign-extends the source to 32 bits before the native conversion, for both FP16
and FP32 results. This cannot rely on an earlier NIR widening operation:
algebraic optimization may fold that operation into the conversion, including
during legalization before preamble extraction. The narrow vertex conversion
graphics regression covers packed signed and unsigned byte/short attributes
with a uniform transform.

Right-shift selection likewise extends 8- and 16-bit operands before shifting,
and all shifts mask their counts to the NIR operand width. Without that
extension, format unpacking such as `(u16(x) << 6) >> 11` can shift truncated
high bits back into a packed color channel. Native Vulkan packed-image copies
exercise this path; narrowing every producer is unnecessary.

Device loads, stores and atomics name their 64-bit base either as a GPR pair
or as a uniform pair: the seven-bit base field at bit 32 holds the GPR, or
half the first uniform when the file bit (bit 101 for loads, 99 for stores)
is clear. Native T8132 Metal kernels address every buffer argument through
uniform pairs (`device_load ..., u0_u1, r1`, `device_store ..., u4_u5, r1`)
and switch to GPR pairs only for loaded pointers. The non-indirect VIR
memory forms are this uniform-base form: their binding is the pair index. A
global address held by an even-aligned pair of preamble uniforms therefore
selects it directly, without copying the pair into GPRs. Global accesses in
either form may alias each other, so a device store or atomic orders against
any pending global load or atomic; native Metal does not issue device stores
with loads pending (8,461 of 8,464 stores in 2,500 public pipelines).

Vulkan reaches selection with 64-bit global addresses that int64 lowering has
split into a low add and a high add with carry. Selection recognizes
`lo = base_lo + offset`, `hi = base_hi + b2i32(ult(lo, base_lo))`, which is
exactly the pair `(base_lo, base_hi)` plus the zero-extended 32-bit offset,
and passes that offset through the index, shift and displacement folding
above. Successive constant offsets compare each low word with the previous
one, while reassociation folds the constants into one low add from the base:
`lo = base_lo + k2`, `hi = hi1 + b2i32(ult(lo, lo1))` with `(lo1, hi1)` equal to
base + k1. Each step adds a zero-extended value, so this is base + k2 exactly
when k2 >= k1, and it splits the same way. Loads and stores then address a stable base, typically a descriptor's
buffer address, instead of computing and assembling a new pair per access.
Atomics keep the complete address: their element-index conversion is exact
only for an aligned base. A load whose base is a longer tuple's leading
components, such as a loaded descriptor, uses that tuple directly and
retains its registers, since the tuple's split components share them.
Destroyed operands, such as an atomic's base, always receive a private pair.

`load_global_constant` keeps its constant-memory meaning as
`ACCESS_CAN_REORDER` on the pair form, and the preamble admits such loads.
Honeykrisp descriptor reads with uniform addresses therefore move to the
preamble, and buffer addresses reach the main shader in uniforms, as native
Metal arguments do. Multi-word preamble values start at even uniforms so a
stored pointer is a valid uniform base pair. Load/store vectorization covers
global and constant memory while their 64-bit addresses are still whole, and
legalization moves 64-bit data as dword tuples, so a descriptor's pointer
arrives in one load and one register pair instead of two scalar loads. On a
300-case Vulkan selection these steps reduce main-shader instructions by 21%,
device loads by a third and 32-bit adds by more than half.

T8132 hardware validates native index shifts from zero through four. Larger
values in the public M3 descriptor failed the requested-stride probe and are
rejected by the packer; larger source-language strides retain explicit ALU
arithmetic. Vector width does not set the address scale. Adjacent dword
accesses need only four-byte alignment, so normal memory vectorization can
retain vec2/vec3/vec4 vertex fetches with strides such as 12, 20 and 28 bytes.
The vector's complete extent and NIR's alias and bounds constraints still
govern merging.

The current narrow-store encoding requires its data operand in r0. Legalization
inserts a short constrained copy immediately before each such store rather than
fixing the original SSA value's entire lifetime to that register. This occurs
before scoreboard assignment so pending-result handoffs see the final operands.

## Varying publication completion

Scalar float and logic exports produce asynchronous publication results. The
publication index names storage; a separately allocated completion tag names
readiness. Export bits 58..60 encode the completion slot minus one, and varying
stores wait through the mask at bits 12..17. Completion allocation therefore
includes publication producers, their consumers, source-register reuse, and
block boundaries. Retiring a publication waits without treating its storage as
a GPR result. Publication values remain available for subsequent uses.

Varying stores stay in normal instruction order. The former pass that moved all
stores to a block tail hid a missing producer-to-store completion dependency.
EXP-M4-61 reproduced the original Mesa shader through public Metal: no wait
failed 57/5,000 draws; waiting on its publication slot passed 5,000. A wrong-slot
control failed 115/5,000, while retagging both producer and consumer to that slot
passed 5,000. Native source-authored shaders also permit position stores before
later loads. These results establish the dependency without claiming the rest
of the varying-store control fields are fully understood.

## Shared uses and dependency finalization

The synthetic shared zero is initialized before structured control flow.
Caching a value does not make it dominate later uses: a zero first emitted
inside a masked arm is undefined for lanes bypassing that arm. CFG liveness
can preserve its register but cannot repair the missing definition. Keep this
dominance requirement in mind when adding other cross-block selection caches.

`agx_apple9_analyze_uses` builds a shared definition table and ordered operand-use
chains for a stable IR snapshot. Producer, consumer, publication-lifetime, and
device-load index queries use this analysis. Structural mutation APIs invalidate
it automatically; passes that edit operands directly must call
`agx_apple9_invalidate_uses`. Use-chain pointers must not survive a rebuild.

Use order identifies consumers during selection and scoreboard legalization.
The shared allocator determines cross-block liveness. Physical lowering then
selects index lifetime and operand-release fields from that result and inserts
explicit completion handoffs. SFU result hints retain their conservative
selection-time setting; this does not establish additional hardware semantics
for that field. Physical expansion and byte packing follow this stage.

## Regression checks

`agx_tests` covers all 256 four-source assignments, CFG liveness through a loop
with an early exit, phi ownership in a successor, stable targets across COLLECT
and handoff insertion, and empty targets after deletion and storage growth.

Addressing regressions cover unary, quadratic, selected, shifted, and wrapping
offsets, plus partial vector stores and unaligned access legalization. Analysis
regressions cover layout and operand edits, loop-carried index lifetime, and
rejection of stale dependency metadata after allocation.

The shared-allocation regression interprets an 80-value loop through both the
shared physical IR and final Apple9 lowering. Additional regressions cover a
completely occupied operand bank, destructive shifts with live and dying inputs,
and load waits folded into consumers after independent arithmetic. They check
ordered output values,
destructive SAVE semantics, deferred completions, tag reuse, scratch bounds,
and block positions. Hardware validation on T8132 passed CPU-checked compute
loops with 80 and 160 live values and one through three iterations, plus all
568 GLES2 indexing tests (including all 160 temporary-array cases). The focused
13-case pressure list passed 11; the remaining two compile successfully but
exceed the driver's shader-archive capacity.


## Expanded publication namespace (EXP-M4-174)

Export publications are indexed independently of the 96 ordinary GPRs. Float
and logic export destinations and varying-store sources accept 128 publication
indices; their machine operand constraints carry the publication flag. Ordinary
register constraints keep their existing bounds. Both publication allocators and
the shared allocator's canonical physical-value map cover all 128 entries.

The current graphics interface limit is 96 user scalars (24 vectors), with
128-bit interpolation masks. The driver places per-draw coefficient tables in
separate slots and supplies the compiled publication count to the source-generated
launcher. This is validated through normal NIR compilation and T8132 pixel
readback; no captured instruction sequence or fixed publication assignment is
required. Export lifetime shortening and Metal's full 31-vector interface remain
separate work. Compiler tests cover publication exhaustion and all canonical IDs.

## Native cube coordinates and integer texture reads

T8132 authored-Metal tests establish three cube-coordinate operations. `CUBE`
mode 0 returns `max(abs(x), abs(y), abs(z))` in one 32-bit GPR and the face
index in the low 16 bits of the next GPR. The high half of that second GPR is
not defined by the operation. X wins absolute-value ties, then Y. Modes 1 and
2 return the signed U and V numerators multiplied by one half. Normal texture
lowering emits these three VIR operations, a reciprocal, two FMAs with 0.5,
and a mask of the face's low half. The destination pair and all source values
participate in ordinary allocation and liveness. No fixed register tuple or
captured instruction sequence is used.

The 12-byte cube family has destination bits 24..30, source fields at
41..47, 50..56, and 59..65, keep bits 48/57/66 with complementary dead bits
73/74/75, and operation bits 90..91. The model admits the tested source range
r0..r95 and destinations whose complete result fits r0..r95. Input dependencies
are materialized before the instruction; folded dependencies remain unmodeled.
521 independently constructed executions check field movement, lifetimes,
signs, and ties, followed by normal Linux GL cube-sampling conformance tests.

Integer fetches retain integer spatial coordinates through instruction
selection. The read modes for 2D, 2D array, and 3D are respectively `80 24`,
`a0 24`, and `98 01` in bytes 6/7, with byte 10 zero. LOD uses the existing
Q6 field at bits 16..27, so an integer mip level is shifted by 22; array layer
occupies the low 16 bits of that parameter. The private fetch sampler remains,
as in the common AGX path. Texture-size queries, normalization, half-texel
addition, and integer/float round trips are no longer part of fetch lowering.
These contracts were checked with authored Metal texture reads, then GLES3
fetch and fetch-offset cases in both graphics stages and all three dimensions.


### Native multisample block stores

`image_store_block_agx` with a 2D multisample destination selects the native
MSAA block-store mode (`0x228`; the ordinary 2D mode is `0x5a8`). It exports
all samples in the implicit tile, with the 2×/4× sample count supplied by the
PBE descriptor. The publication is `(pixel_x, pixel_y, layer, tile_byte_offset << 16)`,
with layer zero for non-array images. Array stores select `0x528` for one sample
or `0x028` for multiple samples. Ordinary 2D stores use
`(pixel_x, pixel_y, tile_byte_offset << 16)` in the same four-register allocation.

The MSAA publication and store have four-source encoding contracts, participate
in ordinary allocation and liveness, and complete through the existing export
slot handoff. Putting the offset in the third word silently exports attachment
zero for every MRT. Eight-target half, R16F, RG16F and mixed-format tests validate
the fourth-word offset, including reload, blending and partial tiles. No shader
archive, runtime instruction patch, or fixed register assignment is used by Mesa.
The end-of-tile NIR generator supplies 32×32 tiles for 2× MSAA and 32×16 for 4×.

## Software transform feedback

Apple9 uses the Apple8 software-capture model: assemble complete primitives,
execute the API vertex program on the GPU, and store its captured outputs in
ordinary buffers. `apple9_lower_xfb` imports the shared Poly line/triangle
input-assembly helpers into NIR and lowers capture outputs to ordinary SSBO
stores. Vertex pulling, UBOs, samplers, integer data, register allocation and
memory completion use the normal graphics compiler. The capture variant has
no raster varying interface and retains values before clipping/depth conversion.

A compact draw parameter record supplies first vertex, index bias, base instance,
draw ID, chunk origin and instance boundaries. Index loads and primitive
expansion execute on the GPU. Private SSBO bindings 26–31 are possible because
public Apple9 graphics shaders currently expose no SSBOs; this reservation must
be revisited when that API capability is enabled. Addresses remain runtime
resource bindings, with no fixed-register or captured-executable template.

Gallium caches strides and final output ends with the shader. Direct primitive
counts, complete-primitive bounds, append offsets and queries are maintained on
the CPU, as in the existing direct-query path. Indexed restart is scanned and
expanded into a u32 buffer by an ordinary compute shader using Poly topology
helpers. Only its resulting count is read back; indirect argument decoding also
still uses CPU readback. Internal compute shaders normalize function temporaries
and integer division through the same NIR lowering used by graphics. Capture
runs as a discard draw on
a private target and ends its producer batch, allowing the ordinary resource
tracker to order subsequent buffer consumers. Rasterization then uses the
original draw, with Gallium primitive conversion where needed. Geometry and
tessellation stages are not added by this vertex-stage implementation.

Hardware tests and limitations are recorded in EXP-M4-67 and the workspace's
`tools/gpu/asahi/transform-feedback` harnesses.

## Optimization and allocation policy

Apple9 runs constant-division optimization before general division lowering,
then memory vectorization with NIR's alignment and aliasing checks. Late
algebraic optimization, scalar packing legalization, profitable if-conversion,
and cleanup run for every shader. Selection fuses multiply/add and comparisons
feeding selects. The backend's own pure SSA expressions receive wrapping
integer folding, CSE, exact minifloat selection, and floating source-modifier
selection before shared allocation. CSE ends at block/mask boundaries and
uniform-window writes. Buffer contents are never treated as pure expressions;
only immutable graphics binding-table addresses are reused within a block.

The T8132 occupancy table remains unmeasured. Apple9 does not use the older
GPU's occupancy/rematerialization thresholds. Shared allocation considers
operand-bank scarcity and completed physical candidates: baseline placement,
constraint-aware placement, and resource/latency scheduling. Selection compares
loop-weighted spills, instructions, bytes, register extent, then estimated
unhidden dependencies. A larger pressure reduction cannot justify a worse
completed stream under this cost ordering. The latency priorities are relative
heuristics, not measured instruction-cycle counts.

The scheduler tracks execution state, device memory, publications,
interpolation state, and tile state. A write follows every preceding reader of
the same resource, while independent reads can overlap. Unmodeled stateful
operations remain barriers. PHI/PRELOAD instructions retain their block-entry
invariant before target resource dependencies are applied. Physical completion
tracking still runs after allocation and resolves actual register and tag
hazards introduced by copies and spills.

## Native arithmetic forms

A 32-by-32 multiply can produce an aligned two-register 64-bit result. Signed
and unsigned forms share the low word and have distinct high-word semantics.
NIR high-multiply and extended multiply/unpack operations select this tuple;
ordinary unsupported 64-bit arithmetic is not implied.

Arithmetic and logical shifts have native register and immediate forms.
Register amounts use a low-16-bit hardware amount with saturation beyond the
32-bit data width. NIR's modulo-32 semantics therefore require an explicit
low-five-bit mask unless already established by the source expression.
Source-retention fields are part of each shift family's contract.

Binary16 conversion uses native half writes and half reads. Two half writes
form a complete 32-bit packed value; a lone half result has an explicit high
half zero-extension companion whose destination is limited to r63. Half-pair
packing and unpacking otherwise address the full validated GPR file.

Float abs/neg modifiers compose on the source values, before arithmetic. The
absolute-value FADD form is ten bytes and the FMUL form is twelve bytes.
Product negation and addend modifiers are independent in FMA. Inline constants
use only exactly representable minifloats; zero, NaN, infinity, and source
lifetime behavior are not approximated to fit an encoding.

## Uniform preambles

Integer add, subtract and multiply-add accept uniform operands in every source
position. Extended FMA also accepts a uniform first multiplicand. These are
ordinary alternative operand files in the machine model, including uniform
indices through 255; only GPR sources contribute to register liveness. A lone
FMA uniform multiplicand is commuted to the second position when that permits
the compact encoding. Two uniform multiplicands use an extended encoding.

The ordinary NIR path extracts eligible uniform UBO expressions into a linear
setup function. Boolean and half intermediates may move with an expression;
transferred results are 32-bit words. The full resource map is fixed before
extraction and shared by setup and main, including resources used only in the
preamble. Unused vector components are trimmed before assigning uniform words.
Unsupported dependencies, masked setup, scratch, and setup bodies exceeding
8192 bytes retain the original main shader. Only work in the shader's
top-level blocks is eligible: `nir_opt_preamble` keeps the control flow around
what it hoists, so one eligible expression under an `if` used to discard the
whole preamble. Every application fragment shader begins with such an `if`,
the uniform test guarding the pipeline-statistics invocation counter. With
that rule, 247 more programs of a 300-case Vulkan selection get setup code and
main-shader device loads fall by 16%. At 3840x2160, vkmark's effect2d edge and
blur scenes run 17% and 11% faster, and its desktop scene 28% faster.

Preamble results occupy every word after the shader's root publication blocks
in the 256-word argument window. Graphics roots occupy words 0 through 11,
leaving 244 words. Buffer-only compute with up to 18 resources publishes one
group-count pointer and N compacted resource pointers, reserving the final
four-word block in full. Setup starts at `align(2*(N+1), 4)`: three resources
leave 248 words and 18 resources leave 216. Texture-using compute, including
lowered API vertex and geometry shaders, or compute with more resources uses
six roots and begins setup at word 12. Setup and main derive both the ABI and
this boundary from the same full resource map.

T8132 software vertex shaders exposed the block boundary: with four resources,
preamble values at words 10 and 11 were intermittently replaced by zero while
words 12 and 13 remained correct. CPU uniform data and GPU-generated restart
indices were correct. Explicit load waits did not fix the result. Reserving the
whole root block passed 20 consecutive official primitive-restart runs; the
unrounded control failed all 10 runs. This boundary is independent of GPR
allocation and applies to every compute preamble.
`IOR_UNIFORM` explicitly reads one word with a GPR operand, preserving integer
bits, signed zero, subnormals, and NaN payloads when used as a move with zero.
`STORE_UNIFORM` consumes its GPR operand. Its clobber constraint lets shared SSA
allocation preserve a value that remains live; the packer rejects a surviving
unpreserved source. Both forms have machine-table operand constraints.

Explicit floating LOD and bias conversion is expressed in NIR before preamble
extraction: clamp to [-32, 2047/64], multiply by 64, floor, convert to integer,
mask to 12 bits and place the signed Q6 field at bit 16. Integer texel-fetch
levels shift directly to bit 22. A backend texture source identifies the packed
word, so instruction selection cannot apply the conversion twice. Constant
packing folds away, uniform packing can move into setup, and varying packing
stays in main. The clamp order and floor remain explicit without fast-math
reassociation, including for negative fractional inputs.

The compiler appends the setup body, including STOP, after the main body.
Apple9-specific offset/size fields describe it; the older USC preshader fields
are not reused. The launcher initializes roots and stage state, then uses the
ordinary relative-branch encoder to enter the persistent setup body. Setup's
STOP completes the argument context and starts the selected main shader.
Keeping setup in the shader BO avoids recopying code into every launch record.
The BO has the main shader's existing ownership and batch lifetime.

`AGX_APPLE9_STATS` reports setup with `APPLE9_PREAMBLE_CODEGEN` and main with
`APPLE9_CODEGEN`. Preamble work must be accounted for separately rather than
subtracted from total work. Authored atomic-counter probes observed ten setup
executions for both 4,096 and 262,144 main invocations, and one setup execution
for a single workgroup. This establishes amortization for those dispatches;
it is not a universal promise of one execution per API draw.

The complete optimization changes and numerical/code-generation evidence are
recorded in the parent workspace's
`linux/tools/gpu/asahi/codegen-quality-fixes/REPORT.md`.

## Normalized unpacking

UNORM and SNORM 2x16 and 4x8 NIR unpack operations select native normalized
unpack instructions. Each instruction defines two adjacent FP32 SSA results
in an aligned register pair. The 8-bit formats use separate low/high source
halves; source liveness preserves the packed word until both halves have read
it. RGBA8 destination reads for blending use the same general NIR operation.

T8132 validation covers 786,432 output values across the four operations,
including signed minima and both input halves, with bit-exact agreement with
the CPU division reference. Host checks cover tuple allocation, high registers,
input-half selection, source retention and ordinary NIR reachability.

Detailed hardware probes and the Tokyo comparison are recorded in the parent
workspace's `linux/tools/gpu/asahi/m3-compiler-20260914/REPORT.md`.

## Subgroup scans and shared memory

Integer `exclusive_scan` selects the native 32-bit addition scan. Inclusive
scans add the input to that result. `read_invocation` selects a broadcast with
a subgroup-uniform lane index. Its current encoding consumes both the data
and lane-index operands; allocator clobber constraints preserve later SSA
uses of either. This broadcast form does not implement a divergent shuffle.

Aligned 32-bit shared loads and stores scalarize through the usual NIR path.
Their address operand is a word offset in the low half of an allocated GPR;
the instruction selects a separate shared-memory argument root. This initial
model constrains data and address operands to r0..r63. Shared loads use their
own slot-6 completion encoding, with ordinary pending-result materialization
before consumers, conflicting stores, and workgroup barriers. No registers
are reserved for a particular shared array or prefix algorithm.

T8132 validation includes partial and full subgroups, integer wraparound,
1,024-thread workgroup scans and reductions, multiple shared arrays, and
variable geometry transform feedback across a 2,059-input draw. These probes
use development API overrides; they do not advertise general GL 4.3 support.
The geometry lowering also excludes dummy culled primitives from transform
feedback when a source invocation emitted no primitives.


## Per-sample fragment shading

Monolithic Apple9 fragment programs use the common AGX NIR sample loop at two
or four samples. Sample ID, position, API coverage, shader coverage outputs,
discard, depth, and tile accesses are lowered together. The sample-position
word comes from the same helper used to program PPP. Disabling multisample
rasterization selects center inputs while retaining the attachment's sample
count and broadcast stores.

The interpolation-position operation has distinct centroid and sample modes.
Centroid consumes raster coverage; sample mode consumes an allocated sample
index. Both return an allocated packed position for explicit-coordinate
coefficient evaluation. A one-bit coverage mask in centroid mode does not
substitute for sample mode on T8132. Minimum sample shading uses the state
tracker's per-sample input variant and this same coefficient path.

T8132 readback checks cover single-sample targets, two- and four-sample targets,
per-sample builtins, discard, depth comparison, sample masks, repeated mask
assignments, additive blending, minimum shading, and disabled multisampling.
The official GL33 multisample fragment-coordinate case also passes. The host
compiler test exercises sample inputs, color, depth, and coverage outputs.


## Textures in compute and geometry stages

Texture-using compute programs share the normal graphics texture lowering and
an indirect buffer table. Their six argument roots are texture descriptors,
sampler descriptors, compacted buffer addresses, group counts, shared memory,
and a reserved zero pointer. This ABI supports 32 combined buffer resources;
buffer-only programs with at most 18 retain the direct ABI. Loads, stores,
atomics, preambles and shared accesses all use the selected resource map.
The compute launch frame includes the compiler's publication requirement.

The driver uploads the shader's normal sampler mapping and graphics sysvals
for API vertex/geometry programs lowered to compute. T8132 readback tests cover
vertex-only, geometry-only and combined texture fetches, variable emitted
primitive counts, transform feedback, and primitive queries across partial and
multiple workgroups. Compiler tests combine fetch/filtering with preambles,
shared memory and group counts, and cover both resource ABIs at their limits.

## Layered rendering

The vertex export layout packs the layer in the low 16 bits of a dedicated
layer/viewport word after user varyings and point size, before clip distances.
PPP enables that system output and includes it in the export stride. Fragment
`load_layer_id` reads the raster layer through GET_SR selector `0xc3`; workgroup
Z is a separate value and does not identify the layer of an end-of-tile helper.
Geometry copy shaders and internal layered clears use this ordinary path.

Layered reloads add the resource's layer stride to their existing mip address.
Native array PBE stores use the array coordinate and page-aligned layer layout.
For mipmapped layered allocations, the generated end-of-tile shader reads each
tile pixel and writes its explicit resource address. This GPU fallback avoids
assuming a native bulk-store mip selector; it is slower than bulk export.
Single-layer mip views continue using a rebased PBE descriptor.

T8132 pixel oracles cover layered color and depth, geometry primitive IDs,
per-layer clears, partial tiles, reloads, blending, RGBA8/RGBA16F, 2x/4x MSAA,
and array/3D mip levels. Host tests check layer outputs together with point
size and clip distances, plus sparse image bindings in every bulk-store mode.


## Implicit fragment primitive IDs

When a fragment shader reads `gl_PrimitiveID` without an API geometry shader,
Apple9 inserts Mesa's common passthrough GS with primitive-ID forwarding. The
Poly lowering normalizes its scalar primitive-ID input to the same intrinsic
used by GLSL geometry shaders. The generated raster shader exports the ID as
an ordinary flat integer varying. Its cache distinguishes shaders with and
without primitive-ID forwarding. Native generated primitive coefficients remain
unmodeled; this path uses the existing geometry implementation.

After tessellation, a geometry shader's `gl_PrimitiveIDIn` is the patch ID the
tessellation evaluation shader saw, shared by every primitive generated from
that patch, not the index of the tessellated primitive. The tessellator's
indices encode their patch. When the geometry parameters name the draw's
tessellation parameters, the Poly GS lowering loads its first input vertex's
tessellator index and derives the patch ID with the same helpers the
evaluation shader uses. Otherwise it uses the input primitive's index. Drivers
set the field only for tessellated draws. The evaluation shader writes nothing
extra, and only geometry shaders that read the ID pay for it. This also covers
the passthrough GS, which uses the same lowering
(`dEQP-VK.geometry.builtin_variable.primitive_id.matching`).

The passthrough is inserted before transform feedback and primitive conversion,
so it receives the original topology and captures only once. Queries on draws
with primitive restart also use geometry assembly, excluding restart markers
from generated-primitive counts. T8132 checks cover points, lines, triangles,
strips, fans, adjacency rasterization, instancing, restart, repeated shader
switches, captured point data, buffer tails, and both primitive query counts.
The core transform-feedback validator now admits only the documented triangle
modes without a GS, while preserving compatibility-profile quad/polygon modes.

Native unary integer operations use the eight-byte conversion-family operand
layout. Population count, unsigned find-MSB and bit reversal preserve ordinary
GPR liveness and consume pending loads through the shared dependency mask.
The T8132 hardware tests cover all 16-bit values, every bit position, random
32-bit inputs, zero (find-MSB returns -1), retained sources and 96-register
pressure. Signed find-MSB and find-LSB use the common NIR composition around
the native unsigned operation. Narrow inputs are explicitly zero-extended;
64-bit inputs use common pair lowering. Diagnostic GLSL tests enable
`ARB_gpu_shader5` privately so the GLSL frontend preserves find-MSB instead
of choosing its older float-exponent fallback; this does not advertise that
extension in the driver.


Binary16 arithmetic uses the native half add, subtract, multiply and FMA
families. Values occupy the low half of word-allocated SSA registers; the high
half is undefined until extended. Half min/max, comparisons and predicates,
rounding, reciprocal, reciprocal square root, exp2 and log2 also have native
forms. Square root uses a native half-input, float-output special factor and a
mixed float/half multiply with a half result. Division uses the same mixed
multiply with a float reciprocal of the half divisor, avoiding premature half
reciprocal overflow. Sign changes remain exact bit operations.

T8132 probes cover every half bit pattern, random and edge-case operand pairs,
retained sources, comparison-controlled branches, and 80 simultaneously live
values reaching all 96 word registers. Add/subtract/multiply/FMA, rounding,
min/max, square root, reciprocal and division match the independent binary16
CPU oracle. Native exp2 and log2 each differ by one half ULP on one tested
finite input; they are hardware approximations, not correctly rounded
functions. The public [FP16 encoding work](https://github.com/pac85/applegpu/commit/bb97333d6c914289bef9a7d5e73797627d804e6f)
provides an independent source for register-half, lifetime and modifier fields.
The opcode family selects the arithmetic datapath; source and destination
precision fields in the float datapath also allow mixed-width operations.

Fragment primitive IDs use a rasterizer-generated coefficient and the matching
VDM/fragment state enables. User varyings, depth and point coordinates retain
separate coefficient slots. A geometry shader supplying its own primitive ID
continues to use its ordinary output varying.

Explicit texture gradients now publish native parameter blocks for 2D arrays,
3D textures and depth comparisons. The allocator reserves eight words for the
2D forms and sixteen for 3D, even though the 3D operation publishes only nine
coordinate/derivative words. Separate live gradient operations retain separate
blocks. Hardware validation covers nonzero mip gradients in compute and
fragment shaders, including two simultaneously live 3D samples.

Layered mip attachments use native PBE block exports with the complete mip-tree
base and independent selected-level fields. Color reload shaders use texture
fetches, including per-sample fetches, then convert to the tile representation.
The texture unit supplies addressing and format conversion. Texture descriptors
are constructed from the current view for both graphics and compute uploads.

Indirect draws consume GPU argument buffers through native VDM commands, with
all three index widths, first-index offsets, signed base vertices, instance
counts and primitive restart. Instanced vertex fetches read base instance from
the draw-parameter UBO, which can point directly into an indirect descriptor.
The original shader's draw-ID usage survives its conversion to compute for
geometry processing. Multidraw count uses the existing GPU predication helper;
transform feedback on indirect draws uses the geometry path without mapping
arguments back to the CPU.

CDM mode 1 reads three workgroup counts and takes local sizes from the command.
Mode 2 reads six words containing thread counts followed by local sizes. The
latter supports the partial groups used by indirect geometry processing. Its
shader-visible workgroup counts are rounded separately on the GPU. This avoids
interpreting a five-vertex input as five full workgroups.

### Indexed private arrays

Large function-local arrays use native indexed private loads and stores. As on
Apple8, direct accesses are promoted to SSA first; small indirect arrays use
NIR decision trees. Scratch legalization handles narrower values through
32-bit words. The native index and displacement both count words. Stores
consume their data operand and retain their index, so ordinary allocator
constraints preserve later uses. Loads complete through slot 6, and the physical
scheduler retires a pending private read before any private write.

Register spills occupy the beginning of the scratch frame, with indexed arrays
after the aligned spill region. The allocator fixes this displacement after
spilling and checks the combined 32 KiB limit. Vector spill moves lower every
component to a scalar save or reload. Compact FILL addresses only the first
1024 words: higher offsets alias lower words on T8132. Reloads beyond that range
use indexed private loads through the reserved copy register, which also bridges
the indexed encoding's low-register constraint. Compact SAVE addresses the first
2048 words; larger stores use indexed private stores with a separate reserved
zero-index register. Native SSBO and authored spill probes cover frames through
31,680 bytes, including 8,001 distinct outputs across three submissions. Authored hardware probes validate distinct
per-invocation contents, register operands through r63, dynamic and constant
word offsets through 8191, and the shared frame layout with SAVE/FILL. Native
GLSL vertex and fragment tests cover 2–32 KiB frames and repeated aliased access;
the original giant-array and fragment-predication Piglit tests also pass.

## Flat interpolation and triangle fans

Flat user inputs lower to `load_coefficients_agx` and select a word from the
allocated three-word coefficient tuple. The first-provoking triangle-fan mode
uses shade selector 6 and word B; ordinary first/last flat modes use word C.
This matches the common AGX interpolation contract. A per-draw system value
selects B/C without recompiling the fragment shader when topology changes.
Rasterizer-generated primitive IDs retain their independent coefficient source.
The coefficient-table reuse key includes the effective provoking vertex.

Native T8132 CTS validation covers Vulkan fan flat shading, provoking vertices,
indexed/indirect/instanced fans, and GL/GLES flat-shading and fan controls. The
allocator regression exercises both B and C with overlapping asynchronous
coefficient loads. Gallium no longer needs primitive conversion for first-flat
Apple9 fans.

## Texture operands and register-indexed access

The texture operand is an 8-bit texture-state index split across bit 71
(bit 0), bits 11-14 (bits 1-4) and bits 68-70 (bits 5-7); the sampler operand
is bits 32-35 (bits 1-4) and bit 72 (bit 0). Both index the tables named by the
launch header's first two pointers, with a 32-byte stride.

The 20-byte form (bit 73) can instead read either index from a GPR, as Metal
does for Tier-2 argument buffers. Metal's resource IDs are small dense
integers indexing a device-wide descriptor table, which it installs as the
header's texture (or sampler) table pointer; its entries are byte-identical to
the per-draw copies. Decoded from our own compiled kernels and validated on
T8132 by per-lane selection probes:

- texture index register: bits 11-14 name its half-register, reaching r0-r7;
  bits 68-71 are clear, bit 87 is set, byte 15 bit 5 enables the
  register and byte 15 bit 3 discards it at its last use;
- sampler index register: bits 32-34 name its half-register, reaching r0-r3;
  bit 72 is clear, bit 88 is set, byte 14 bit 7 enables the register and
  byte 14 bit 5 discards it at its last use;
- byte 18 is 0x40 in every observed register-indexed operation.

Bits 39-41 are a completion-group wait, independent of the descriptor source.
Our packed coordinate publishers use group 0, so both texture forms retain
the encoded wait value 1 (bit 39). Clearing it in the register-indexed form
raced coordinate publication: occasional waves sampled the right image at
`(x, 0)` or `(0, 0)`. A standalone wait before the sample also resolves this,
but the texture instruction's own dependency field avoids an extra instruction.
The native bindless test with several distinct source images reproduces the
failure; the constant-index control stays correct. The Heaven replay's replaced
tone-map shader likewise returns to its 12 legitimate isolated black blocks.
In an unmodified replay, frame 2 went from 87 isolated black 8x4 blocks to zero.

Only the low 32 bits of the index are read. A discard bit set while the
register remains live corrupts a later reader of the same value.
Honeykrisp points both table pointers at the USC aperture base, where G16
descriptor memory lives, so a descriptor's native entries are addressed as
(descriptor VA - base) / 32.

## Cube arrays

Our own Metal kernels, compiled once per texture type and operation, give the
cube-array mode for bytes 6/7: `08 03` for a register LOD, the cube mode
`00 03` with the array flag. Honeykrisp uses `1c 00` for its biased form.
The face-selection sequence is shared with plain cubes. The packed LOD word
carries the cube index in bits 0-11 and the face in bits 12-14. The Q6 LOD
remains in bits 16-27. The descriptor keeps the cube-array dimension, and its
layer-count field holds the number of cubes. The hardware clamps the cube
index to that count. Per-texel probes of layer, face and level matched the
2D-array emulation, including gathers and non-seamless samplers.

With native cube arrays, hardware derivatives give implicit LODs, as for
plain cubes. Implicit-LOD depth comparisons are the exception: their biased
form is not yet decoded. They still take direction derivatives and use an
explicit LOD. LOD queries supply layer 0 as the array operand.
