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
active lanes in the target loop's predicate bank before mask unwind. Ordinary
conditional pushes consume bank zero; they do not establish the predicate
required by unwind. Publish after exit-phi source selection so comparisons in
those expressions cannot replace the break predicate. Omitting this write can
lose break-edge values or prevent loop termination.

The allocation CFG comes from the original NIR blocks and logical edges, not
from linear execution-mask layout. Both arms stay represented so edge copies
have the correct lane population. Synthetic loop-mask restoration blocks retain
their machine placement without becoming SSA predecessors. Branch byte offsets
are assigned after expansion. The physical program must be reindexed before
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
four (r0-r3) when spill/parallel-copy scratch temporaries are required. Ordinary
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
source, as do the modeled atomic inputs. General sources are late-killed so an
instruction's destination cannot unexpectedly overwrite its inputs. Shared
SSA kill information supplies Apple9 source-release bits. Pure opaque operations
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

After allocation, Apple9 lowers physical MOV/SWAP and memory moves. A SAVE
releases its data register, so a live allocated value is copied to the reserved
copy temporary first. A rematerialized literal destined above r63 also uses
that temporary followed by a full-register copy, since the long immediate
form has only six destination bits. FILL retains the scratch word for later uses and loop
iterations. The physical completion scheduler tracks pending operations and
places waits at consumers or hazards, folding a wait into a suitable consumer
when its operand lifetime permits. It also accounts for register overwrites,
scratch aliases, device-memory ordering, and borrowed publication inputs.
Execution-mask changes, block boundaries, and operations with implicit hardware
dependencies retain conservative completion boundaries. Atomic publication
records stay adjacent to their producer. Tags are retired before reuse.
The scratch frame is rounded to 16 bytes and bounded at 4 KiB per invocation.

Device-load completion applies to the full result tuple, so its first consumer
need not read every component. Compact float and extended logic consumers,
including their scalar export forms, can retain a load operand while retiring
its completion slot; later reads use the ordinary GPR path. Float exports expose
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
retaining that component and its siblings for later GPR reads. Other texture
consumer forms still materialize, and borrowed texture-parameter inputs remain
live until completion. EXP-M4-63 validates these address, SR and texture paths
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
loads. Selection converts the completed byte expression to a hardware element
index. It does not require an affine NIR expression shape or move arithmetic
across the conversion, which would change overflow behavior.

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
The fixtures in `../drm-shim/scenes/phi-copies/` exercise divergent rotations and
vertex swaps on hardware.

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
PBE descriptor. The publication is `(pixel_x, pixel_y, 0, tile_byte_offset << 16)`;
ordinary 2D stores use `(pixel_x, pixel_y, tile_byte_offset << 16)` in the same
four-register allocation. The zero word is part of the validated MSAA contract;
its other potential meanings are not modeled.

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

The ordinary NIR path extracts eligible uniform UBO expressions into a linear
setup function. Boolean and half intermediates may move with an expression;
transferred results are 32-bit words. The full resource map is fixed before
extraction and shared by setup and main, including resources used only in the
preamble. Unused vector components are trimmed before assigning uniform words.
Unsupported dependencies, masked setup, scratch, and setup bodies exceeding
2048 bytes retain the original main shader.

Words 48 through 63 of the argument window hold preamble results. Compute roots
and state occupy at most words 0 through 41; graphics roots occupy 0 through 11.
`IOR_UNIFORM` explicitly reads one word with a GPR operand, preserving integer
bits, signed zero, subnormals, and NaN payloads when used as a move with zero.
`STORE_UNIFORM` consumes its GPR operand. Its clobber constraint lets shared SSA
allocation preserve a value that remains live; the packer rejects a surviving
unpreserved source. Both forms have machine-table operand constraints.

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
`linux-m4-integration/tools/gpu/asahi/codegen-quality-fixes/REPORT.md`.
