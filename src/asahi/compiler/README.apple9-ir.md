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
values can occupy the remaining registers through r63. Instruction-local
constraints describe each operand's legal range, alignment, fixed register,
and destructive-input behavior. The shared allocator places operands in their
legal banks and relocates live occupants with a parallel copy when necessary;
there is no post-allocation bank-repair sequence.

Destructive inputs receive private SSA lifetimes. Move coalescing removes the
copy when the original value dies at that instruction; otherwise the copy
preserves the original. The currently encoded extended shift releases its
source, as do the modeled atomic inputs. General sources are late-killed so an
instruction's destination cannot unexpectedly overwrite its inputs. Shared
SSA kill information supplies Apple9 source-release bits. Pure opaque operations
can be removed when dead, and redundant identity copies are eliminated.

After allocation, Apple9 lowers physical MOV/SWAP and memory moves. A SAVE
releases its data register, so a live allocated value is copied to the reserved
copy temporary first. FILL retains the scratch word for later uses and loop
iterations. The physical completion scheduler tracks pending operations and
places waits at consumers or hazards, folding a wait into a suitable consumer
when its operand lifetime permits. It also accounts for register overwrites,
scratch aliases, device-memory ordering, and borrowed publication inputs.
Execution-mask changes, block boundaries, and operations with implicit hardware
dependencies retain conservative completion boundaries. Atomic publication
records stay adjacent to their producer. Tags are retired before reuse.
The scratch frame is rounded to 16 bytes and bounded at 4 KiB per invocation.

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

## Varying commit scheduling

Vertex varying stores are scheduled as a contiguous tail of their execution
region, after calculations and loads. Their publication producers stay in place;
the separate publication namespace retains those values until the stores run.
The pass runs before dependency assignment and allocation, preserves store order,
and never moves a store across a block boundary or execution-mask operation.

On T8132, dense indexed grids exposed intermittent missing primitives when
varying stores were interleaved with later device loads. The same Mesa shader
reproduced this through public Metal submission after replacing only the bounded
main symbol in an archive compiled from authored MSL. Moving only its stores to
the tail passed 5,000 draws; a compiler-generated version passed another 1,000.
Native shader controls and changes to export hint bits isolated the scheduling
change. This is an observed scheduling constraint, not a claim that the complete
varying-store synchronization protocol is understood.

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
