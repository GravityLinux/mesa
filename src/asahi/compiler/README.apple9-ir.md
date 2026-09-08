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

After legalization, CFG construction splits blocks at conditional branch
terminators and records successors and predecessors. Masked regions execute in
layout order; explicit branches additionally have their target edge. Branch
byte offsets are assigned only after all variable-length instructions and phi
copies have been emitted. Empty blocks use the next emitted instruction's offset.

## SSA and allocation

A PHI declares one SSA value in its successor block. Selection creates its
identity before selecting incoming edges, then places the declaration in the
successor once structured selection has placed the blocks. PHI_SRC records are
incoming uses, naming the source SSA value and the destination phi. All records
for one predecessor edge share a stable `phi_edge` identity. No sequential writes
or cycle-breaking virtual temporaries are introduced during selection.

Liveness is a backwards fixed-point analysis over the CFG. Phi edge sources are
uses at their predecessor's copy site. Masked edges also preserve the destination
storage for inactive lanes until reconvergence. This is important because the
machine CFG traverses both arms of a divergent conditional in layout order.

Allocation uses conservative enclosing intervals derived from CFG liveness.
A phi's physical storage is reserved at its earliest incoming copy site while
its SSA definition stays in the successor. Every source in a parallel edge stays
live through the entire assignment. Ordinary scalar and adjacent-tuple register
constraints and the separate publication namespace remain in force. This is
still a no-spill allocator; it does not yet exploit holes in live intervals or
implement Apple8's full SSA allocation strategy.

After allocation, `agx_apple9_resolve_phi_edge` uses the existing Apple8
`agx_emit_parallel_copies` scheduler. The adapter converts 32-bit GPR indices to
Apple8's half-register units and translates the resulting physical MOV/SWAP
operations into Apple9 IOR/IXOR instructions. Cycles require no reserved scratch
register, and copies preserve all bits, including NaN payloads. Pending returns
are materialized before phi edges, so the physical copy scheduler handles
ordinary GPR values only.

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

## Shared uses and dependency finalization

`agx_apple9_analyze_uses` builds a shared definition table and ordered operand-use
chains for a stable IR snapshot. Producer, consumer, publication-lifetime, and
device-load index queries use this analysis. Structural mutation APIs invalidate
it automatically; passes that edit operands directly must call
`agx_apple9_invalidate_uses`. Use-chain pointers must not survive a rebuild.

Use order identifies consumers; CFG liveness determines whether an index must
remain live, including across loop backedges. Device-load index contracts and
SFU result hints are finalized after legalization and allocation. Subsequent IR
edits invalidate the finalization flag, and allocation validation rejects stale
dependency metadata. Only physical pseudo expansion and byte packing follow
finalization in the compiler pipeline. The SFU hint's existing conservative
memory-only/other-consumer distinction remains; this analysis change does not
establish additional hardware semantics for that field.

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
