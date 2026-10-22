# T8132 U0 fixed producer and U input waits

The r6 device result (device runner `b1e454ba0f9`,
`results/mesh-u0-r6-v2/RESULTS.md`) establishes the compiler-visible
relation: either slot 1 or slot 2 completes U0's dependency before the first
channel submit. Both embedded masks rescue N32 128/128. A matched wait just
before that submit passes 128/128 and just after fails 128/128; both sides of
the first post-U0 store pass. The ordinary-load probes independently establish
bits 12..17 as the U input wait mask on control, scalar and channel forms:
slot-1/slot-2 right-mask probes pass 128/128 each, with stale zero/wrong controls.

H4 (one U0 completion class with two selectors) and H5 (coupled completions)
remain observationally equivalent. The compiler chooses slot 1, mask 0x01.
Slot 2, mask 0x02, is an alternative selector for this producer; the two masks
are not combined into a required 0x03 wait. Ordinary slot-1 and slot-2 loads
remain distinct, as r5 ALIAS and the r6 input probes demonstrate.

## Machine model and scoreboard

`fixed_producer_slot` names the canonical implicit slot. The same descriptor
now holds the existing async SR/shared-atomic slot 1 and shared/private-load
slot 6 assignments. `MESH_CONTROL` has canonical slot 1 and
`fixed_producer_alternates = 0x02`. Neither field is an encoded output tag.
U0 has no GPR result; its A read/clear and B read remain ordinary operand
contracts, and both inputs are consumed at issue.

U0 enters the normal pending-producer list exactly once. Its live reservation
covers both selectors, so programmable producers cannot allocate slots 1/2
until U0 completes. This reservation is conservative: reuse of a live U0 slot
has not been qualified. A fixed-slot collision retires it before issuing the new
producer. Issuing U0 also resolves existing ordinary producers on either
reserved slot. All other fixed producers retain their previous assignments.

The first channel submit is a producer hazard, and both submit forms retain
their general publication completion-barrier contract. The normal fold-or-WAIT
path consumes the pending slot, with the selected slot encoded in the U input
wait mask. An explicit fence, halt, other established completion barrier,
loop boundary or slot collision reached earlier retires U0 through the same
scoreboard. Fields are used only where their encoding is established.

The native lowering's global-memory fence moves immediately before U0. It
still follows the completed output writes and precedes every submit. U0 does
not write those global output planes. This separates output-memory publication
from U0's allocation completion, allowing the normal first channel to fold its
wait. A source-program barrier encountered after U0 retains its usual meaning.

Forward CFG states still union pending instructions; loop edges drain. A wait
is applied to individual producers, including an authored WAIT. At a join,
slot 1 may contain U0 from one predecessor and an ordinary load from another:
waiting on slot 2 removes U0 but preserves that load. A canonical slot-1 wait
can complete both alternatives. Overflowed pending lists retain the existing
entry-drain rule. No simultaneous U0 instances need hidden token accounting.

The scalar region can inherit U0 along an edge that skips the channel region.
Its first completion barrier then folds slot 1 too. This is a CFG obligation,
not a second retirement inserted along a straight-line already-retired path.

## Repeated U0 and limits

The fully waited double-U0 N1 r6 control timed out, and no token-count or
rearming arm qualified. VIR validation rejects more than one MESH_CONTROL in
a program, even with intervening waits or mutually exclusive branches, and
rejects U0 inside a NIR loop or a backward-branch span. The shared allocation
entry point applies this guard before scheduling; it is a compile error, not
an opportunity to insert a wait. Normal lowering emits one U0 outside loops.
The scheduler's fixed-reservation ordering remains a backstop.

The production-r6 device check also timed out with the racing CURRENT_U2_M00
negative control as the oldest stalled Work. Its 16 completed CURRENT smoke
draws passed, but the 128-trial production block was never reached. This does
not qualify production or establish the root cause of the hang. Together with
the double-U0 timeout, it makes the repetition guard and folded U2 wait
correctness requirements, not performance choices. No replay sequence or
queue-depth claim is introduced into the production compiler or device kit.

The old `completion_effect_mask`, separate effect state, and unconditional
0x03 retirement are removed. U0 is not a blanket completion barrier for
unrelated slots; its inputs and reservation conflicts determine what it needs.
Existing `requires_prior_completion` contracts on actual publication/barrier
consumers remain meaningful independently of U0.

The unit tests cover both alternative selectors, distinct ordinary loads at
joins, normal folding, early barriers, explicit waits, slot collisions and
exhaustion, rejected repetition and loop placement, CFG bypasses and exit. Raw and VIR encoders
round-trip U input masks without changing operand/control bits. The complete
native-fixture checker rejects missing/wrong U2 completion as well as its
existing operand, tuple, halfword and pressure mutations. Compiler review and
the small production v2 device check are separate from r6's probe evidence.
