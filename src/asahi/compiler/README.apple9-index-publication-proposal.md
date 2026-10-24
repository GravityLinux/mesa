# Native mesh packed-index publication proposal — v2 evidence, 2026-10-23

The smallest supported change is a publishing-lane prefix with a uniform
source base. A single lane is insufficient for the tested multi-primitive
classes. A varying per-lane base is unsafe at V16/P16. Neither changing the
first scalar operand nor adding a completion dependency is established as the
fix. This proposal remains default-off and is narrower than the research kit.

## Evidence and competing explanations

For the six discriminator shapes with all S0–S7 policies, S2–S5 pass both
repetitions. S0/S1 fail every multi-primitive shape. S6/S7 pass the five smaller
shapes but fail V16/P16. The two additional comparable rows are V3/P1 W16/W32,
where S0 and S1 both pass; S2–S7 were not run there. The high-index V4/P1 row
passes all policies. Thus the evidence does not select V versus K*P as the first
operand or P versus ceil(K*P/4) as the exact number of required lanes.

S0/S1 issue from lane 0 with base 0. S2/S3 use ceil(K*P/4) lanes with base 0;
S4/S5 use P lanes with base 0. For packed byte indices and K<=3, both prefixes
cover at least ceil(K*P/4) lanes. S6/S7 use the word-sized prefix but supply
4*lane as the second operand. Their small-shape passes prevent us from claiming
a proven byte-address equation such as base+4*lane. For example, transfer
alignment/grouping could hide low offset bits for short arrays. The tests do
not distinguish that explanation from other collective granularity or bounds
semantics. Do not encode any of those extra claims in the compiler.

The 19 changing raw images are confined to two S0, sixteen S1 and one S6 arm;
none is S2–S5. All 264 completed Q mirrors and uploaded bodies/state are exact.
Incomplete or incorrectly addressed publication can leave downstream output
undefined and produce this variability with stable input memory. That is a
single explanation consistent with all eight comparable rows and all 19
unstable arms. It is not proof of hardware reading uninitialized local input.

A missing wait is still an alternative, but it is not implied: the first
channel already waits for U0, and no new later-U producer slot has been
identified. Wider participation can also change timing, so successful S2 alone
cannot conclusively eliminate a completion race. V3 pairs S0 and S2 with
full-mask U inputs plus the existing final WAIT on three discriminator shapes.
These are research controls; production keeps its current dependency model.

## Candidate and bounded implementation

For normalized runtime vertex count V, primitive count P, and topology size K:

```
participants = max(1, ceil(K * P / 4))
if physical_lane < participants:
    submit_scalar(V, 0)
```

The empty path keeps the existing one-lane zero-count handoff. The first scalar
operand stays V because v2 does not distinguish it from K*P. Base zero is shared
by all publishing lanes. The compiler owns packed indices at local offset zero;
this is an ordinary local-layout contract, not a capture address or GPR choice.

The new compiler option `packed_index_publication` is false unless explicitly
supplied. With it, admission requires triangle topology, no task payload,
position-only outputs, physical T16 or T32, Vmax<=physical width and
ceil(3*Pmax/4)<=physical width. Thus the first candidate ceiling is V16/P21 or
V32/P42. Runtime counts remain bounded by declarations. Multi-round publication,
wide physical launches, per-primitive outputs/cull transport and extra varyings
are excluded from this proposal. A declaration larger than this class must
continue through the software mesh path; do not silently truncate lanes.

`agx_nir_apple9_index_publication_lanes` builds the generalized count expression.
Both the proposed normal NIR completion and the S2 research source use it.
Normal lowering/RA selects registers and encodings. No fixture offsets,
fixed-register assignments, replay bodies or experimental wait masks enter the
production compiler. Existing fences, single U0 and its dependency are retained.

The driver currently still admits only its existing V3/P1 native shape:
`agx_apple9_native_mesh_shape_eligible` and all environment defaults are unchanged.
No driver caller sets the new compiler option. This patch is a concrete,
reviewable compiler proposal, not a Vulkan eligibility expansion. Generic
body-only upload protection remains unchanged.

## Integration after device evidence and compiler review

After the v3 narrow candidate passes, propose a separate explicit driver gate
under the existing default-off native-mesh hybrid. Its shared early-IO and
compile preflight must agree on the same narrow triangle declaration class and
select software fallback elsewhere. Include the policy in cache identity if
it becomes selectable. Check layout-derived VDM primitive bounds, row count,
launch allocation, PPP output count and object/mesh width together. The state
builder already derives those fields from the declared layout; their presence
does not qualify a broader completion contract. Keep primitive broadcast on
its existing single-primitive path. Do not enable W64 or increase public mesh
limits from this result. No default flip is part of either patch.

## Host checks and review

Seven focused compiler suites pass, including the new test's 387 count cases,
six whole normal-NIR program compilations through V32/P42, four out-of-class
rejections, and unchanged disabled/legacy gates. The other suites cover native
mesh, runtime counts, shared memory, logical workgroups, object and primitive
paths. These tests establish compiler behavior, not GPU consumption semantics.
V3 host submission/oracle/parser validation and the separate compiler
review are required before queueing the final kit. Device qualification remains
pending. No missing result is treated as a pass.
