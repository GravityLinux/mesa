# Apple9 parallel phi copies

Loop-carried phi assignments are simultaneous. Previously Apple9 emitted each
assignment as soon as its source was lowered, so a backedge `a = b; b = a`
overwrote `a` before the second copy read its old value. The unmodified Piglit
`copy-propagation/glsl-copy-propagation-loop-1.shader_test` returned yellow
(1,1,.75) instead of (1,0,.75).

Phi-edge lowering now materializes all source expressions first and resolves
the complete assignment group. A destination is written only when its old
value is no longer needed by another pending source. Cycles preserve a source
in an ordinary allocator-managed integer-OR bit-copy temporary. Copies retain
the predecessor's execution mask. No fixed registers or shader-specific rules
are used, and non-cyclic assignments need no cycle temporary.

The compiler unit test `ParallelPhiAssignmentsPreserveAllSourcesAfterAllocation`
checks all 256 four-source assignments by executing the allocated copy sequence
and comparing it with simultaneous assignment. Values include NaN and sign-bit
patterns to check bit preservation.

Hardware fixtures use Piglit shader_runner's default 250x250 framebuffer:

- `rotate-divergent.shader_test`: three-way rotation, with one or two iterations
  depending on fragment X. Passes T8132 and llvmpipe.
- `rotate-four.shader_test`: four-way rotation with zero, one, and three
  iterations. Passes T8132 and llvmpipe.
- `swap-vertex.shader_test`: vertex swaps with odd and even uniform iteration
  counts. Passes T8132 and llvmpipe after the branch-relocation fix below.

After resetting/chainloading the standalone m1n1 image, from the local workspace:

```sh
bash tmp/apple9-piglit-graphics/run.sh \
  piglit/tests/spec/glsl-1.10/execution/copy-propagation/glsl-copy-propagation-loop-1.shader_test \
  mesa-m1n1-shim/src/asahi/drm-shim/scenes/phi-copies/rotate-divergent.shader_test \
  mesa-m1n1-shim/src/asahi/drm-shim/scenes/phi-copies/rotate-four.shader_test
```

Evidence and full regression manifests are in `tmp/apple9-phi-copies` at the
workspace root. This change fixes the swap; it does not resolve all loop issues.
The subsequent branch-relocation fix also resolves the constant increment/
decrement tests and the vertex swap. See `tmp/apple9-loop-investigation`.

## Branch relocation during memory-tuple legalization

A later pass inserts COLLECT instructions to assemble scalar address/store
sources into register tuples. It must relocate existing branch targets: a
branch to the affected memory consumer includes its new COLLECT, while later
targets shift forward. Previously targets were left stale, so loop backedges
could reexecute preheader accumulator/counter initialization and forward exits
could land on the wrong instruction.

The original increment/decrement tests returned about 0.1 instead of 1.0, and
the vertex swap timed out. All three pass on T8132 with branch relocation.
`MemoryTupleLegalizationPreservesControlFlowTargets` tests repeated insertions,
forward/backward branches, exact insertion points, and end-of-program targets;
it fails before the fix and passes after it.
