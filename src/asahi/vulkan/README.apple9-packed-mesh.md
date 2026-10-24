# Packed index publication: qualified opt-in policy

The qualified mode-4 path is **disabled by default**. Set
`HK_APPLE9_NATIVE_INDEX=1` to opt into the 31 exact declarations in
`hk_apple9_native_index_shapes.def` on G16G. This also requires a positive
kernel advertisement sampled around each VM creation. A missing, malformed,
research or changing advertisement falls back to emulation. T16V16P16 is held.
The Position-only, triangle, fixed T x 1 x 1 predicate still applies.

`HK_APPLE9_NATIVE_INDEX=0` or drirc `hk_disable_native_index=true` disables
this policy. The latter may be set per application. An explicitly present
`HK_APPLE9_NATIVE_MESH` retains the separate research configuration: `0` opts
out, and `1` selects the research route and its explicit class list. A shape
list alone cannot enable or widen qualified admission.

The physical-device policy participates in compiler/cache identity. Each
logical VM must independently qualify before native draw recording. Native
preflight examines a clone and compilation retains the software realization.
Default settings preserve ordinary shader code and data. This series lands
the implementation; it does not enable the 31-class default.

## Original research control

The explicit research route defaults off. This caller prepared device testing after native-capability v3;
no v3 result or production enablement is assumed. It sits on compiler proposal
3e86031221a, reviewed as default-off research in compiler review. Its arithmetic
bound is a compiler constraint, not a driver capability range.

`HK_APPLE9_NATIVE_MESH=1` enables the existing hybrid. The additional option
`HK_APPLE9_NATIVE_MESH_PACKED_SHAPES` takes a comma-separated list such as
`T16V3P2,T16V4P2`, selected only after the matching v3 classes are qualified.
Unset/empty enables no new class. Boolean values, wildcard/range syntax,
unknown tokens and malformed lists enable none. Do not enable all table entries
because some v3 shapes pass: a shape must be selected individually.

The candidate universe is the 32 triangle declarations in the frozen v3 S2
catalog, listed in hk_apple9_packed_mesh_shapes.def. That table is deliberately
not a qualification list. Each tuple fixes physical T, maximum V and maximum P;
there is no inference from V<=T or ceil(3P/4)<=T. In particular T16V16P16 is
excluded unless explicitly selected after its v3 evidence and the S6/S7 caveat
have been assessed. An untested T16V16P15 or T32V3P2 cannot be selected.

`hk_apple9_native_mesh_packed_shape_eligible` also requires triangles, no task
payload, Position-only output, no API shared allocation, fixed W x 1 x 1 API
workgroups and subgroup size 32 (or unspecified), with the actual selected
physical width W. Existing P1 logical expansion/serialization, attributes and
primitive broadcast stay on their existing path. This deliberately narrow
candidate can leave CTS shaders on emulation; native route logs distinguish
that from candidate execution.

In the explicit research route, the predicate controls early output-deref
lowering. In both routes it controls the options used
by both immutable creation-time preflight and actual compilation. Candidate
options disable attributes/primitive broadcast and enable the generalized
compiler rule. Unsupported shaders keep the existing software realization.
Postcompile/upload validation repeats the selected class, logical width and
Position-only layout checks, including count header, pitch, size and combined
allocation. No kernel UAPI or exposed limits change.

The full selection value keys the compiler cache. Its parsed mask also keys
native graphics-state hashes. `HK_APPLE9_MESH_ROUTE_LOG=1` reports the final
selected draw path and T/V/P without affecting code-generation keys. A route
message records selection; completed pixel validation is still required.

Host tests exercise parser rejection, V16/P16 exclusion/explicit selection,
wrong-width and in-bound-but-unlisted declarations, immutable preflight, actual
hk compilation, upload admission and layout-derived launch/state fields.
Existing native mesh/options/hybrid/routing suites remain in the focused set.
The Air kit will carry one private ICD for base-versus-opt-in comparison,
explicit v3 qualification input and a compatible-kernel preflight. It must not
install Mesa, change kernel mode, reboot, or queue itself.
