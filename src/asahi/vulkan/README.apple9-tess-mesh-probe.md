# Apple9 tessellation via native mesh: stage 1

`HK_APPLE9_TESS_MESH_PROBE=1` enables a compiler-only diagnostic in the ordinary
TES compilation path. It clones the immutable TES, lowers it to ordinary mesh
NIR, compiles it with the existing native mesh backend, reports
`HK_TESS_MESH_PROBE compiled=… bytes=… submitted=0 reason=…`, and frees the result.
The usual TES compilation continues on the original shader. The option defaults
off. It changes neither device features nor draw routing, launches, queues,
statistics or the tessellator. Its value participates in the compiler cache key
so a cached fallback cannot silently conceal a requested probe.

For a host compiler probe, also select the existing native mesh backend and
shape with `HK_APPLE9_NATIVE_MESH=1` and
`HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1`. These settings do not make a TES
submit through native mesh. The new helper contains no instruction replay,
captured state, fixed register assignment or alternate backend.

## Contract

The initial scope is lowered, inlined, Position-only TES for triangle or quad
domains, with no point mode, XFB, memory writes, clip/cull outputs or patch
varyings. Constant Position input indices must be below 32. Tess coordinates,
patch ID, patch vertex count, tess levels and read-only uniform loads reuse
existing poly lowering. Unsupported intrinsics and I/O return a reason without
changing the source. The ordinary backend may reject further unsupported code.

The existing tessellator still computes factors, partitions the domain and
writes the index/coordinate/TCS buffers. `libagx_tess_alloc_draw` writes the total
index count to `poly_tess_params.out_draws[0]`, and `index_buffer` already points
to the allocated index stream (do not add the indirect draw's heap-relative
start again). One logical mesh workgroup consumes each complete triangle from
that stream. Its first three lanes evaluate the three TES corners, with the
existing `poly_load_tes_index` winding swap and existing coordinate/input
layout. Each workgroup declares T16/V3/P1 and emits indices `(0, 1, 2)`; lane
zero publishes counts. This may evaluate a shared tessellated vertex repeatedly,
so side effects and future statistics routing need separate treatment.

Bounds are applied to accesses and output stores, not just final counts:

- Only lanes below 3 in the one-dimensional dispatch read metadata.
- Compare group x against `index_count / 3` before calculating `3*x + lane`.
  This rejects oversized groups without overflow and ignores partial triangles.
- Require the runtime TCS patch size to cover every statically referenced input
  and to lie in 1..32 before coordinate, index, or TCS-buffer reads.
- Wrap the entire lowered TES body and all mesh stores in that predicate;
  rejected groups publish 0/0. Unused lanes and nonzero y/z groups perform no
  metadata or payload memory reads. Nonzero y/z groups publish 0/0 from lane 0.
- Preserve component and write masks for partial Position stores.

Index contents, patch-coordinate allocations and buffer backing retain the
existing tessellator producer contract. These checks bound dispatch and static
patch references; they do not validate corrupt producer data or allocation
failure. Device synchronization, draw admission, dynamic indices, statistics,
non-Position interfaces, isolines and points remain outside stage 1.

## Host validation

`hk-tes-mesh` specializes the actual lowered NIR against sparse test memory and
checks the resulting load addresses, Position components, primitive indices and
counts. It covers zero/partial/full counts, winding, triangle and quad domains,
unused lanes, invalid y/z slices, patch sizes 0..33, maximal 32-bit counts and
oversized group IDs. It also checks immutable rejection and compiles eight
triangle/quad × input/no-input × full/partial-output variants through the normal
native mesh backend. No GPU instructions execute in these tests.

Strict shipped-code identity compares the pre-change baseline with the probe
unset and explicit 0. The harness requires a successful entry receipt,
balanced assembly receipts and hashes every final shader upload including its
preamble; initialization and discarded probe programs are not counted. Known
corpus assertions remain failures, recorded separately from successful entries.
Review artifacts and exact results live in the accompanying host stage-1 packet.

No tessellation device qualification or native tessellation submission is
claimed. The compiler probe must be reviewed independently before stage 2 adds
routing and device tests.
