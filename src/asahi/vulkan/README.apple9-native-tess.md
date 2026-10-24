# Apple9 native TES stage 2 (default-off draft)

`HK_APPLE9_TESS_NATIVE=1` additionally requires `HK_APPLE9_NATIVE_MESH=1`
and explicitly selected `HK_APPLE9_NATIVE_MESH_PACKED_SHAPES=T16V3P1`.
Stage 1's Position-only triangle/quad TES contract is unchanged. This is
unqualified device code pending a separate source and device-packet review.

The draw gate admits one non-indexed direct patch (1..32 input control points),
one instance, one view/layer/sample, and matching native VS/TCS/TES/FS leaves.
VS/TCS memory side effects, queries, conditional rendering, transform feedback,
non-fill rasterization, rasterizer discard, other primitive producers, and
unsupported TES declarations use the ordinary path. TCS output size may differ
from input size, subject to the existing 1..32 Vulkan limit. Native compilation
and cache ownership preserve the original API stages. A TES leaf owns a mesh
program and generated object, with API tessellation metadata stored separately.

The existing software VS and TCS feed the existing tessellator. Native draws
flush previous tessellation batches and record their five passes inline in
`pre_gfx`: VS, TCS, count, prefix sum, emit. Each dispatch retains its barrier;
the render has the existing full tiling dependency (`HK_READS_INTERNAL`). The
prefix-sum pass writes `poly_tess_params.out_draws[0]` and the index-buffer
pointer. Emit writes indices and coordinates. The object reads that same draw
word and launches `max(1, floor(index_count / 3))` groups. Each mesh group runs
one triangle's three TES corner evaluations using the existing index/winding
and TCS/coordinate ABI. An empty patch launches one group that emits zero
vertices and primitives. There is no CPU count read, synthetic replay sequence,
new kernel UAPI, or clamping of a generated grid.

The one-patch limit bounds the group count below the existing 65535 X-grid
limit. `src/poly/cl/tessellator.h` clamps even/integer factors to 64 and odd
factors to 63. Its per-edge point counts are at most 65 and ring counts at most
33. Dropping negative terms from its index-count expressions gives at most
13066 triangle-domain groups and 17552 quad-domain groups. These are conservative
bounds, not exact tessellation counts. The two domain special cases at factor
one are smaller. Increasing the patch-count scope requires a new grid proof.

Shared vertices are evaluated separately at every triangle corner: with T
triangles the native TES executes 3T corner evaluations. Relative to V distinct
domain points, the duplicated work is 3T-V and the amplification is 3T/V
(undefined for an empty patch). V is a reuse ideal, not a claim about actual
emulated TES invocations or vertex-cache hit rate. A device comparison must
report native/emulated GPU intervals and submit/wait times separately from
compile times, with identical shaders, levels, patches, fresh caches and driver.
Pipeline-statistics queries deliberately fall back and cannot measure native
invocations. Host recorder timings establish no GPU performance result.

The authored probes and draft device matrix live in `tests/native-tess/`.
Host recordings establish packet construction and dependency fields only.
Device acceptance requires actual pixel comparisons, completion/fault checks,
and paired timings, including domains, orders, three spacings, level extremes,
all patch sizes, empty patches, fallback state, and producer/consumer transitions.

Native TES applies domain winding and origin in `poly_load_tes_index`; it
therefore bypasses the hardware-VS TES rasterizer winding adjustment. Applying
both would reverse `ccw`/lower-left faces twice. Culling controls cross domain,
TES order, origin, API front face, front/back culling and low/high levels. Host
packet checks verify the final native raster bits independently of the index
swap; a CPU Vulkan reference validates the probe's pixel oracle.
