# Experimental Apple9 Honeykrisp

This is the experimental native Vulkan port for the T8132 M4 Mac mini
(G16G B1). It is not a conformant Vulkan implementation. Device enumeration
requires `HK_EXPERIMENTAL_APPLE9=1`, and the conformance version is zero.
Older chips retain their existing behavior.

## Status

The device reports Vulkan 1.4 on one graphics/compute/transfer queue with
sparse binding. Everything it advertises is listed explicitly in
`hk_apple9_restrict_caps()`, so a new Apple8 capability never appears on Apple9
by accident. Beyond the core 1.1-1.4 extensions, it exposes geometry and
tessellation shaders (emulated), transform feedback, graphics pipeline libraries,
shader objects, extended dynamic state 1-3, vertex input dynamic state,
sample locations, advanced blending, depth bias control and depth clamp
control, host image copy, pipeline binaries, sparse buffer and 2D image
residency, image view minimum LOD, descriptor buffers, custom border colors,
conditional rendering, primitives-generated queries, fragment shader
barycentrics, multisampled storage images, memory budget and priority, and
mesh and task shaders (run on the compute path), and sample fragment shader
interlock. Pixel interlock is off by default and available through the
driconf `hk_enable_pixel_interlock`: it sets the fragment control "Pixel
order" bit under MSAA (the acquire alone orders per covered sample, which at
1x is the same thing), and the bit orders whole submitted microtiles (32x16 at
4x, 32x32 at 1x/2x), not pixels, so MSAA pixel interlock with small
primitives can be several times slower. Measured 4x on/off for contiguous
grids of N-pixel primitives: 1 px 11.2x, 2 px 7.9x, 4 px 6.9x, 8 px 3.4x,
16 px 1.45x, 32 px 1.06x; large overlapping primitives cost nothing
measurable. Sampler min/max reduction is
not supported by the hardware either; only the vkd3d driconf fakes it for
feature level 12_0. Pageable memory, unified image layouts and nested
command buffers are not advertised.

Graphics state which Apple9 compiles into shaders (vertex formats, attachment
formats and blending, multisampling) is part of a draw-time key. Pipelines whose
state is static compile when created. Libraries, shader objects, partial stage
sets and keyed dynamic state defer compilation to draw time. A per-command-buffer
cache skips key computation while none of its inputs change. Tile load and store
programs are cached per device by their tile-buffer key.

Behaviour that differs from Apple8 Honeykrisp:

- Images are compressed as on Apple8, except RGB9E5, which Apple9's raw
  transfer path keeps uncompressed.
- FP32 arithmetic flushes denormals, but minimum, maximum and sign operations
  preserve them. `shaderDenormFlushToZeroFloat32` is therefore not advertised.
- The rasterized interface carries at most 96 user scalars, which bounds the
  vertex, tessellation evaluation and geometry output limits and the fragment
  input limit.
- Native triangle fans flat shade consistently only from their first vertex.
  Fans drawn with the last provoking vertex are unrolled to triangle lists.
- Cull distances are applied by the fragment program. Draws without a fragment
  shader link an empty one to the last pre-rasterization stage.
- With `dynamicRenderingUnusedAttachments`, a render's color tile layout can
  differ from the one the pipeline formats imply. Such fragment shaders take
  their layout from the render at draw time and discard writes to attachments
  the pipeline leaves undefined.
- Each sample holds up to 128 bytes of color at every sample count. Metal keeps
  the default 32x16 four-sample micro-tile even for 128-byte samples; the
  kernel encodes that 256 KiB tile allocation with a doubled bank tag.
- FP32 atomic addition is native in device memory, which image atomics also
  use; shared memory supports exchange but not addition.
- Custom border colors are sampled natively in the format the application
  gives; `customBorderColorWithoutFormat` is not advertised.
- 2D non-shadow samples without texel offsets take a minimum LOD natively, as
  signed Q6 in bits 16-27 of the word that carries the array layer: gradient
  samples directly, implicit and biased samples through the biased form.
  Other samples with a minimum LOD (`minLod`, `shaderResourceMinLod`) go
  through gradients of the projected coordinates, scaled by the instruction's
  bias, or select an explicit LOD.
- Hardware bounds shader-exported depth only to [0, 1]. Depth exports are
  clamped in the fragment shader to the draw's depth-clamp range (viewport 0's
  range, or the user range of `VK_EXT_depth_clamp_control`), as Metal clamps
  them against its near/far values.
- A NEVER depth test skips the fragment shader, and a NaN depth export passes
  ALWAYS. Fragment shaders with side effects therefore run against a LESS test
  and export a depth of 2.0, which fails against any stored depth while still
  applying the stencil depth-fail operation. With early fragment tests, NEVER
  stays in hardware.
- An early depth/stencil test updates the attachment before the fragment shader
  reads it as an input attachment. Pipelines with a depth/stencil feedback loop
  (static or dynamic) export the interpolated depth at the end of the shader,
  so the test follows the reads.
- Sampling has no residency mode. Unbound sparse pages map the scratch page
  (`residencyNonResidentStrict` is false), and each sparse-resident image has
  a table of bound pages, with a header locating each level's tiles. Sparse
  operations look up the page of the nearest texel at the nearest level; the
  view descriptor records the table and the view's base level and layer.

- Task and mesh shaders (`VK_EXT_mesh_shader`) run as compute kernels ahead
  of the render. Each mesh workgroup writes a record of its vertices,
  primitives and counts; a generated vertex shader then draws every
  primitive slot of every record, placing unused or culled slots behind the
  eye. Task workgroups write their payloads and launched grids, and a prefix
  sum maps the linear mesh dispatch back to task workgroups. Fragment shaders
  read user per-primitive varyings from the records, by a flat varying
  holding the primitive's offset, so they do not use the 96-scalar interface.
  Records of draws sized on the GPU come from a 512 MiB heap, created on first
  use and freed after each command buffer; draws which do not fit are
  dropped.

Known gaps:

- Multisampled early fragment tests triggered from the shader (with side
  effects, discard or depth/stencil writes) test interpolated depth bounded
  only to [0, 1], not to a narrower depth-clamp range. Single-sample shaders
  replace the initial coverage test with a clamped depth publication, before
  side effects and discard. Replacing that test avoids a second depth/stencil
  update; `dEQP-VK.depth.ez_fs_clamp` and the early-fragment depth/stencil tests
  pass. Extending this to MSAA requires depth interpolated at each sample.
- Exported depth is clamped to viewport 0's range for every viewport.
- Sampler reduction modes are not implemented. Sparse residency covers 2D
  and 2D array images, as on Apple8.
- Mesh shaders run in software (below), not on the native mesh pipeline.

The sections below record the port's development in order; statements in
earlier sections may be superseded by later ones and by the status above.

## Implementation

Honeykrisp retains its normal SPIR-V frontend, NIR descriptor lowering, Vulkan
objects, command-buffer recording, and DRM submission. Apple9 lowers the root
and descriptor-set address intrinsics to reads from one internal UBO containing
Honeykrisp's root table. Descriptor contents remain GPU memory and are not
specialized into shader binaries. Push constants and storage-buffer addresses
use the existing root/descriptor layouts.

Compute NIR uses the same Apple9 instruction selection, register allocation,
and binary encoder as the GL driver. The compiler's compute ABI profile is
serialized with the shader metadata. Honeykrisp uploads the complete Apple9
main/preamble binary without the Apple8 shader linker and records direct or
indirect CDM dispatches through the common Apple9 launch helpers.

The existing Apple9 launch, geometry and hardware-state helpers have moved
from the Gallium driver to `src/asahi/lib`. A small allocation callback lets
Vulkan own its immutable launch records in the command buffer's USC pool;
Gallium retains its existing pool wrapper. This avoids a second copy of the
hardware encoding. Executable entry records remain tied to their shader BOs.

Honeykrisp command buffers now have a separate recyclable render-context pool
for native PPP/VDM records, in addition to ordinary and USC allocations. The
native attachment load/store NIR builder, native PBE encoder and stage launch
preparation are shared with Gallium. Render-pass color clears and scissored draws
pass guarded GPU readback. Direct and indexed basic-draw CTS selections pass
24 and 33 cases respectively, with 16 skips each. These are focused development
checks, not the Vulkan 1.0 conformance matrix.

Depth/stencil `DONT_CARE` store operations are honoured by the last control
stream of a render pass; earlier streams of a split pass still store for the
next one to load, and a suspending pass keeps its stores. The firmware keeps
depth and stencil across tiler partial renders independently of these stores:
a probe drawing up to a million depth-tested full-screen triangles past the
tiler heap limit (several refused growths) stays exact with the stores dropped.
Not storing depth removes about 0.36 ms per 3840x2160 frame; vkmark's model
scenes there run 65-77% faster.

## Current boundaries

Validated: storage-buffer and uniform-buffer reads, storage-buffer writes,
descriptor arrays and multiple/discontiguous descriptor sets, uniform texel-buffer
reads and size queries, dynamic offsets,
push constants, direct/indirect dispatch, zero-work dispatch, workgroup geometry,
shared-memory exchange, indexed private arrays, returned buffer atomics, shader
disk-cache reuse, buffer fill/update/copy, binary semaphores and fences. Typed
storage-image loads/stores, image atomics, and image/buffer copies use GPU
descriptor address calculations and format conversion through global memory.
Tiled and linear image addressing share the existing libagx helpers. Software
image accesses check bounds before touching memory; no image data is read on
the CPU to implement shader access. Internal transfers copy integer texel bits
to preserve shared-exponent representations and floating-point NaN payloads.

Internal precompiled kernels use the same Apple9 launcher as API compute
shaders. Their argument block occupies UBO 0; shared memory uses the compiler's
shared root, and scratch/atomic frames come from the dispatch profile. Buffer
transfers therefore execute on the GPU through the normal libagx kernels.
Adjacent NIR memory/execution barriers are combined before Apple9 instruction
selection. Empty compute entrypoints retain Asahi's empty-shader marker.

Sparse residency remains disabled. Memory allocation omits its unused read-only
shadow mapping when the physical device lacks sparse buffer residency. This
avoids rebinding the same shadow when a cached BO is reused by a later
allocation, which the current kernel rejects with EEXIST.

Native vertex/fragment pipeline compilation is available for core monolithic
pipelines. Static vertex formats and attachment/blend state are cache keys;
vertex output layouts drive fragment interpolation. A compile-only probe passes
24 cases covering attributes, procedural vertices, push constants, two color
formats, depth-only passes, blending and pipeline-cache reload. Compiler tests
also cover zero color targets. These results do not validate rendered pixels.
Native draw recording bypasses Apple8 shader linking and emits native VDM/PPP
state. Vertex-buffer bindings, per-stage descriptors, push constants, blend
constants, viewport, scissor and raster state use ordinary Vulkan state.
Vertex fetches use a GPU-readable address/size table and 64-bit byte offsets.
Robust vertex input defines out-of-bounds per attribute: an attribute with any
byte past the binding's range reads the zero sink page at offset zero instead,
so its components load unconditionally and vectorize; missing format
components keep their specified zero/one defaults. The earlier branch per
component made up about a fifth of vertex-shader code: on a 300-case selection
vertex shaders shrink by 53%, their device loads by 61%, and nearly all of
their execution-mask regions disappear. Binding offsets participate in effective
address alignment. Packed 11/11/10-bit floats convert through half-float NIR.
Index buffers reserve a page-aligned USC mapping because native index pointers
are relative to that aperture. Command-stream continuations use render-context
offsets. Indirect draws use GPU-resident arguments. Pipelines without a fragment
shader use a native empty fragment shader, and zero-color-target linking supports
depth-only rendering. Its runtime sample mask also controls depth writes and
occlusion when no API fragment shader is bound. Native multisampling exposes 1x, 2x and 4x. Two-sample renders use 32x32
tiles, four-sample renders 32x16.
Graphics remains incomplete; broader attachment, robustness, query, and
presentation need broader validation and further work. Native shared integer
atomics pass the basic compute and GLSL atomic-operation CTS selections. Nonuniform
descriptors, full robustness coverage, other memory scopes, queries, and broader
API completeness need further work. Shared-memory allocation supports 16 KiB.
Full-range neighbor exchange at 2, 4, 8, and 16 KiB passed across 1025 workgroups
with direct and indirect dispatches and changing inputs.
Passing the selected cases below does not establish Vulkan conformance.

## CTS development profile

`src/asahi/ci/deqp-asahi-m4-vk.toml` runs Vulkan CTS through the existing
`src/asahi/ci/m4/run.py vk` launcher. The configuration maps `/deqp-vk` to a
native CTS runtime and supplies the unchanged, flattened `vk-main.txt` list.
The candidate prefix must have `lib/libvulkan_asahi.so` and exactly one Asahi
ICD manifest under `share/vulkan/icd.d`, naming that library by absolute path.
Set `HK_EXPERIMENTAL_APPLE9=1` and `ASAHI_MESA_DEBUG=nocompress` in the launcher's
configured environment. Use `--include-tests` while functionality is incomplete.
The launcher verifies the Apple M4 renderer, hashes the driver and inputs, and
retains upstream skips, flakes, single-thread restrictions and expected failures.

On September 27, 2026, the selected foundation matrix covered basic compute,
compute UBO/SSBO descriptor access, indirect dispatch offsets, basic binary
semaphores/fences, buffer fill/update and buffer-to-buffer copy. It produced
363 passes, 130 skips and five shared-atomic failures after image support, with
no crashes, timeouts, expected failures or flakes. Raw outcomes remain failures;
no local expected-failure entries were added. A separate image store/load-store
selection produced 897 passes and 1833 skips, and image atomic/size/consistency
selection produced 396 passes and 5924 skips, both with no failures. These skip
counts include unsupported optional features and modern CTS load/store cases
that unconditionally request SPIR-V 1.3 despite the API 1.0 interface. They are
coverage gaps, not evidence of passing those operations. Buffer robustness
scalar/vector/matrix selection produced 81 passes and 48 skips without failure.
After the narrow-shift and raw-copy fixes, the combined transfer/basic-compute/
storage-image selection produced 11949 raw passes, 50737 raw skips, and five
shared-atomic failures, with no crashes or timeouts. Transfer cases alone had
10610 raw passes and 42951 skips, without failures. The runner classified 244 of
those passes and 344 skips as improvements against unchanged upstream failure
expectations; no local expectation edits were made.
The compiler unit suite passed all 369 tests.

Uniform texel-buffer reads use a 56-byte descriptor: the ordinary texture-query
prefix plus a GPU-readable base address, texel count, byte size, and four channel
conversion words. This supports descriptor arrays and formats selected at runtime
without recompiling the shader. Bounds checks precede loads, and byte/halfword
loads avoid reading beyond a short view. Other chips retain their original layout.
The authored Vulkan 1.0 probe passes all 30 formats, including signed/unsigned,
normalized, sRGB, half/single float, RGB32 and packed float representations.
It re-records commands after descriptor updates, as core Vulkan requires.
The separate CTS texel/compute-robustness selection has 171 passes and 75 skips;
compute integer builtins have 121 passes and 30 skips. Neither has failures.
The expanded foundation has 393 passes, 130 skips and the same five shared-atomic
failures. These suites overlap and must not be summed as distinct coverage.

Sampled images now use native Apple9 texture and sampler tables. At compilation,
static descriptor set/byte offsets become logical slots; at command recording,
the current descriptors are translated into compact native tables. Sampler and
image packing is shared with Gallium, and descriptor set layouts stay unchanged.
The command uploads metadata for native texture lowering through its private UBO.
API image queries retain GPU reads of the ordinary descriptor prefix, including
storage texel-buffer queries. Metadata queries do not retain texture-cache fences
in shaders whose image accesses have all become global memory operations.
Sampled-image and sampler limits are sixteen per stage; optional dynamic sampled
array indexing stays disabled. This is valid for core descriptor updates, which
invalidate commands using the changed binding and require re-recording.

Compute sampled binding coverage has 516 passes and 2892 skips; sampler
indexing has 46 passes and 46 skips. The first texture-function run had 286
passes, 634 skips and 455 filtered-offset failures. Runtime sampler metadata
now drives the same per-level texel offset, wrapping, filtering, comparison and
mip blending code used by Gallium's static keys. All 455 failures became passes:
the full compute texture-function selection has 741 passes and 634 skips.
The existing compiler test exercises both static and runtime sampler state,
including narrow LOD inputs; all 369 compiler tests pass.

Unnormalized 1D/2D sampling converts pixel coordinates using per-view dimensions
before native sampling. Exact-sampling and sRGB-read coverage has 237 passes
and 267 skips without failures. The shared native descriptor translator now
preserves the two-channel sRGB decode flag. Apple9 only advertises typed storage
formats in the SPIR-V/Vulkan compatibility table: untyped storage features are
not implemented. Unsupported sRGB/other untyped storage combinations remain
skips, rather than evidence of working storage-image encoding.

The expanded sampling/binding selection has 1915 passes and 4373 skips without
failures. The final combined transfer/image/compute/descriptor/synchronization
baseline has 12296 raw passes, 50838 raw skips and five shared-atomic failures,
with every raw outcome unchanged from the pre-offset baseline. Native GL 3.3,
GLES 3.0 and the 30-format texel-buffer probe also pass with this candidate.
Each expanded run retains raw outcomes and unchanged upstream expectations.
These selections overlap and do not establish Vulkan conformance.

Depth coverage now has 1355 passes and 7166 skips; selected RGBA8 blending and
S8/D32S8 stencil coverage has 8460 passes without failures. The vertex-offset
selection has 112 passes and 116 skips. These selections overlap other runs.
Authored robust vertex probes check both a four-byte buffer with adjacent
allocation sentinels and a 4 GiB fetch offset that must not wrap to zero. Both
pass three submissions with complete pixel and output-guard checks.

The multisample/render-target selection has 726 passes and 11569 skips without
failures. Standard sample positions are set on every native render submission.
Alpha-to-coverage precedes all color stores and remains active when color
attachment 0 is absent or unused. Multisample input attachments accept an
explicit zero LOD. A separate authored probe checks that a zero shader sample
mask suppresses single-sample Vulkan rendering, while compiler tests retain
OpenGL's different single-sample behavior.

Core occlusion/query-copy and empty-fragment coverage has 229 passes and 120
skips without failure. Query copies use normal native compute kernels, including
narrow boolean-to-integer conversion. Newer CTS `get_reset_results` variants
that call host query reset without first checking `VK_EXT_host_query_reset`
remain recorded separately as crashes; they are not part of that passing core
selection. No expected-failure files were changed.

Packed RGBA4 render formats use the shared FP16 tile representation. RGB9E5
render-target/blit-destination support remains disabled as in native GL; its
sampling, storage, and transfer paths remain available. A broad blend/stencil/
interface/constant selection produced 19872 passes, 54210 skips and one
fragment-storage compilation failure. After adding the missing printf, depth
state and helper-invocation lowering, the focused constant/empty-fragment/
discard selection has 38 passes and 67 skips with no failures. Optional fragment
storage features remain disabled pending broader validation. The compiler unit
suite now has 375 passing tests.

The combined graphics regression covers vertex input, assembly, depth, blend,
stencil, shader interfaces, constants, multisampling, framebuffer attachments,
interpolation, draw state and core occlusion copies: 26880 passes and 84043
skips with no failures. The compute/transfer baseline remains exactly unchanged
at 12231 raw passes, 50903 raw skips and five shared-atomic failures.

Attachmentless draws now infer their sample count from the pipeline and update
both the native tile layout and submission sample positions. The authored
`apple9-attachmentless.c` probe checks zero/nonzero conservative occlusion with
zero and single-bit API sample masks, including only sample 3 enabled at 4x.
It reproduced a zero result before the fix and passes all four cases afterward.

Native device loads now support volatile accesses, with device fences and
preservation through native optimization/scheduling. This enables the ordinary
timestamp-copy meta kernel. Timestamp coverage has 120 passes and 158 skips;
the combined timestamp/info/empty-fragment selection has 144 passes and 165
skips without failures. Point/line limits match the disabled optional features,
and experimental instance extension reporting omits the Vulkan 1.1-only
protected-surface capability extension. Core device/instance information checks
pass. These are development results, not a complete Vulkan 1.0 validation run.

API format validation now rejects sparse image requests consistently with the
disabled sparse-binding feature. Typed storage access includes the packed ABGR
aliases of RGBA8. A guarded native storage-texel probe validates loads, stores,
negative/out-of-range accesses, nonzero view offsets and descriptor replacement
for all four ABGR UNORM/SNORM/UINT/SINT formats.

BC, ETC2 and ASTC LDR texture features are enabled after native compressed-texture
coverage: 816 passes and 996 skips across graphics/compute, 2D/3D and mip/size
variants. The accompanying API info/format/memory-requirement/smoke selection has
4458 passes and 6642 skips without failures or warnings. Block-compressed texture
formats are separate from the unsupported framebuffer-compression path; native
validation continues to use `ASAHI_MESA_DEBUG=nocompress`.

Shared 32-bit integer atomics now lower from ordinary NIR through semantic native
IR and allocation. This includes add/subtract, bitwise operations, signed and
unsigned min/max, exchange, and compare-exchange. Atomic inputs use allocated
private copies; compare-exchange collects adjacent desired/compare registers.
The shared-memory completion protocol publishes through slot 1 and retires
results before dependent shared accesses. Unused atomic returns reuse the consumed
data register rather than requiring a fixed scratch register.

The five basic shared-atomic/barrier CTS cases pass. The complete GLSL atomic
selection has 32 passes and 1008 skips, with no failures. The authored
`apple9-shared-atomic.comp` probe checks returned values, final shared values,
and successful/unsuccessful compare-exchange over three changing direct/indirect
submissions, each with 65537 outputs plus 63 guard words. It has zero mismatches.
Compiler tests have 377 passes; native GL 3.3 and GLES 3.0 smoke tests pass.
These results do not establish Vulkan 1.0 conformance.

The shared-atomic compute/transfer regression has 12236 raw passes and 50903 raw
skips. Its only changes are the five former shared-atomic failures becoming passes.
Descriptor copy/update, dynamic-offset, combined-stage and invalid-unused-descriptor
coverage adds 268 passes and 176 skips without failures.

Render-pass testing exposed two raster correctness bugs. Omitted or fully masked
color targets now request ordered tile access so opaque visibility cannot discard
prior fragments needed by those targets. Depth publication now preserves shader
sample demotions instead of reconstructing killed samples from entry coverage.
The isolated partial-clear preservation and 4x D16 depth-export cases pass after
the fixes. Compiler regression coverage now has 379 passing tests.

The experimental Vulkan interface reports four color/output attachments. This is
the limit supported for every advertised format at 4x by the current 64-byte tile
layout; eight RGBA32 targets need tile spilling or a smaller tile layout. Input
attachments are limited to the implemented sixteen texture slots. These limits
replace inherited Apple8 limits that exceeded the implemented Apple9 path.

After the preservation, depth, and limit corrections, the renderpass1 plus API
information selection has 8942 passes and 21292 skips. The subsequent
compute/transfer baseline is unchanged (12236 raw passes, 50903 raw skips), and
pipeline/draw graphics regression has 26656 passes and 83926 skips. No failures,
crashes or flakes occurred in those final selections; the earlier failures remain
in the checkpoint's raw logs.

Coarse derivative intrinsics now select the same native differencing instruction
as fine derivatives, matching upstream AGX lowering. The lowered back-face
predicate is accepted alongside the public front-face intrinsic. Their complete
focused CTS selection has 1177 passes and 522 skips.

Array LOD queries no longer read a nonexistent layer coordinate during sampling
lowering. The native query gets a zero layer; shadow comparison does not apply.
Existing query bias is retained as a biased operation instead of adding a second
bias operand. The LOD-query selection has 165 passes and 25 skips with no crashes
or failures. Compiler tests have 382 passes. Full GLSL and texture selections are
still being expanded; these focused results are not full API conformance.
The earlier full texture selection had four uniform sRGB texel-buffer failures.
A diagnostic isolated differing hardware image and software texel conversion.
Uniform sRGB texel buffers now use a bounds-checked native 2D fetch through the
same hardware decoder as images. The texel-buffer selection has 18 passes and
five skips; the authored 30-format, bounds, and descriptor-replacement probe
also passes. The descriptor height includes any initial texel offset.

Dynamic indexing exposed three independent issues. Perspective interpolation's
reciprocal cache is now scoped to its defining block, so sibling branches do not
use a non-dominating SSA definition. Native graphics scratch uses launch-frame
extents and no longer requests the Apple8 scratch helper at submission. Dynamic
fragment input dereferences are expanded before IO lowering to fixed interpolation
slots. The complete GLSL indexing selection passes all 760 cases without crashes.

Apple9 math lowering now runs before common preprocessing can replace sine and
cosine with the older fsin_agx operation. The focused trigonometric selection has
48 passes and 12 skips. Vertex shaders with unwritten position components receive
zero defaults for those undefined components, allowing side-effect-only shaders
to compile. Compiler regression coverage has 384 passing tests; native GL 3.3
and GLES 3.0 clear/draw/texture/framebuffer smoke probes also pass. The complete
GLSL group after the indexing, math, and position fixes has 15175 passes and
13221 skips, with no failures, crashes, or flakes.

The full texture rerun now has 5935 passes and 19734 skips. The compute/transfer
baseline remains 12236 raw passes and 50903 raw skips, with no result changes.
SPIR-V quantization exposed unsupported explicit f2f16_rtne operations. They now
select the native round-to-nearest conversion; all 55 supported quantization
cases pass, with 120 skips. Compiler regression coverage has 385 passes.
Broader SPIR-V coverage is still in progress.

Next: remaining graphics and multisample coverage,
presentation, and broader API/CTS coverage, including remaining texture ops.
Each stage must preserve the compute/transfer baseline.

`libagx_imm_write` holds a 64-bit address and a 32-bit value. i686 aligns
`uint64_t` to 4 bytes, so 32-bit builds wrote 12-byte records that the GPU
read at 16 bytes, sending batched query writes to wrong addresses. That
corrupted memory and hung 32-bit DXVK Heaven. Explicit padding and a size
assertion keep the stride at 16 bytes on every host.

## Focused test

Events share stable, individually addressable slots in 16 KiB backing pages.
The previous one-BO-per-event allocation made the 10,000-event large-primary
command-buffer test exhaust mapping resources. The device locks allocation and
slot reuse; empty pages are released after their last event is destroyed.
`tests/apple9-events.c` checks 10,000 independent statuses, 128 GPU writes
including writes across a page boundary, and 5,000 destroyed/recreated slots.
It passes on native T8132, as do the large-primary CTS case, nine event object
tests, and five event synchronization tests (one and five additional skips,
respectively). These focused results do not establish Vulkan conformance.

The native test kernel also avoids copying the entire mapping ownership vector
on every unmap. It reserves a single extra record only when an interval splits
one mapping. This removes an allocation failure from whole-mapping teardown;
other page-table/invalidation allocations can still fail. The original large
event test now passes without the previous GEM-close ENOMEM/device failure.

Build Mesa with `-Dvulkan-drivers=asahi`. Keep the candidate outside the system
Mesa installation and select its ICD explicitly. Use an ICD manifest whose
`ICD.library_path` is the absolute path to the candidate `libvulkan_asahi.so`.

From the source root, with a GLSL-to-SPIR-V compiler and Vulkan loader installed:

```sh
glslangValidator -V src/asahi/vulkan/tests/apple9-compute.comp -o /tmp/apple9-compute.spv
cc -Iinclude src/asahi/vulkan/tests/apple9-compute.c -o /tmp/apple9-compute -l:libvulkan.so.1
VK_DRIVER_FILES=/absolute/path/to/candidate.json \
  HK_EXPERIMENTAL_APPLE9=1 ASAHI_MESA_DEBUG=nocompress \
  /tmp/apple9-compute /tmp/apple9-compute.spv
```

The test requires an Apple M4 renderer. It reads one storage buffer and writes
another using a push-constant salt and a bounds check. Five 64-thread groups
produce 257 output words and leave 63 guard words unchanged. Three submissions
change the input data and push constants; the middle submission uses indirect
dispatch. Each fence has a ten-second timeout. All outputs and guards must match.
Repeat with `MESA_SHADER_CACHE_DISABLE=false` and a dedicated
`MESA_SHADER_CACHE_DIR` to exercise cached shader deserialization.

On September 27, 2026, this test passed with both cold and warm caches on native
T8132. The shared launcher and geometry unit suites also passed, as did focused
native OpenGL 3.3 and GLES 3.0 clear, draw, texture-upload and framebuffer-readback
checks using the rebuilt GL driver. The normal desktop driver was not replaced.

Additional authored probes use the same executable:

- `apple9-compute.spv --transfers`: GPU fill, compute, GPU copy, then host readback,
  with command-buffer barriers and no intervening CPU waits.
- `apple9-shared.spv --shared`: each invocation reads its neighbor's shared value
  after a workgroup barrier.
- `apple9-shared-range.spv --shared`: fills and checks every word of an array,
  with independent values per workgroup and word. Compile with
  `-DSHARED_WORDS=4096` for 16 KiB. Build the C executable with
  `-DAPPLE9_PROBE_COUNT=65537` for the 1025-workgroup stress run.
- `apple9-private.spv`: a dynamically indexed 512-byte private array; compiler
  diagnostics confirm scratch loads and stores.
- `apple9-atomic.spv`: buffer atomic addition whose returned value is consumed.
- `apple9-atomic-spill.spv`: 128 loaded values stay live across a returning
  atomic, so the allocator spills around it. A checksum taken before and after
  the atomic cancels only when the returned value and every spilled value
  survive. It failed every output before result records named their scheduled
  slot and spills moved above the frame word the atomic writes (see the
  compiler README).

Compile the corresponding `.comp` files with `glslangValidator -V`. All five
probes passed the same three-submission output and guard checks on native T8132.

Build `tests/apple9-texel.c` against the Vulkan loader, then compile
`tests/apple9-texel.comp` three times with `glslangValidator -V`, defining
`TEXEL_FLOAT=1`, `TEXEL_UINT=1`, or `TEXEL_SINT=1`. Pass those three binaries in
that order to the probe. The test checks seven alternating texels per view,
negative/out-of-range reads, size queries, and output guards, while reusing three
pipelines across descriptor formats.

`tests/apple9-depth-partial.c` (with the `.vert`/`.frag` shaders of the same
name) draws `NTRIS` instanced full-screen triangles into a 1024x1024 target whose
depth is cleared and not stored: the first tenth near and green, the rest far and
red, with a LESS test. Every pixel must stay green. From about 200,000 triangles
the tiler heap reaches its limit and the firmware renders in parts, so depth must
survive partial renders; `ZSTORE=1` stores depth for comparison. On October 1,
2026 it passed on native T8132 at 1,000, 300,000 and 1,000,000 triangles in both
modes.

For vertex bounds, compile `tests/apple9-vertex-bounds.vert` and `.frag` with
`glslangValidator -V`, build `tests/apple9-render-draw.c` against the Vulkan
loader, and pass the vertex and fragment SPIR-V paths followed by `bounds` or
`bounds-wrap`. The first mode checks that a short buffer does not expose adjacent
allocation contents; the second checks that 64-bit fetch offsets stay out of
bounds instead of wrapping to the start of the buffer.
Use the ordinary draw vertex shader with `apple9-sample-mask.frag` and the
`sample-mask` argument to check single-sample shader sample-mask writes.

`tests/apple9-sample-rerender.c` (with `apple9-fullscreen.vert`,
`apple9-pattern.frag` and `apple9-sample.frag`) renders a hashed pattern, samples
it, renders a new pattern into the same image and samples again, comparing every
pixel. `WIDTH`, `HEIGHT`, `ROUNDS`, `FORMAT`, `NTRIS`, `PARTIAL`, `LOADOP`, `MRT`,
`DEPTH` and `ATT` select partial renders, load continuation, several attachments
and depth. `tests/apple9-bindless-sample.c` (with `apple9-bindless.vert`,
`apple9-bindless.frag` and `apple9-combined.frag`) samples through DXVK's layout:
a 2048-entry update-after-bind sampler array indexed by a push constant and a
separate sampled image, so both indices reach the texture unit from registers.
`DRAWS`, `SAMPLER`, `SAMPLERS`, `TEXSETS`, `IMAGES`, `NOISE`, `POOLS`, `HOSTWRITE` and
`NOISEIMM` vary the descriptors, pools, concurrent host writes and interleaved
per-draw-table draws. `IMAGES` creates distinct hashed source images, cycled by
descriptor set. With `IMAGES=16 DRAWS=256 TEXSETS=256 POOLS=64 NOISE=16`, this
reproduces the intermittent per-wave sampling corruption from the Unigine Heaven
replay. The wrong pixels often come from the correct image at `(x, 0)` or
`(0, 0)`, revealing lost coordinate components rather than another image.
The register-indexed texture encoding now preserves the coordinate-publication
wait in bits 39-41, just as the immediate-table form does. The packer regression
covers implicit LOD, explicit LOD and gradients, texture-only and combined
register indices, and all six result completion tags. Run native checks with
`HK_EXPERIMENTAL_APPLE9=1 ASAHI_MESA_DEBUG=nocompress` and the intended
`VK_DRIVER_FILES`; the test prints the renderer and compares every pixel.

`tests/apple9-map-placed.c` maps allocations placed (`VK_EXT_map_memory_placed`)
between guard pages, with and without `VK_MEMORY_UNMAP_RESERVE_BIT_EXT`,
including allocations that reuse a larger BO from the BO cache that was freed
while mapped. API maps now cover exactly the page-aligned allocation and are
owned by the memory object, separate from the driver's own BO mapping. Before,
a cached BO's leftover mapping was returned instead of the placed address, and
a larger cached BO's map and unmap covered the caller's pages after the
allocation. That caused the intermittent `dEQP-VK.memory.map_placed` failures
and crashes.

Uniform texel buffers now read every format except RGB32 through the texture
unit, with a bounds-checked native 2D fetch of the view's linear descriptor that
starts at the view's first texel. RGB32, which has no native format, loads its
three words. Reads no longer decode the descriptor's per-component layout in the
shader. As with image descriptors, the texel descriptor carries its native texture
entry at offset 64, so dynamically indexed and update-after-bind texel buffers
use register indices. Sampled images and uniform texel buffers need not observe
writes from the same dispatch, so they no longer keep memory-to-texture fences,
which Apple9 cannot encode. Only texture-unit storage image loads still do. With
256 fetches per compute thread, T8132 reads 16.8 billion texels per second
from RGBA8 and R32F views (4.5 before), 12.8 from RGBA16F and 16.6 from RGB32F.
All 30 formats of `tests/apple9-texel.c` pass. Its out-of-range check now accepts
an alpha of 0 or 1, as `robustBufferAccess` allows.

Pipeline statistics queries halved Unigine Heaven's frame rate through DXVK,
which keeps queries of all eleven counters open around some draws. Three
problems, measured on the native replay of a captured frame:

- Ending a statistics query left its counters active, so every later draw in
  the command buffer kept counting into the ended query. That also affected
  its results. Ending the query now disables the counters.
- Each fragment added one to the invocation counter with its own atomic, which
  serialized the draw's fragments on one address. One lane of each wave now
  adds the wave's count of non-helper invocations. The counting branches are
  marked not to flatten: if-conversion otherwise hoisted the coverage read
  and ballot into every invocation of every fragment shader. The counter no
  longer disables triangle merging unless the shader itself requires it.
- Every draw ran an input assembly compute dispatch behind a full barrier.
  Direct draws without primitive restart now add their vertex, primitive and
  clipper counts on the CPU. The sums are queued as counter increments when
  the query ends or the counters change. Indirect and restart draws keep the
  per-draw dispatch.

The three-frame Heaven replay went from 17.9 to 36.7 frames per second. That
matches disabling statistics queries entirely, which gave 36.4.

The fragment invocation counter counts the invocations that run. Hidden-surface
removal skips the fragments of opaque draws that later draws in the same pass
cover, so such draws can count fewer invocations than they cover, or none. The
Vulkan specification allows it. Under Fragment Shading (`fragops-shader`), "A
fragment shader may not be executed if: An implementation determines that
another fragment shader, invoked by a subsequent primitive in primitive order,
overwrites all results computed by the shader (including writes to storage
resources)." The counter's atomic is the driver's, not a storage write of the
application, so it doesn't keep the shader running. Draws whose fragments stay
visible count exactly: on T8132, disjoint draws inside one statistics query
matched 48 of 48 counts for both complete and library pipelines.

Ending an occlusion, statistics or primitives-generated query also queued its
availability write into a compute dispatch after the render pass. DXVK ends
such queries in every pass, so each render pass was followed by its own
compute command. On T8132 a render command costs about 150 us even when it
draws one triangle into a 64x64 target, and a compute command between two
passes adds about 110 us. Availability writes and counter increments now wait
in the command buffer until the queries can next be observed: a query reset,
a query copy, or the end of the command buffer. One dispatch then applies
them. The replay's frame went from 73 GPU commands and 17.4 ms of GPU time to
38 commands and 12.9 ms. All 19,236 `query_pool` cases pass.

Half of the remaining cost per render pass is its barrier. Every command
waits for all earlier commands; without those waits a render command costs
about 80 us instead of 160. The kernel's fragment-stage render barrier
(`DRM_ASAHI_RENDER_VDM_BARRIER_FRAGMENT`) lets tiling overlap the previous
pass, but each pass then cost 217 us, because the fragment queue's own wait
entry is slower than the full barrier. Neither a production kernel
configuration nor batching commands into one submission changed these costs,
so they belong to the firmware's handling of dependency entries.
