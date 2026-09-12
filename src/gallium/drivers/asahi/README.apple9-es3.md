# T8132 OpenGL ES 3.0 development coverage

This checkpoint implements the ES 3.0 graphics paths on the M4 Mac mini
(T8132, Apple9), with transform feedback excluded. It is development support,
not a Khronos conformance claim. The driver does not advertise complete ES 3.0
support while transform feedback is absent; the comparison runs explicitly
override the GLES and GLSL versions to 3.0 and 300.

The subsequent desktop trial raises the native ESSL capability to 300 and
uses an explicit GLES 3.0 override in application/session launchers. It also
adds refreshed tiled staging for linear 2D color sampler views. Direct
rendering into imported linear attachments remains a separate unsupported
path. Desktop validation and remaining browser/compositor issues are recorded
in the parent workspace's
`linux-m4-integration/tools/gpu/asahi/plasma-es3/README.md`.

## Implemented paths

Graphics shaders use ordinary NIR lowering, semantic Apple9 IR, register
allocation, and instruction encoding. There are no per-test shader replacements
or captured program replays. The implementation covers:

- Integer and floating-point vertex formats, vertex arrays, indexed and
  instanced draws, attribute divisors, and primitive restart.
- Structured shader branches and loops, integer operations, uniform buffers,
  dynamic indexing, and program binaries.
- Smooth, flat, and centroid varyings. Multisample centroid interpolation
  selects a position from coverage and evaluates both the varying and its
  perspective denominator there. Position temporaries use normal allocation.
- 2D, 3D, array, cube, and shadow texture operations; explicit LOD, gradients,
  queries, fetches, and offsets. Sixteen API samplers leave room for one private
  physical sampler used by fetch lowering. Offset operations that lack a
  validated direct encoding use generalized NIR lowering.
- ETC2/EAC uploads through decompression into ordinary supported storage,
  sRGB and packed color formats, integer targets, and multiple render targets.
- Color masks, standard blending, depth/stencil operations, fragment depth,
  discard, multisample rendering, and resolves. Advanced blend equations also
  use Mesa's common NIR blend implementation through the driver blend key.

Multisample draws use the ordered tile-access mode because explicit per-sample
color stores must preserve uncovered samples. This also preserves mixed stencil
coverage across an intervening render and a later stencil-tested draw. It may
cost performance compared with a future validated opaque multisample path.

The tile layout accepts up to 128 bytes per pixel for single-sample rendering
and 64 for multisample rendering. Eight RGBA32F targets are independently
verified at one sample. Wider multisample layouts remain unsupported; small
readouts were insufficient to validate them across tile boundaries.

## Validation method

The comparison set contains 43,792 GLES3 functional cases that actually passed
on LLVM 22 llvmpipe, with transform-feedback cases removed. Tests run on the
native Linux T8132 driver in batches of 64, using a 256-by-256 pbuffer and
`rgba8888d24s8ms0`; individual tests create multisample framebuffers as needed.
Separate runs exercise multisample default-framebuffer varying linkage.

The harness sets `EGL_PLATFORM=surfaceless`, `ASAHI_MESA_DEBUG=nocompress`,
`AGX_APPLE9_DIRECT_RENDER=1`, `MESA_GLES_VERSION_OVERRIDE=3.0`, and
`MESA_GLSL_VERSION_OVERRIDE=300`, and loads the development Mesa library.
Shader caching is enabled. A batch timeout or device submission error stops
the sweep. QPA files, case lists, logs, and parsed results are retained rather
than replacing failures with later isolated passes.

All 265 compiler tests and six launcher tests pass at this checkpoint.
An additional 858-case framebuffer, blending, depth, and multisample sweep
completed with 816 passes, 38 unsupported optional sample-count cases, four
compatibility warnings, and no failures. Independent native probes verify
16 filtered samplers plus a fetch, and centroid interpolation under both
varying pressure and unequal clip-space W.

The final full comparison (run 294) completed all 43,792 distinct reference
cases: 43,691 passes, one failure, 95 unsupported results, four compatibility
warnings, and one quality warning. No cases crashed or timed out, and the
runner reported no submission errors. Both texture failures from the previous
full sweep passed in this run, as did all ten advanced-blend and both stencil
failures fixed since that checkpoint.

The remaining failure was
`dEQP-GLES3.functional.fragment_out.array.fixed.rgba8_lowp_vec4`: blue and alpha
were zero on its first two attachments. Six subsequent runs of its exact
64-case batch passed all 384 case executions. This remains an unresolved
intermittent result; those reruns do not replace the failure in the full-run
totals. The warning results concern multisample wide lines and mixed-sample
framebuffer quality.

Evidence lives in the parent workspace under `tmp/es3-237/llvmpipe/`
(reference), `tmp/es3-288/` (advanced blend and samplers), `tmp/es3-291/`
(centroid probes), `tmp/es3-292/` (multisample linkage), `tmp/es3-293/`
(focused regression and unit tests), and `tmp/es3-294/` (full comparison).
Native per-batch artifacts are under `/g16/results/es3-204/full294/`.

## Remaining scope

Transform feedback is intentionally excluded. Optional sample counts above
four, the tested optional sRGB R8/RG8 extension formats, and multiview are not
implemented by this work. Coverage of a functional comparison set does not
establish API conformance, every extension, or performance readiness.

Hardware behavior is established on T8132 from independently authored shaders
and execution tests. T8140 behavioral documents are not treated as proof of
identical hardware behavior, and proprietary Apple binaries are not
disassembled or decompiled.


### Native linear render-target follow-up

Single-sample linear color targets now use byte row strides in the generated
GPU background/export shaders, including mixed linear/tiled MRTs and nonzero
dma-buf offsets. The tiled render-target/export-copy workaround was removed.
The native regression probe is
`linux-m4-integration/tools/gpu/asahi/plasma-es3/linear-render.c`; see that
directory's README for the validation matrix and remaining mapping-lifetime
limitation. Linear sampling still uses staging; this change does not claim a
validated fixed-function PBE linear export encoding.
