# T8132 graphics feature coverage

The Apple9 compiler and m1n1 transport are an experimental T8132 backend, not
an OpenGL conformance claim. The public graphics caps remain GLSL 1.10 / ESSL
1.00, eight varying vectors, and **eight render targets**. The test harness uses
an explicit development context override.

## Implemented paths

| Feature | Implementation and coverage |
| --- | --- |
| Partial fragment outputs | Collect masked/component writes into per-output temporaries, then lower the final values. Unwritten components have undefined API values; the packer selects zero after undef trimming. The original post-increment test passes without changing its shader. |
| Border sampling and legacy `GL_CLAMP` | Standard NIR texture lowering combines native border samples for custom colors; legacy clamp preserves its border interpolation. The four local fixtures check nearest/linear filtering, and the original five sampler failures now pass. |
| `gl_FrontFacing` | Read the hardware back-face input and invert it, following the Apple8 NIR convention. Both winding directions are covered. |
| Dynamic local arrays and vectors | Standard NIR lowering handles small indirect temporaries. Registers are allocated normally; this adds no replay sequences or fixed register assignments. Large live arrays still require the deferred spill/scratch implementation. |
| Derivatives | Generalized quad differences implement `dFdx`, `dFdy`, and `fwidth`, including their sign and axis conventions. |
| Cube sampling | Direction projection, face selection, and implicit/explicit/bias LOD use generated code and ordinary sampler/texture bindings. Fixtures cover faces, corners, and mip selection. |
| Shadow samplers | Native comparison state supports all eight compare functions, nearest comparison, and linear PCF. Coverage includes the 28 upstream GLSL 1.10 shadow tests and the two local comparison fixtures. |
| Dynamic varying arrays | Stage-interface temporaries lower indirect VS writes and FS reads into ordinary varying slots. Boolean values used across loop exits are materialized in their defining blocks. |
| 1D and 3D textures | 1D lowers through the ordinary 2D path. 3D descriptors and sampling carry depth, coordinates, and LOD explicitly; local tests distinguish volume slices and mip levels. |
| Rejection before submission | Shader compilation failure leaves no usable stage. Resource validation runs before recording a draw, rejecting unsupported or missing sampler/texture/buffer state rather than constructing partial descriptors. |

Spilling and scratch storage are deliberately deferred. Custom-border shadow
sampling, seamless cube filtering, texture arrays, and unimplemented texture
operation/dimension combinations are not claimed by these results. The tests
exercise specific paths; passing them does not establish every combination of
format, filtering, state, and shader control flow.

## Multiple render targets

The RGBA8 path supports one through eight color outputs, per-target blending
and color masks, legacy `gl_FragColor` broadcast, sparse `GL_NONE` slots, and
attachment remapping. Textured and discard shaders use the same target layout.

Three independent state errors caused the original black attachments:

- The export graph retained the single-target store mask and omitted later
  targets' byte offsets. It now describes every active attachment.
- Load, reload, fragment, and store launch setups retained an eight-byte tile
  sample stride. Their measured stride field now follows the framebuffer's
  padded sample size, including sparse slots.
- The generic G13 tile-size threshold shrank seven/eight-target passes to
  32×16 tiles. This T8132 path uses 32×32 tiles through eight RGBA8 outputs.

The compiler uses normal per-output tile addresses and NIR blend lowering.
Standard constant-loop unrolling avoids unnecessary live output arrays in
bounded loops. Already-lowered legacy color stores broadcast as well as
variable-based frontend outputs. MRT clears use ordinary clear draws so that
independent attachment masks cannot become an unintended whole-pass clear.

The setup fields were isolated through authored Metal workloads, state-byte
comparisons, and hardware tests. Compatibility helpers remain opaque. No
proprietary helper was disassembled or decompiled, and the production compiler
contains no capture-specific MRT shader, fixed-register replay, or per-count
native pipeline image.

`tests-graphics/mrt/` includes constant-output tests for every count from two
through eight at 16×16 and 67×73, plus an EGL API harness covering all eight
outputs across tile boundaries, broadcast, sampling, discard, sparse slots,
remapping, and indexed blend/color masks. The API harness is also checked
against llvmpipe.

## Running the graphics suite

Run the normal m1n1 build/reset/chainload procedure first, with display enabled
for this standalone graphics path. Run only one GPU client at a time.

From the Mesa checkout:

```sh
src/asahi/drm-shim/piglit/run-graphics.sh ../logs/apple9-graphics-new
```

The default `graphics-tests.txt` manifest contains 158 cases: the previous
144 graphics regressions plus 14 MRT cases. `mesa/` entries resolve
against this checkout and `piglit/` entries against `PIGLIT_ROOT`. The upstream
Piglit shaders are used unchanged. Local fixtures live in `tests-graphics/`;
loop reductions from the earlier investigation are retained in its `loops/`
subdirectory.

An explicit case list is also accepted:

```sh
src/asahi/drm-shim/piglit/run-graphics.sh ../logs/apple9-volume-new \
  src/asahi/drm-shim/piglit/tests-graphics/volume-lod.shader_test
```

After another build/reset/chainload, run the MRT API harness (requires a C
compiler and the EGL/epoxy development packages):

```sh
src/asahi/drm-shim/piglit/run-mrt-api.sh ../logs/apple9-mrt-api-new
```

The common runner also accepts `ASAHI_GRAPHICS_BINARY` for upstream Piglit
executables. Arguments following the result directory go to that executable;
`ASAHI_GRAPHICS_EXPECTED_RESULTS` defaults to one. For example:

```sh
ASAHI_GRAPHICS_BINARY="$PIGLIT_BUILD/bin/fbo-drawbuffers" \
  src/asahi/drm-shim/piglit/run-graphics.sh ../logs/apple9-masked-clear-new masked-clear
```

`MESA_BUILD`, `M1N1_SHIM_ROOT`, `PIGLIT_ROOT`, `PIGLIT_BUILD`, and `PIGLIT_LOCAL`
can override the adjacent-checkout defaults. `T8132_PIGLIT_TIMEOUT` defaults to
300 seconds. The existing `setup.sh` installs the development test dependencies.

Each run writes `run.log` and `summary.json`. The wrapper checks every Piglit
result and the expected result count, so a zero runner exit status cannot hide
failing subtests. Existing result logs are never replaced.

## Initial feature validation recorded on 2026-09-09

Hardware: M4 Mac mini, Mac16,10 / J773g, T8132, standalone m1n1 DRM shim.

- Graphics: **144/144 pass**, including the earlier 65 regressions.
- Compiler: **236/236 pass**.
- Compute regression: **22/22 pass**.
- Deferred register-pressure cases: **five rejected**, each followed immediately
  by a passing supported draw in the same process; no process abort or GPU hang.
  These rejected execution tests are not counted as passes.
- This initial 144-case run predates the completed MRT setup. The MRT
  validation is recorded separately below.

The corresponding graphics result record is retained as
`tests-graphics/validation-2026-09-09.json`.

## Completed MRT validation on 2026-09-09

The final T8132 build passes:

- **158/158 graphics shader tests**, including all 144 earlier regressions and
  the 14 count/dimension MRT fixtures.
- **7/7 MRT API tests** through `run-mrt-api.sh`, also **7/7 on llvmpipe**.
- **10/10 unchanged upstream MRT runs**: `fbo-drawbuffers` (ordinary and
  masked-clear), `fbo-drawbuffers-fragcolor`, `fbo-drawbuffers2-colormask`
  (ordinary and clear), `fbo-drawbuffers2-blend`, `fbo-draw-buffers-blend`,
  `arb_draw_buffers-state_change`, `fbo-drawbuffers-blend-add`, and
  `fbo-drawbuffers-maxtargets`.
- **238/238 compiler tests**, including lowered legacy color broadcast and
  packing after undefined alpha is trimmed.
- **22/22 compute regression cases**.

The final review also removed a native-capture vertex-address adjustment that
had incorrectly been treated as an attachment-count field. The 158-case
suite and seven API cases passed again after that cleanup. No diagnostic
shader override remains in the compiler.

Machine-readable results are in
[`tests-graphics/validation-mrt-2026-09-09.json`](tests-graphics/validation-mrt-2026-09-09.json).

## Dense vertex varying commits

`run-varying-commit-api.sh NEW_RESULTS_DIRECTORY` builds and runs a repeated
indexed-grid regression after the usual fresh m1n1 chainload. Six vertex shaders
cover passthrough, ceil, negate, integer comparison, arithmetic, and values
clamped during rasterization. Each draw submits 8,649 vertices and 16,928
triangles through client arrays and checks the complete viewport against a
color range that excludes the clear color. This detects missing primitives
without depending on exact interpolation rounding.

The default is 1,000 frames per shader (6,000 draws). `VARYING_COMMIT_FRAMES`
selects a positive frame count; `T8132_PIGLIT_TIMEOUT` defaults to 900 seconds.
The original T8132 failure left intermittent spans of roughly 24 clear pixels
when varying stores were separated by later loads. Scheduling the commits at
the end of their execution region passed all 6,000 draws of the development
reproducer. The compiler test also checks that scheduling preserves store order,
mask boundaries, and block ownership.

## Texture completion handoffs

`run-completion-handoff-api.sh NEW_RESULTS_DIRECTORY` exercises first consumption
of each texture-result component, retained reuse of that component, later reads
of its siblings, and a dependent second texture access. Five GLSL 1.20 programs
run 100 draws each and compare every RGBA byte against a CPU oracle. These
checks complement the shared-allocator tests for dependent load addresses and
explicit draw-ID/coverage completion. Native Linux can build `api.c` against
libepoxy and run it with surfaceless EGL and the intended Mesa library.
