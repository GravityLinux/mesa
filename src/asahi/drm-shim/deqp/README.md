# T8132 standalone dEQP-GLES2

Current development tests run on the native Linux Asahi DRM driver using
`linux-m4-integration/tools/gpu/asahi/run_g16_deqp.sh` in the workspace.
See the adjacent `G16_DEQP.md` there for build and execution instructions.
The standalone transport below is retained for historical diagnostics.
The result summarizer and deferred register-pressure list are transport independent.

This runner executes unmodified Khronos GLES2 tests through Mesa and the
m1n1 hardware transport on the M4 mini. It does not override the GL or GLSL
version. The surface is a 256x256 RGBA8888, depth24/stencil8, single-sample
pbuffer; smaller surfaces cannot exercise the rasterization limit cases.
This is development validation, not a conformance claim.

Build VK-GL-CTS with its upstream dependencies and the surfaceless target:

```sh
python vk-gl-cts/external/fetch_sources.py --protocol https
cmake -S vk-gl-cts -B deqp-build -G Ninja \
  -DDEQP_TARGET=surfaceless -DCMAKE_BUILD_TYPE=Release \
  -DSELECTED_BUILD_TARGETS=deqp-gles2
ninja -C deqp-build deqp-gles2
```

The initial development run uses upstream commit
`659bbe6987197b4ff7ac20011261b92009286100` (19,825 cases, including performance
and stress). Build Mesa, reset the target through the controller Mac, and
chainload `m1n1-m4-agx/build/m1n1.bin` before **each process**. Do not run two GPU
clients concurrently. Follow the workspace `AGENTS.md` common preparation;
standalone graphics needs display enabled in the local m1n1 configuration.

From the workspace root:

```sh
mesa-m1n1-shim/src/asahi/drm-shim/deqp/run-gles2.sh \
  --deqp-case='dEQP-GLES2.functional.*' \
  --deqp-log-filename=/absolute/path/functional.qpa \
  > /absolute/path/functional.log 2>&1
python mesa-m1n1-shim/src/asahi/drm-shim/deqp/summarize.py \
  /absolute/path/functional.qpa --cts-source vk-gl-cts
```

`DEQP_BINARY`, `MESA_BUILD`, `M1N1_SHIM_ROOT`, `M1N1DEVICE`, and
`M1N1_SHIM_PYTHON` can select alternate builds and transport configuration.
The archive path defaults to the executable's directory, which must contain
the upstream GLES2 test resources. The JSON summary retains every case and
status, flags interrupted sessions, and exits unsuccessfully for failures or
incomplete logs. Unsupported cases and quality warnings remain distinct from
passes.

The GLES2 work adds ordinary point and line primitive encoding, shader point
size and point-coordinate interpolation, byte-index widening through Mesa's
existing vertex-buffer utility, and integer division through NIR's general
quotient/refinement lowering. Unsigned float-to-integer conversion uses the
unsigned T8132 format bit, avoiding the earlier 16-bit saturation. Division
coverage also includes the Piglit `integer-division.shader_test` fixture's
15 signed edge cases, including values near the 32-bit limits. Narrow integer
to float conversion explicitly extends byte and short inputs before the
32-bit machine conversion, including optimized vertex attribute loads.
Cube faces are accepted as renderable surfaces, and logical packed depth
formats retain the transfer helper's depth32f storage path.

The ESSL 1.00 frontend now gives the function body a scope nested inside the
parameter scope, as specified by section 4.2.2 of the
[ESSL 1.00 specification](https://www.khronos.org/files/opengles_shading_language.pdf).
Later language versions keep their existing scope rules. A 12-case compiler
check covers accepted shadowing and rejected duplicate declarations.

Further hardware failures exposed shared state and compiler issues:

- CPU write mappings synchronize their baseline with the remote GPU before
  changing bytes. This preserves writes of an old CPU value over newer GPU
  output, including recycled buffer objects. Real DRM mappings need no extra
  transport synchronization.
- Resource shadow copies fetch completed GPU writes before preserving the
  untouched contents for a partial CPU update with pending GPU readers.
- Scoreboard pressure accounts for fixed texture slots as well as automatic
  buffer-load slots, materializing live values before a slot is reused.
- The sampler publishes the API cube seam mode in Apple9 descriptor bit 63.
  GLES2 linear filtering stays within the selected cube face.
- Fragment stages carry NIR's coarse quad helper requirement into raster
  state. Disabling triangle merging preserves derivatives across primitives
  with different perspective interpolation planes.
- Texture samples preserve helper invocations for later samples and
  derivatives. Retiring helpers at each sample corrupted multi-texture LOD
  along shared triangle edges when triangle merging was disabled.

The texture-state API regression lives in
[`../piglit/run-texture-state-api.sh`](../piglit/run-texture-state-api.sh).
It covers rebinding, independent samplers, sampling two rendered textures,
rewriting an old CPU value over GPU output, and a partial update that preserves
GPU-written pixels while an earlier draw still reads the texture. All five
passed on T8132 and llvmpipe without inserting per-draw finishes.

Register starvation and spilling remain deferred. The explicit list in
[`deferred-register-pressure.txt`](deferred-register-pressure.txt) contains
five dynamic-indexing shaders, six nested uniform-array rendering cases, and
two random uniform cases exposed by enabling vertex textures on native Linux.
Two of the indexing shaders exhaust registers only in the full sequence;
they passed in a fresh process. Keep their allocator diagnostics alongside
the raw results and report all thirteen separately from semantic failures.
