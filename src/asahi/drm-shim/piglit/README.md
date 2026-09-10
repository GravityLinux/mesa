# T8132 Apple9 Piglit harness

The graphics suite and its feature limits are documented in [GRAPHICS.md](GRAPHICS.md).
Run it with `run-graphics.sh NEW_RESULTS_DIRECTORY` after a cold chainload.

This directory contains compute profiles for the m1n1-backed DRM shim. The
GLES profile drives `t8132_apple9_compute_runner`, a native test binary whose
named cases use ordinary GLES 3.1 shaders and compare complete guarded buffers
against independent CPU formulas. Piglit batches those logical cases in one
process and one EGL context, but records each case as a separate subtest and
can resume at the next case after a failure.

The current T8132 hardware baseline is 117 passes with complete output and
guard checks. The runner continues after ordinary oracle mismatches so one
failure cannot hide later results.

The geometry group also covers native indirect dispatch rather than only
direct-grid bookkeeping. Its exact guarded-buffer cases include asymmetric 2D
and 3D records, nonzero indirect-buffer offsets, a zero group dimension, and a
record written by an ordinary preceding compute shader. Dedicated direct and
indirect cases read all three components of `gl_NumWorkGroups`; the latter is
also exercised with a GPU-authored record and no CPU readback. The ten-case
geometry group passes in one cold-boot process on T8132. The complete native
batch reaches 137 exact hardware publications because several lifecycle cases
contain multiple dispatches.

The desktop OpenGL profile remains declarative. Its default set contains three
negative linker tests, three upstream variable-workgroup-size tests, and nine
exact SSBO execution tests. The complete 35-test `ARB_compute_shader` discovery
corpus remains available explicitly. Both profiles use Waffle's surfaceless
EGL backend; GLX and X11 are not required.

Install the pinned development dependencies beside Mesa:

```sh
src/asahi/drm-shim/piglit/setup.sh
```

The setup keeps the Waffle and Piglit source checkouts unchanged. It installs
Waffle under `../piglit-local`, builds Piglit's desktop shader runner under
`../piglit-build`, and places two ignored symlinks in Piglit's `tests` package
so its controller can import the Mesa-owned native and desktop profiles.

After resetting and chainloading the target, run the batch directly:

```sh
src/asahi/drm-shim/piglit/run.sh direct
```

Run the same batch through Piglit's result controller:

```sh
src/asahi/drm-shim/piglit/run.sh piglit
```

The `direct` and `piglit` names are GLES aliases. Run the desktop OpenGL
compute corpus with:

```sh
src/asahi/drm-shim/piglit/run.sh direct-gl
src/asahi/drm-shim/piglit/run.sh piglit-gl
```

Set `T8132_PIGLIT_GL_SUBSET=smoke`, `supported`, `linker-basic`, `linker`,
`execution`, or `all` to select one exact desktop SSBO test, the enabled
15-test set, five basic link tests, all seven link tests, the 28 upstream
execution tests, or the complete discovery corpus. `supported` is the default.
The broader subsets intentionally include compiler features that are not yet
implemented.

The enabled upstream cases are the negative-link tests `no_local_work_size`,
`mismatched_local_work_sizes`, and `mix_compute_and_non_compute`, plus
the three `ARB_compute_variable_group_size` global-ID tests. They execute a
runtime 2x1x1 local group and check global, local, and workgroup-derived IDs,
including two multi-source link arrangements.
The local variable-size case additionally checks every component of
`gl_NumWorkGroups` with a non-power-of-two runtime local tuple. The complete
15-case supported desktop batch passes in one cold-boot process on T8132.
The remaining upstream positive-link tests need empty compute programs, while
the other execution tests require general control flow, atomics, images,
shared memory, barriers, subgroup operations, or other unsupported facilities.

The controller receives no `LD_PRELOAD`. Each profile applies the shim, Mesa
EGL/Gallium paths, m1n1 Python path, and surfaceless platform only to its GPU
child. Controller results default to a timestamped directory under
`../piglit-results`; an explicit unused result path may be passed as the second
argument.

Set `MESA_SKIP_BUILD=1` to skip the incremental Mesa and Piglit builds. The
usual `MESA_BUILD`, `M1N1_SHIM_ROOT`, `PIGLIT_ROOT`, `PIGLIT_BUILD`, and
`PIGLIT_LOCAL` overrides are supported. The default controller timeout is 300
seconds for either batch;
`T8132_PIGLIT_TIMEOUT` overrides either value.
