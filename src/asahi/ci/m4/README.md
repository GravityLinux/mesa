# Native M4 Mesa CI

Use Mesa's `deqp-runner suite` for routine M4 testing. The default profile is
[`deqp-asahi-m4-gl33-gles30.toml`](../deqp-asahi-m4-gl33-gles30.toml), derived from
the upstream Asahi GL profile. It covers OpenGL 3.3, GLES 2.0 and 3.0, and EGL on
Wayland and X11. The surface settings and batch sizes match upstream Asahi.

The API-specific changes are the GL 3.3 mustpass list, omission of GLES 3.1
lists, an ES 3.0 version check, and the Apple M4 renderer check. The version
check accepts our build's `Mesa 26.3.0-devel` string without requiring the
`git` suffix used by upstream CI. No API or extension overrides are used.

## Running

The persistent target has the launcher installed as `g16-mesa-ci`:

```sh
# Complete GL 3.3 / GLES 3.0 / EGL CI profile.
ssh root@m4-target g16-mesa-ci gl

# Focused CTS run, retaining the same profile and upstream exclusions.
ssh root@m4-target \
  "g16-mesa-ci gl --include-tests '^KHR-GL33[.]nearest_edge[.]'"

# Run against a particular installed candidate without changing the desktop.
ssh root@m4-target \
  g16-mesa-ci gl --driver /path/to/candidate-prefix

# Supplemental Piglit GPU profile, also through Mesa's runner.
ssh root@m4-target \
  "g16-mesa-ci piglit --include-tests '^spec@!opengl 1[.]1@draw-pixels$'"
```

`--jobs` defaults to 2 while the desktop is running. Other options include
`--output`, `--timeout`, `--max-fails`, and `--prepare-only`. An existing output
directory is never reused. Scheduling, timeouts, retries, filtering, and result
classification are handled by Mesa's runner; `run.py` only prepares paths,
environment, and provenance, then propagates its exit status.

Runs default to `/g16/results/mesa-ci/<timestamp>-<suite>/`. Each contains the
resolved `suite.toml`, combined upstream baseline, and `run.json` with the
command, driver and test-input hashes, environment, and exit status. Mesa's
`results/` contains raw results, failure CSVs, and expectation metadata.
Keep raw failures, skips, expected failures, and flakes distinct when reporting
a run. Mesa's runner can return success when a failure is expected or flaky.
This CI profile is distinct from a full Khronos certification configuration
matrix; use the official conformance procedure when that is explicitly needed.

## Upstream expectations and Piglit

The launcher supplies the existing `all`, `asahi`, and `asahi-agx2` skip, flake,
single-thread, and expected-failure files, as applicable. It does not create
new expected failures from local test results. The global skip list includes
`streaming-texture-leak`, so routine runs exclude that workload. Direct test
executables remain useful for intentional, focused debugging.

Upstream Asahi does not define a Piglit suite. The supplemental `piglit.toml`
uses Piglit's `gpu` profile with process isolation and the same expectation
files. This is an additional local profile, not an upstream Asahi Piglit preset.

## Installation and updates

Native configuration is `/g16/tools/mesa-ci/config.json`. It selects the runner,
driver prefix, display environment, runtime path mappings, and result directory.
The checked-in TOML retains Mesa's `/deqp-gl` and `/deqp-gles` path conventions;
the launcher resolves them using that machine configuration.

Mesa's pinned runner version is currently 0.23.3, from
`.gitlab-ci/container/build-deqp-runner.sh`. It was installed natively with:

```sh
cargo install deqp-runner --version 0.23.3 --locked \
  --root /g16/tools/mesa-ci/runner-0.23.3 -j 4
```

The deployment has this layout:

- `ci/`: this profile and launcher, upstream expectation files, and provenance.
- `deqp-gl/`: GL CTS runtime and the unchanged `gl33-main.txt` list.
- `deqp-gles/`: GLES CTS runtime, GLES 2/3 lists, and both EGL executables.
- `runner-0.23.3/`: the native Mesa runner installation.

The existing CTS source is `opengl-es-cts-3.2.14.1`, with the four previously
documented corrections used for our GL 3.3 validation. It is not a fresh build
of Mesa CI's separate OpenGL CTS release. The exact source and binary provenance
is retained in `ci/cts-source-provenance.json`; each run hashes its executables
and case lists. Both standalone EGL executables were built from the existing
source using the `wayland` and `x11_egl_glx` CMake targets. None of the previous
CTS binaries or result archives were replaced.

The native `ci/` is a deployed copy. After updating the profile, launcher, or
upstream expectation files, copy those changes to the same relative paths there
and refresh `ci/provenance.json`. Do not modify the upstream Apple8 profile for
M4-specific settings. When updating runner or CTS versions, update the machine
configuration and provenance together, then repeat a filtered native smoke run.

## Setup verification, September 16, 2026

- 20 selected CTS/EGL results pass: GLES 2/3 and GL 3.3 identity checks, draws,
  shader checks, framebuffer operations, and EGL pbuffer rendering on Wayland
  and X11.
- The supplemental Piglit smoke run records 276 passes and exactly one skip:
  `streaming-texture-leak`, excluded by upstream's global skip list.
- These are setup checks, not a completed run of the full profile. The installed
  desktop driver and running Minecraft session were preserved.

Native evidence: `/g16/results/mesa-ci/setup-smoke2/` and
`/g16/results/mesa-ci/setup-piglit-smoke/`.

## Experimental Apple9 Vulkan

`run.py vk` uses `deqp-asahi-m4-vk.toml` with the same upstream expectations and
result recording. It requires a `/deqp-vk` runtime mapping, the flattened Vulkan
mustpass list at `mustpass/vk-main.txt`, and a separate candidate prefix containing
`lib/libvulkan_asahi.so` and `share/vulkan/icd.d/*asahi*.json`. The manifest must
name the candidate library by absolute path. Configure `HK_EXPERIMENTAL_APPLE9=1`
in `environment`; inherited Vulkan and Mesa selection variables are cleared
before that configuration is applied. Apple9 compression is enabled by default
in both drivers. Leave `ASAHI_MESA_DEBUG` unset for normal regression tests;
use `nocompress` only for an explicit uncompressed comparison.

The September 27 native development deployment uses
`/root/honeykrisp-apple9/ci/config.json` on the Fedora target. Earlier `/g16`
examples describe the older desktop image. Run the new deployment explicitly:

```sh
ssh root@m4-target 'python3 /root/honeykrisp-apple9/ci/m4/run.py vk \
  --config /root/honeykrisp-apple9/ci/config.json \
  --include-tests "^dEQP-VK[.]compute[.]pipeline[.]basic[.]" --jobs 1'
```

This profile is development coverage, not a certification configuration. See
[`README.apple9.md`](../../vulkan/README.apple9.md) for current capabilities and
known failures. Preserve the raw failures rather than turning them into local
expected failures.

For desktop clear coverage, retain the existing
`^dEQP-GLES3[.]functional[.]color_clear[.]` group in focused runs (19 cases).
On October 3 these passed with compression enabled, including with KWin itself
using the candidate. Check the compositor's loaded libraries: a file capability
on `kwin_wayland` makes the loader ignore `LD_LIBRARY_PATH`, so setting that
variable alone does not select the candidate. The installed system driver is
available to the prefix-based runner through `/opt/mesa-apple9-system`, whose
`lib` and `share` directories link to `/usr/lib64` and `/usr/share`.
