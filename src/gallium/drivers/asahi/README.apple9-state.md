# Apple9 graphics state ownership

The graphics path uses compiled-shader ownership and batch-owned immutable
records, following the existing pre-Apple9 AGX lifetime model while retaining
Apple9 packet and launch encodings. It no longer creates render packages or
compares multi-megabyte images to discover changed byte ranges.

## Records and lifetimes

| Data | Owner | Construction/publication |
| --- | --- | --- |
| Shader body | Compiled shader; referenced by every using batch | One executable BO upload per compiled variant; no attachment-dependent copies or program-byte interning |
| Stage resource-pointer table | Batch ordinary pool | Explicit GPU addresses; reuse the previous table when the address sequence is unchanged |
| Stage roots and launch | Batch USC pool | 256-byte root allocation plus 256-byte launch allocation per changed stage; commands use its USC-relative address |
| Coefficient bindings | Batch USC pool | 544-byte record; reuse for the same stage identities and interpolation mode |
| PPP state | Batch context pool | Complete 256-byte canonical record; reuse an identical predecessor record in CPU storage |
| Scissor/depth bias | Batch CPU arrays, uploaded with the batch | Reuse unchanged scissor records and unchanged enabled rasterizer depth-bias state |
| VDM stream | Batch context pool | Copy only the finalized stream length; firmware receives this allocation's address |
| Attachment graph | Device transport, serialized at submission | Independent framebuffer identity; 3 KiB of complete records per view when targets/layout change |
| Entry thunks | Device short-entry aperture | Publish only slots whose shader-body address changed; no image scan |
| Invariant header/context records | Device transport | Initialize once, or rebuild after explicit invalidation |

Batch cleanup follows the completion syncobj before releasing its pools.
Shader deletion may drop the compiled object's reference immediately: using
batches keep their own BO references. Batch-local reuse never reads or mutates
another batch's storage. Context switches do not inherit a prior context's
record-reuse state.

Framebuffer identity includes target addresses, dimensions, formats, target
count, and samples. Those fields are not shader-object identity. Compiler keys
still retain the format/sample information actually needed by generated tile
programs. Disabled attachment slots are explicitly zeroed when constructing a
new graph, so shrinking and later growing an MRT set cannot revive old fields.

## Addressability

Shader code, launch records and coefficients use the existing 4-GiB USC heap
at `shader_base`. The VDM vertex-launch and fragment-launch fields contain the
allocated launch address in 64-byte units. Coefficient fields retain the
USC-relative byte address.

PPP and VDM records use `AGX_BO_CONTEXT`, backed by a separate allocator for
32-bit offsets from the Apple9 render-context base. Its lower 64 MiB remain
reserved for the existing fixed client/firmware aliases. Its upper limit is
bounded by the kernel-private carveout and the 4-GiB context aperture. It does
not overlap the ordinary resource heap or the USC heap.

The launch instruction's short target field still selects a thunk in the small
fixed entry table. The existing header alias and attachment graph placement
are retained. This is a bounded publication transport, not a pipeline cache.
Submission still waits for the preceding fixed-USC user before replacing entry
slots or selecting the compute/render backing. Relocating draw records does
not by itself authorize removing that wait.

Apple9 continues to emit complete native draw packets referencing these
records. Reusing record allocations does not assume unverified persistence of
omitted packet fields. No Apple8 packet encoding is substituted for Apple9.

## Removed work and remaining work

Removed: attachment/shader package LRU, multi-megabyte per-package BOs and CPU
images, exact predecessor transition caches, whole-image `memcmp` scans,
shader-byte interning after per-package code copies, fixed per-draw launch/
coefficient/PPP slots, and whole-reserved-encoder clearing/copying.

A bounded 256-byte PPP equality check and a small scissor equality check remain.
These compare canonical CPU records to reuse allocations; they never inspect
large GPU arenas or compute patch ranges. Ordinary small shader/meta cache
keys elsewhere in Mesa are unaffected.

The fixed header/context allocation size and synchronous transport remain.
Linear texture sampling still uses its existing staging copies, limited to
the current shader's bindful texture range. Utility reload/store draws do not
refresh unused inherited sampler views. These are
separate costs and must be profiled independently of the removed package work.

## Regression coverage

`linux-m4-integration/tools/gpu/asahi/plasma-es3/state-churn.c` in the workspace
keeps 24 framebuffer/program combinations in each of three contexts, uses
changing dimensions and scissor/color-mask state, switches between two and one
active render targets, and deletes programs before draining the last pending
draws. It verifies both attachments, including preservation of the disabled
one. `--compute` additionally alternates graphics with checked SSBO dispatches.
Context-release flushing is explicitly disabled.

Compile with the target EGL/GLES/GBM development packages. Normal rendering
uses `g16-es3-exec`; the compute variant additionally uses the development-only
GLES 3.1 and GLSL 310 overrides. These overrides are not desktop defaults.

The existing eight-context test, imported/GBM linear-render matrices, mixed
linear/tiled MRT tests, and imported-texture sampler checks cover the same
submission and resource-lifetime changes. See the desktop bring-up README for
native results and the before/after KWin measurement.
