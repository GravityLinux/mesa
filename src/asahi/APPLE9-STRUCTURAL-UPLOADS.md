# Structural ownership of Apple9 uploads

Baseline: `20263f73679a4d1f3edca3343a55d88fd84cac43` (integration-next when this work started). Implementation: `f6cb0f71d2072f73c6c7570dac716e25e167116b`, topic `apple9/cpu-structural-uploads` in a separate worktree. Host only; no push or device operation.

## Site inventory

The baseline has one production call of `hk_pool_upload_cached`, for tile-helper attachment metadata. Other occurrences are its definition, declaration, wrapper call and test fixture. The shared `hk_pool_alloc_cached` path also receives interpolation tables and the rare graphics-launch fallback through `hk_apple9_state_cache_alloc`. The PPP callback already bypasses hashing. Normal graphics launches already have compiled-shader-owned templates. The audit covers both the direct upload and these callbacks, plus adjacent repeated immutable packing.

| Data | Baseline work | Ownership after this change |
| --- | --- | --- |
| Fragment interpolation/CF bindings | Build 672 bytes and content-hash when consecutive-draw reuse misses | Compiled fragment shader allocates and fills once; one 704-byte slot, or three slots when flat inputs depend on provoking vertex; draw chooses a pointer |
| Null-FS interpolation interface | Build using the current producer interface during draw preparation | Command buffer owns the derived null fragment stage and table; rebuild when producer program ID changes |
| BG/EOT attachment dimensions | Build/hash/upload per helper | One upload per rendering instance, reused by its four helper variants and internal splits |
| BG/EOT clear arguments | Upload the same 192 bytes per helper | One upload per rendering instance |
| BG/EOT reload sampler | Allocate/zero/pack identical bytes per helper | One device-owned entry alongside the existing fetch sampler |
| BG/EOT texture/store descriptor parts | Unpack, adjust sRGB and translate on every helper use | Image view owns both layered and non-layered packed parts and dimensions; render copies the selected dense table |
| BG/EOT launch encoding | Rebuild each launch | Compiled helper owns existing generalized launch templates; instance supplies addresses |
| Rare normal-launch fallback | Canonical launch hash on template miss/exhaustion | Direct instance allocation/encoding; dynamic address-bearing state stays command-owned |
| PPP record | Callback bypassed cache, with previous-draw byte comparison | Direct command allocation with the same previous-draw comparison |

The content-cache implementation, callbacks, statistics and debug option are removed. Small consecutive-state comparisons in `hk_pool_upload_last` are outside this content-hash removal: their payloads include dynamic draw state and addresses.

## Lifetime and memory

The existing retained 64 KiB native-PBE slab allocator becomes a byte/alignment USC heap shared by image views and fragment shaders. Objects free their suballocations at destruction; slab addresses never move and slabs remain until device destruction. A mutex serializes allocation/free. Normal API shader/variant lifetimes preserve the referenced state through valid command execution; native-body cache eviction only removes a cached BO reference. The shader failure path releases partially initialized state via its existing destroy path.

No per-shader 16 KiB BO is added. Coefficients use 704 bytes for non-flat input or 2,112 bytes when all three provoking vertices are needed. Image-view descriptor parts stay on the CPU because helpers require a dense per-mask table. Changing the shader resource ABI just to reference discontiguous per-view entries is outside this change. Address-bearing roots and launch records remain per instance. Render-owned pointers are cleared by the existing rendering-state initialization and survive internal stream splits/resubmission through command-pool lifetime.

## Validation and acceptance status

Focused host tests passed in release and assertion builds: `hk-memory`, `hk_descriptor_memory`, `apple9-graphics-roots`, `apple9-coefficients`, `hk-structural-uploads`, `hk-fragment-tri-merging`, `hk-native-mesh-hybrid`, `hk-variant-cache`. Additional release checks passed: `hk-native-image-store`, `hk-compiler-env` and the final heap test. These are focused tests, not a full CTS run.

The coefficient test checks 960 interface/interpolation/extra-input/provoking-vertex combinations against the unchanged encoder and compares roots, launch, PPP and VDM bytes at explicitly fixed semantic addresses. The render-view test covers 73 combinations including mip/layer offsets, linear layout, 1/2/4 samples and compression. It also verifies repeated render-input pointer reuse, new-render separation and retained old contents. Heap tests cover mixed descriptor/coefficient alignment, reuse, live allocations and allocation failures.

**Literal deterministic capture identity is not satisfied.** Twelve fixed-64-iteration vkoverhead cases (0, 3, 7, 8, 9, 10, 13, 14, 15, 16, 27, 125) each have two byte-identical baseline repetitions. All twelve candidates differ in raw submit/BO records. Moving ownership changes allocation order, addresses and allocation contents. These observations do not prove that every difference is solely an address relocation. No blanket masking or normalization has been treated as a pass. The original raw-byte requirement remains an open acceptance gate; a pending question asks whether a strict explicit relocation comparison is acceptable. The fixed-address unit tests cover bounded paths and do not substitute for whole-capture identity.

Strict shader comparison, CPU timings and native kit provenance are recorded separately alongside this document. The kit is a review/testing handoff, not an accepted integration candidate, until the command-stream requirement is resolved.

Strict baseline/candidate comparison is complete: 7,008 matching shader entries, 11 matching state-only pipeline-library entries, six matching baseline failures (three CTS, three Willems; keys, exits and assertion expressions match). Strict pre-raster adapters are separate diagnostic builds and are not included as device drivers. Fifteen generated files match between the candidate host release build and baseline AArch64 release build; assertion-build helpers contain extra assert/printf code and are kept separate.
