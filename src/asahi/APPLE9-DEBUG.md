# Apple9 diagnostics

Use `AGX_APPLE9_DEBUG=asm,stats` to select diagnostic output. The flags below
only inspect compilation or runtime state. Their values and legacy aliases do
not split Honeykrisp shader caches. Dumps still require a compile: disable the
shader cache explicitly when a cached shader would otherwise suppress output.

| Flag | Legacy alias | Legacy parsing |
| --- | --- | --- |
| `asm` | `AGX_APPLE9_ASM` | Presence |
| `binary` | `AGX_APPLE9_DUMP_BINARY` | Presence |
| `stats` | `AGX_APPLE9_STATS` | Presence |
| `trace` | `AGX_APPLE9_TRACE` | Presence |
| `nir` | `AGX_APPLE9_TRACE_NIR` | Presence |
| `final_nir` | `AGX_APPLE9_TRACE_FINAL_NIR` | Presence |
| `shared_ra` | `AGX_APPLE9_TRACE_SHARED_RA` | Presence |
| `ra_audit` | `AGX_APPLE9_RA_AUDIT` | Presence |
| `ra_webs` | `AGX_APPLE9_RA_WEBS` | Presence; also enable `ra_audit` |
| `spill_loops` | `AGX_SPILL_TRACE_LOOPS` | Boolean |
| `pressure` | `HK_APPLE9_PRESSURE_STATS` | Presence |
| `vectorize` | `HK_APPLE9_TRACE_VECTORIZE` | Presence |
| `eager` | `HK_APPLE9_EAGER_TRACE` | Presence |
| `merge` | `HK_APPLE9_MERGE_TRACE` | Boolean |
| `tiling` | `HK_TILING_ELISION_STATS` | Boolean |
| `launch_check` | `HK_APPLE9_LAUNCH_CHECK` | Boolean |
| `bo_labels` | `AGX_BO_TRACE_LABELS` | Boolean |
| `replay` | `HK_REPLAY_DEBUG` | Boolean |

Presence aliases enable output for any value, including an empty string or
`0`. Boolean aliases use Mesa's boolean parser, so `0` and `false` disable
them. Namespace flags and enabled aliases combine. Options are read once per
process. `launch_check` validates emitted launch templates against the normal
packer and aborts on a mismatch; it does not select a different command stream.

The existing timeline/timing controls and fragment-dump output directory keep
their interfaces. `AGX_MESA_DEBUG` remains separate because it mixes upstream
compiler diagnostics with code-generation controls. Active mesh, hoisting,
register-allocation and command-stream research controls keep their names and
semantics. Do not add code-generation controls to `AGX_APPLE9_DEBUG`.

## Shipping defaults and retired experiments

Spill-home coalescing has been enabled by default since `ca623583d40`. Focused
and corpus gates must use that shipping default, rather than setting
`AGX_SPILL_COALESCE=1`. The explicit opt-out remains for active research.
`AGX_APPLE9_NO_BFE` and both of its conflicting readers were already removed
in `6beee3d9a7f`; there is no remaining parsing discrepancy to fix.

The October 23 cleanup removes the unused spill loop-handling opt-out, texel
class limit, and uniform-safe class selector. Loop-aware spilling, complete
texel class branches, and metadata-only uniform-safe reads remain the shipping
paths. Ordinary UBO and SSBO loads do not acquire the metadata hoisting flag.
