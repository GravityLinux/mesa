# Aggregate application GPU busy counters

Enable `-Dhk-gpu-busy-counters=true` for the native and guest **test** builds.
The option defaults off and changes only the HK compilation flags. Use it for
all compared runs, including the baseline. It is independent of the long-wait
watchdog; enabling both lets the watchdog describe an unexpected diagnostic
query stall too.

The driver requests existing UAPI firmware timestamps at vertex, fragment and
compute command boundaries. This is additional firmware work, so an enabled
run includes that instrumentation cost. There are no command-record logs and
no forced GPU waits. Existing application timestamp pairs are preserved;
occupied pairs are counted as uncovered rather than replaced.

Each queue has one 256 KiB shared, uncached timestamp BO with 8192 command slots.
The slot metadata is bounded too. Once per second, on submission activity, a
non-waiting completed-point syncobj query for each queue permits timestamp
collection and slot recycling. There is no query per submission or command.
The first queue point is skipped because HK initially signals its private
progress object to 1. Queue teardown performs a final non-waiting collection.
If submissions stop entirely, another sample appears on the next submission or
queue teardown; this does not add a periodic GPU-waking background thread.

Output defaults to `/tmp/hk-gpu-busy-PID.log`. Override with `HK_GPU_BUSY_FILE`;
`%p` expands to the process ID. Each file contains:

* `GPU_BUSY_CONFIG`: GPU timestamp frequency and a GPU-clock/CPU-monotonic and realtime
  anchors bounded by the two CPU readings around `GET_TIME`. GPU command
  timestamps use the reported frequency; `GET_TIME` already reports nanoseconds.
* `GPU_BUSY`: exact union of measured intervals for a GPU-clock second,
  across **all queues of one VkDevice**, and independent unions per engine.
  `union_ns / 10000000` is percentage busy for that second. Each union is at
  most one second even when engines or queues overlap.
* `GPU_BUSY_SAMPLE`: cumulative coverage/error counts and pending slots,
  emitted approximately once per second with submission activity. The oldest
  pending submission time makes incomplete-window detection explicit.

**Keep only the greatest revision for each `(device, gpu_second)` key.**
Completed work can arrive late, so an updated row replaces an earlier union;
adding revisions would double count. Use the clock anchor to select GPU seconds
inside the CPU measurement window, and exclude partially covered edge seconds
for a full-second average. These are the application's measured intervals, not
whole-machine GPU utilization: other Vulkan devices, KWin and other processes
are outside this VkDevice's union.

A timestamp pair occupied by an application query, an exhausted slot pool,
invalid timestamps, intervals arriving outside the 64-second retained-bin
window, or interval-storage overflow makes the result a **lower bound**.
Check `app_owned`, `pool_full`, `invalid`, `too_late`, `query_errors`, and each
row's `overflow`. Each bin stores at most 8192 disjoint ranges per union and
reports overflow instead of silently claiming full coverage. No slot is
recycled merely because an end timestamp happens to be nonzero; completion of
its submitted private queue timeline point is required first.

Host test `hk_gpu_busy` covers overlapping engines, out-of-order intervals,
duplicates, adjacency, second boundaries, bounded overflow, and 100,000
insertions against an independent bitmap oracle. Native/virtio hardware smoke
validation is still required before treating measured values as validated.

Decode a measurement using the game's own monotonic clock, or use `--clock unix`
for Unix nanoseconds shared by the host and guest:

```sh
python3 src/asahi/ci/m4/gpu-busy-decode.py /path/gpu-busy-PID.log \
  --clock unix --start-ns START_UNIX_NS --end-ns END_UNIX_NS \
  --series-csv /path/gpu-busy-series.csv
```

The decoder retains final revisions, excludes edge seconds at both extremes of
the clock-anchor call uncertainty, and reports per-engine and combined busy
percentages plus coverage and drain status. The anchor assumes negligible
clock drift; its call uncertainty does not bound NTP clock steps after startup.
Take the file after at least one collection past the measurement window (or
normal queue teardown), and require `window_drained=true` for complete eligible
work. An unobserved/undrained second is not evidence that the GPU was idle.

The interval bound is sized from the CS2 baseline: 78,295 requested pairs over
289 presents (~271 pairs/frame), and a peak GPU-clock second containing 2,141
pairs. The former 2,048-range limit overflowed in two of the 59 measured full
seconds. An 8,192-range bound covers ~30 frames at that observed average
cardinality and 3.8 times the observed peak second, even if every interval is
disjoint. This is headroom, not a promise about all future workloads; overflow
remains explicit. Ranges are allocated lazily, so normal bins retain their
previous allocations. The hard maximum for all 64 retained bins and four unions
rises from 8 MiB to 32 MiB per VkDevice, with no change to timestamp BO size,
query cadence or firmware timestamp requests.

`GPU_BUSY_CONFIG` records `ranges_per_union`. Each aggregate row reports the
four current range counts and `overflow_mask` (bits 0/1/2/3 are overall,
vertex, fragment and compute). Preserve `overflow=0`, all coverage checks and
`window_drained=true` when describing a rerun as exact. The old overflowing
logs cannot be repaired into exact interval unions without rerunning.
