#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Decode aggregate HK GPU busy counters. Revisions replace, never add."""
import argparse
import csv
import json
from pathlib import Path

SECOND = 1_000_000_000
ENGINES = ("union", "vertex", "fragment", "compute")
COVERAGE = ("requested", "written", "collected", "app_owned", "pool_full",
            "initial", "invalid", "too_late", "queries", "query_errors")


def parse(paths, pid=None):
    sessions = []
    active = {}
    for path in paths:
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            fields = line.split()
            if not fields or fields[0] not in ("GPU_BUSY_CONFIG", "GPU_BUSY", "GPU_BUSY_SAMPLE"):
                continue
            row = {}
            for field in fields[1:]:
                key, value = field.split("=", 1)
                try:
                    row[key] = int(value, 0)
                except ValueError:
                    row[key] = value
            if pid is not None and row["pid"] != pid:
                continue
            key = row["pid"], row["device"]
            if fields[0] == "GPU_BUSY_CONFIG":
                session = {"config": row, "rows": {}, "samples": [],
                           "source": str(path), "config_line": number}
                sessions.append(session)
                active[key] = session
            elif key not in active:
                raise ValueError(f"{path}:{number}: no preceding configuration for {key}")
            elif fields[0] == "GPU_BUSY_SAMPLE":
                active[key]["samples"].append(row)
            else:
                rows = active[key]["rows"]
                old = rows.get(row["gpu_second"])
                if old is None or row["revision"] >= old["revision"]:
                    rows[row["gpu_second"]] = row
    return sessions


def summarize(session, start, end, clock="cpu"):
    config = session["config"]
    before, after = config[f"{clock}_before_ns"], config[f"{clock}_after_ns"]
    anchor = config["gpu_anchor_ns"]
    if not anchor or after < before or end <= start:
        raise ValueError("invalid clock anchor or measurement window")
    # Exclude edge seconds even at the extremes of GET_TIME call uncertainty.
    first = (anchor + start - before + SECOND - 1) // SECOND
    stop = (anchor + end - after) // SECOND
    count = max(0, stop - first)
    if clock == "cpu":
        cpu_start, cpu_end = start, end
    else:
        # Conservative bounds when translating wall time to guest monotonic.
        cpu_start = config["cpu_before_ns"] + start - after
        cpu_end = config["cpu_after_ns"] + end - before
    samples = sorted(session["samples"], key=lambda row: row["cpu_ns"])
    previous = [row for row in samples if row["cpu_ns"] <= cpu_start]
    baseline = previous[-1] if previous else {}
    latest = samples[-1] if samples else {}
    drained = bool(latest and latest["cpu_ns"] >= cpu_end and
                   (latest["pending_slots"] == 0 or
                    latest.get("oldest_pending_cpu_ns", 0) > cpu_end))
    delta = {name: latest.get(name, 0) - baseline.get(name, 0) for name in COVERAGE}
    series = []
    sums = dict.fromkeys(ENGINES, 0)
    missing, overflow = 0, False
    for second in range(first, stop):
        row = session["rows"].get(second)
        missing += row is None
        values = {engine: row[f"{engine}_ns"] if row else 0 for engine in ENGINES}
        if any(value < 0 or value > SECOND for value in values.values()):
            raise ValueError(f"out-of-range union in GPU second {second}")
        overflow |= bool(row and row["overflow"])
        for engine, value in values.items():
            sums[engine] += value
        series.append({"gpu_second": second, "revision": row["revision"] if row else 0,
                       "has_row": row is not None,
                       **{f"{engine}_percent": value / 10_000_000
                          for engine, value in values.items()}})
    lower_bound = (not drained or overflow or
                   any(delta[key] > 0 for key in
                       ("app_owned", "pool_full", "initial", "invalid", "too_late", "query_errors")))
    result = {
        "pid": config["pid"], "device": hex(config["device"]),
        "source": session["source"], "config_line": session["config_line"],
        "scope": "measured commands of one VkDevice; not whole-machine GPU utilization",
        "input_clock": clock, "window_ns": [start, end],
        "anchor_call_uncertainty_ns": after - before,
        "anchor_clock_drift_assumed_negligible": True,
        "full_gpu_seconds": [first, stop], "full_second_count": count,
        "window_drained": drained, "lower_bound": bool(lower_bound),
        "seconds_without_rows": missing, "interval_overflow": overflow,
        "coverage_counter_delta_through_last_sample": delta,
        "last_sample_cpu_ns": latest.get("cpu_ns"),
        "oldest_pending_cpu_ns": latest.get("oldest_pending_cpu_ns"),
        "busy_ns": sums,
        "mean_percent": {engine: total / (count * 10_000_000) if count else None
                         for engine, total in sums.items()},
    }
    return result, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--clock", choices=("cpu", "unix"), default="cpu",
                        help="cpu means the game's own monotonic clock, not the VMM host's")
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--series-csv")
    args = parser.parse_args()
    results, all_series = [], []
    for session in parse(args.logs, args.pid):
        result, series = summarize(session, args.start_ns, args.end_ns, args.clock)
        if not result["full_second_count"]:
            continue
        results.append(result)
        all_series.extend({"pid": result["pid"], "device": result["device"], **row}
                          for row in series)
    if not results:
        parser.error("no matching configured VkDevice with full seconds in the window")
    if args.series_csv and all_series:
        with open(args.series_csv, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_series[0]))
            writer.writeheader()
            writer.writerows(all_series)
    print(json.dumps({"devices": results}, indent=2))


if __name__ == "__main__":
    main()
