#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import pathlib
import runpy
import tempfile
import unittest

mod = runpy.run_path(str(pathlib.Path(__file__).parents[2] / "ci/m4/gpu-busy-decode.py"))

class Decode(unittest.TestCase):
    def sample(self, drain=True):
        return {
            "config": dict(pid=44, device=0x123, gpu_anchor_ns=10_000_000_000,
                           cpu_before_ns=2_000_000_000, cpu_after_ns=2_000_000_100,
                           unix_before_ns=100_000_000_000, unix_after_ns=100_000_000_100),
            "source": "test", "config_line": 1,
            "rows": {12: dict(revision=2, union_ns=500_000_000, vertex_ns=300_000_000,
                              fragment_ns=400_000_000, compute_ns=0, overflow=0)},
            "samples": [dict(cpu_ns=8_000_000_000, pending_slots=1 if not drain else 0,
                             oldest_pending_cpu_ns=3_000_000_000 if not drain else 0)],
        }

    def test_uncertain_edges_and_union(self):
        result, series = mod["summarize"](self.sample(), 3_000_000_050, 6_000_000_050)
        self.assertEqual(result["full_gpu_seconds"], [12, 13])
        self.assertEqual(result["mean_percent"]["union"], 50)
        self.assertEqual(result["mean_percent"]["vertex"], 30)
        self.assertEqual(result["mean_percent"]["fragment"], 40)
        self.assertTrue(result["window_drained"])
        self.assertFalse(result["lower_bound"])
        self.assertEqual(series[0]["revision"], 2)

    def test_realtime_and_undrained(self):
        result, _ = mod["summarize"](self.sample(False), 101_000_000_050,
                                     104_000_000_050, "unix")
        self.assertEqual(result["full_gpu_seconds"], [12, 13])
        self.assertFalse(result["window_drained"])
        self.assertTrue(result["lower_bound"])

    def test_late_revision_replaces(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            name = f.name
            f.write("GPU_BUSY_CONFIG pid=44 device=0x123 gpu_anchor_ns=1\n")
            f.write("GPU_BUSY pid=44 device=0x123 gpu_second=9 revision=1 union_ns=100\n")
            f.write("GPU_BUSY pid=44 device=0x123 gpu_second=9 revision=2 union_ns=150\n")
            f.write("GPU_BUSY pid=44 device=0x123 gpu_second=9 revision=1 union_ns=100\n")
        try:
            sessions = mod["parse"]([name])
            self.assertEqual(sessions[0]["rows"][9]["union_ns"], 150)
        finally:
            pathlib.Path(name).unlink()

if __name__ == "__main__":
    unittest.main()
