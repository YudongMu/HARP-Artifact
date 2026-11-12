from statistics import geometric_mean
import unittest

from hardware.baselines.common import ROOT, read
from hardware.harp_latency import SERVICE_TABLE, estimate, load_cases
from hardware.matched_npu_ablation import reconstruct


class MatchedNPUAblationTest(unittest.TestCase):
    def test_replay_matches_reference_using_shared_harp_model(self):
        inputs = read(ROOT / "workloads/harp/matched-npu.json")
        e2e = read(ROOT / "results/harp-paper-audit/e2e.json")
        cases, hardware = load_cases()
        timing = estimate(cases, hardware, read(SERVICE_TABLE))
        report = reconstruct(inputs, {name: row["estimated_full_harp_ms"] for name, row in timing.items()})
        self.assertEqual(set(report["workloads"]), set(e2e["workloads"]))
        expected = {"residual": 1.334534324, "planeclear": 3.377617618,
                    "budget": 3.696720309, "all": 6.837920362}
        for name, row in report["workloads"].items():
            self.assertEqual(row["latency_ms"]["all"], e2e["workloads"][name]["estimated_full_harp_ms"])
            for mode, latency in row["latency_ms"].items():
                self.assertAlmostEqual(row["speedup"][mode], row["matched_npu_dense_ms"] / latency)
        for mode, speedup in expected.items():
            self.assertAlmostEqual(report["geomean_speedup"][mode], speedup, places=8)
        npu_over_h100 = geometric_mean(
            row["matched_npu_dense_ms"] / e2e["workloads"][name]["h100_reference_ms"]
            for name, row in report["workloads"].items()
        )
        self.assertAlmostEqual(
            report["geomean_speedup"]["all"],
            e2e["geomean_speedup_vs_reference_h100"] * npu_over_h100,
        )
