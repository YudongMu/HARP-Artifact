import unittest

from hardware.baselines.common import ROOT, read
from hardware.dataflow import build, read_workloads
from hardware.harp_ablation import compile_workload, run
from hardware.harp_workload import load_profiles


class HarpAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specs = read_workloads(ROOT / "workloads/dataflow")
        cls.graph = build(specs["dit-xl2-512"])
        profiles, _ = load_profiles(ROOT / "workloads/harp/revised", set(specs))
        cls.profile = profiles["dit-xl2-512"]
        cls.hardware = read(ROOT / "configs/harp.json")["hardware"]
        cls.backends = read(ROOT / "configs/harp-ablation.json")["backends"]

    def test_residual_work_separates_refresh_replay_and_steady(self):
        work = compile_workload(self.graph, self.profile, "residual")
        counts = work["counts"]
        self.assertGreater(counts["replay_token_pairs"], 0)
        self.assertGreater(counts["steady_token_pairs"], counts["replay_token_pairs"])
        self.assertEqual(counts["refresh_token_pairs"] * 5, counts["dense_token_pairs"])
        self.assertEqual(counts["executed_token_pairs"],
                         sum(counts[f"{name}_token_pairs"] for name in ("refresh", "replay", "steady")))
        self.assertEqual(counts["dense_qkpv_macs"], self.graph["self_attention_macs"])

    def test_plane_elimination_and_combined_share_correct_denominators(self):
        planes = compile_workload(self.graph, self.profile, "planeclear")["counts"]
        combined = compile_workload(self.graph, self.profile, "combined")["counts"]
        residual = compile_workload(self.graph, self.profile, "residual")["counts"]
        self.assertEqual(planes["executed_token_pairs"], planes["dense_token_pairs"])
        self.assertEqual(combined["executed_token_pairs"], residual["executed_token_pairs"])
        self.assertGreater(combined["qk_plane_pairs_removed_active"], 0)
        self.assertEqual(combined["qk_plane_pairs_before_active"],
                         combined["qk_plane_pairs_issued"] + combined["qk_plane_pairs_removed_active"])
        self.assertGreater(combined["qk_plane_pairs_removed_natural"], 0)
        self.assertGreater(combined["dense_fallback_macs"], 0)

    def test_hardware_comparison_reuses_identical_work_and_residual_bytes(self):
        work = compile_workload(self.graph, self.profile, "combined")
        generic = run(self.graph, work, self.hardware, self.backends["generic"])
        harp = run(self.graph, work, self.hardware, self.backends["harp"])
        self.assertEqual(generic["workload_sha256"], harp["workload_sha256"])
        self.assertEqual(generic["hbm_bytes"], harp["hbm_bytes"])
        self.assertGreater(generic["cycles"], harp["cycles"])

