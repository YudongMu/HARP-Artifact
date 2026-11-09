import unittest

from hardware.baselines.common import ROOT
from hardware.dataflow import build, read_workloads
from hardware.harp_components import qk_issue_split
from hardware.harp_workload import compile_workload, load_profiles


class HarpModelProfilesTest(unittest.TestCase):
    def test_q_and_k_retained_planes_are_multiplied(self):
        self.assertEqual(qk_issue_split(10, 2, 72, 0, 4, 32, key_retained_planes=6), (0, 1440))

    def test_profiles_change_residual_and_plane_work_by_model(self):
        specs = read_workloads(ROOT / "workloads/dataflow")
        profiles, _ = load_profiles(ROOT / "workloads/harp/revised", set(specs))
        for name, selected in (("dit-xl2-512", 0.3933), ("wan21-14b-81f", 0.2139)):
            with self.subTest(workload=name):
                graph = build(specs[name])
                profile = profiles[name]
                work = compile_workload(graph, profile, "combined")["counts"]
                residual = work["qkpv_mac_equivalent_removed_net"] / work["dense_qkpv_macs"]
                self.assertAlmostEqual(residual, 0.8 - selected, places=6)
                self.assertGreater(work["qk_plane_pairs_removed_active"], 0)
                self.assertEqual(work["qk_plane_pairs_before_active"],
                                 work["qk_plane_pairs_issued"] + work["qk_plane_pairs_removed_active"])

