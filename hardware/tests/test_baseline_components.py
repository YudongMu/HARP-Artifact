import unittest

from hardware.baselines import component_compare
from hardware.component_timing import Stage
from hardware.dataflow import build, read_workloads


class BaselineComponentsTest(unittest.TestCase):
    def test_initial_and_differential_ditto_phases(self):
        graph = build(read_workloads(component_compare.ROOT / "workloads/dataflow")["dit-xl2-512"])
        layer = graph["bundles"][0]["layers"][0]
        node = next(n for n in layer["nodes"] if n["id"] == "ffn_in")
        phases = component_compare.ditto.phases(graph, {"difference_issue_fraction": 0.5})
        self.assertEqual([count for _, count, _ in phases], [1, 48, 1])
        first = phases[0][2](node, layer, 2)
        middle = phases[1][2](node, layer, 2)
        self.assertIn(Stage("Ditto_cache_seed", "hbm", layer["tensors"]["ffn_in"]["bytes"] * 2), first.join)
        self.assertTrue(any(stage.name == "Ditto_Encoding_Unit" for stage in middle.serial))
        self.assertTrue(any(stage.name == "Ditto_cached_output_read" for stage in middle.parallel))

    def test_adaptiv_attention_work_tracks_retained_token_pairs(self):
        graph = build(read_workloads(component_compare.ROOT / "workloads/dataflow")["dit-xl2-512"])
        layer = graph["bundles"][0]["layers"][0]
        node = next(n for n in layer["nodes"] if n["id"] == "self.qk")
        plan = component_compare.adaptiv.phases(graph, {"token_fraction": 0.5})[0][2]
        self.assertEqual(plan(node, layer, 2).serial[0].units, node["macs"] // 2)

