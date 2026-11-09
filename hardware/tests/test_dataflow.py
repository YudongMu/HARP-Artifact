import json
from pathlib import Path
import unittest

from hardware.dataflow import build, read_workloads, window_histogram


ROOT = Path(__file__).resolve().parents[1]


class DataflowTest(unittest.TestCase):
    def test_dit_matches_archived_macs_and_cfg(self):
        specs = read_workloads(ROOT / "workloads" / "dataflow")
        result = build(specs["dit-xl2-512"])
        receipt = json.loads((ROOT.parent / "results/dit-micro-floor-smoke-v1/summary.json").read_text())
        self.assertEqual(result["forward_calls"], 50)
        self.assertEqual(result["cfg_lane_evaluations"], 100)
        self.assertEqual(result["macs"], receipt["dense_transformer_block_macs"])
        self.assertEqual(result["self_attention_macs"], receipt["dense_qkpv_macs"])

    def test_wan_keeps_separate_guidance_and_cross_attention(self):
        result = build(read_workloads(ROOT / "workloads" / "dataflow")["wan21-14b-81f"])
        self.assertEqual(result["forward_calls"], 100)
        self.assertEqual(result["cfg_lane_evaluations"], 100)
        group = result["bundles"][0]["layers"][0]
        self.assertEqual(group["tokens"], 32760)
        self.assertEqual(group["region_geometry"]["query_regions"], 2184)
        self.assertEqual(group["region_geometry"]["region_pairs"], 4769856)
        cross = next(node for node in group["nodes"] if node["id"] == "cross.qk")
        self.assertEqual(cross["macs"], 32760 * 512 * 5120)

    def test_window_histogram_counts_real_edge_tokens(self):
        self.assertEqual(window_histogram((2, 5, 7), (1, 4, 4)), {16: 2, 12: 2, 4: 2, 3: 2})

    def test_flux_graph_records_joint_and_fused_single_streams(self):
        result = build(read_workloads(ROOT / "workloads" / "dataflow")["flux1-dev"])
        groups = result["bundles"]
        self.assertEqual([group["repeat"] for group in groups], [19, 38])
        self.assertIn("text.ffn_in", {n["id"] for n in groups[0]["layers"][0]["nodes"]})
        self.assertIn("packed_in", {n["id"] for n in groups[1]["layers"][0]["nodes"]})

