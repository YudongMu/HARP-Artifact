import unittest

from hardware.trace import import_receipt, validate_trace


class TraceTest(unittest.TestCase):
    def setUp(self):
        self.workload = {"id": "dit-xl2-512", "width": 1152, "tokens": 1024,
                         "steps": 50, "lanes": 2, "blocks": [{"count": 28, "ffn": 4608}]}
        self.receipt = {
            "item": {"id": "input"}, "execution_sha256": "run", "noise_sha256": "noise",
            "identity": {"steps": 50, "refresh_interval": 5, "mode": "residual",
                         "resolution": 512, "batch_size": 1},
            "runtime_counters": {"attention_calls": 1400, "attention_totals": {
                "candidate_region_pairs": 11468800, "selected_region_pairs": 8656525,
                "residual_elements_read": 2642411520, "residual_elements_written": 660602880}},
        }

    def test_decodes_cfg_lanes_and_refresh(self):
        trace = import_receipt(self.receipt, self.workload)
        self.assertEqual(trace["residual"]["refresh_calls"], 560)
        self.assertEqual(trace["residual"]["element_bytes"], 2)
        self.assertEqual(trace["residual"]["selected_pairs"], 8656525)

    def test_mismatched_geometry_is_rejected(self):
        self.receipt["runtime_counters"]["attention_totals"]["candidate_region_pairs"] = 1
        with self.assertRaisesRegex(ValueError, "Region pairs"):
            import_receipt(self.receipt, self.workload)




    def test_rejects_mode_without_required_counters(self):
        trace = import_receipt(self.receipt, self.workload)
        del trace["residual"]
        with self.assertRaisesRegex(ValueError, "residual"):
            validate_trace(trace, self.workload["id"])
