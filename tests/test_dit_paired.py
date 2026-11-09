import copy
import unittest

from quality.dit_paired import MODES, compare_generations, evaluation_items, paired_items


def runs():
    rows = [
        dict(item, noise_sha256=f"noise-{item['index']}")
        for item in paired_items(4)
    ]
    return {
        mode: {
            "mode": mode,
            "status": "complete",
            "count": 4,
            "completed": 4,
            "paired_contract": {"batch_size": 2, "checkpoint_sha256": "test"},
            "samples": copy.deepcopy(rows),
        }
        for mode in MODES
    }


class PairedQualityTests(unittest.TestCase):
    def test_primary_cohort_is_spread_across_imagenet_and_nested_in_5k(self):
        primary = evaluation_items(100)
        expanded = evaluation_items(5000)
        self.assertEqual([item["class_id"] for item in primary], list(range(0, 1000, 10)))
        self.assertEqual(
            [(item["class_id"], item["seed"]) for item in primary],
            [(item["class_id"], item["seed"]) for item in expanded[:1000:10]],
        )
        self.assertEqual(expanded, paired_items(5000))

    def test_extension_accepts_dense_and_residual_only(self):
        generated = {mode: runs()[mode] for mode in ("dense", "residual")}
        self.assertEqual(len(compare_generations(generated, 4, modes=tuple(generated))), 4)

    def test_accepts_complete_matched_inputs(self):
        report = compare_generations(runs(), 4)
        self.assertEqual(len(report), 4)

    def test_rejects_changed_noise(self):
        generated = runs()
        generated["combined"]["samples"][1]["noise_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "Paired inputs"):
            compare_generations(generated, 4)

    def test_rejects_changed_sampling_contract(self):
        generated = runs()
        generated["residual"]["paired_contract"]["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "Sampling contract"):
            compare_generations(generated, 4)

    def test_rejects_incomplete_mode(self):
        generated = runs()
        generated["planeclear"]["completed"] = 3
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            compare_generations(generated, 4)
