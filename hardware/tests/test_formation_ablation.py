import unittest

from hardware.formation_ablation import aligned_cycles


class FormationAblationTest(unittest.TestCase):
    def test_exposed_fraction_is_relative_to_full_latency(self):
        self.assertEqual(aligned_cycles(980, 0.02), 1000)
        self.assertEqual(aligned_cycles(980, 0), 980)

    def test_invalid_exposure_is_rejected(self):
        with self.assertRaises(ValueError):
            aligned_cycles(1000, 1)

