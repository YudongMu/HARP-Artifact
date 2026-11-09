import unittest

from hardware.baselines.h100 import summarize


class H100Test(unittest.TestCase):
    def test_excludes_warmups_and_keeps_all_measurements(self):
        result = summarize([50] * 5, list(range(1, 21)))
        self.assertEqual(result["median_ms"], 10.5)
        self.assertEqual(len(result["measurements_ms"]), 20)

    def test_rejects_incomplete_or_nonpositive_measurements(self):
        for values in ([2] * 19, [2] * 19 + [0], [2] * 19 + [float("nan")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize([5] * 5, values)

