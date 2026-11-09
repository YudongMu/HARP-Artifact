import unittest

from hardware.baselines.kernel_latencies import fit_latencies, predict_ns
from hardware.reconstruct_harp_e2e import with_exposed_formation


class HarpE2ECalibrationTest(unittest.TestCase):
    def test_lower_service_times_fit_target_without_increasing_nominal_costs(self):
        counts = [{"shared": 3, "image": 10}, {"shared": 2, "video": 30}]
        initial = {"shared": 4, "image": 20, "video": 30}
        table = fit_latencies(counts, [100, 300], initial, cap_at_initial=True)
        for row, target in zip(counts, [100, 300]):
            self.assertAlmostEqual(predict_ns(row, table), target, delta=1e-5)
        self.assertTrue(all(0 < table[key] <= initial[key] for key in table))

    def test_table_ii_exposure_is_fraction_of_full_time(self):
        full, exposure = with_exposed_formation(98, 0.02)
        self.assertAlmostEqual(full, 100)
        self.assertAlmostEqual(exposure / full, 0.02)
        self.assertEqual(with_exposed_formation(98, 0), (98, 0))

