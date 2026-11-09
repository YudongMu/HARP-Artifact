from math import isclose
from statistics import geometric_mean
import unittest

from hardware.baselines.common import ROOT, read
from hardware.dataflow import read_workloads
from hardware.harp_workload import load_profiles


class PaperAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.paper = read(ROOT / "reference/harp-revised.json")

    def test_table_i_dimensions_and_default_profiles(self):
        graphs = read_workloads(ROOT / "workloads/dataflow")
        profiles, _ = load_profiles(ROOT / "workloads/harp/revised", set(graphs))
        self.assertEqual(set(graphs), set(self.paper["workloads"]))
        for name, expected in self.paper["workloads"].items():
            for field in ("width", "heads", "steps"):
                self.assertEqual(graphs[name][field], expected[field], (name, field))
            self.assertEqual(profiles[name]["selected_fraction"], expected["selected_region_fraction"])
            self.assertEqual(profiles[name]["refresh_interval"], self.paper["algorithm"]["refresh_interval"])

    def test_table_ii_latency_and_hardware_parameters(self):
        timing = read(ROOT / "reference/e2e-speedups.json")["workloads"]
        harp = read(ROOT / "reference/harp-speedups.json")["values"]
        formation = read(ROOT / "reference/harp-table-ii.json")["workloads"]
        hardware = read(ROOT / "configs/harp.json")["hardware"]
        for name, expected in self.paper["workloads"].items():
            h100 = timing[name]["H100-Dense"]["latency_ms"]
            self.assertAlmostEqual(h100, expected["h100_ms_printed"], delta=0.00051)
            self.assertAlmostEqual(h100 / harp[name], expected["harp_ms_printed"], delta=0.00051)
            self.assertEqual(formation[name], expected["formation_exposed_fraction"])
        physical = self.paper["hardware"]
        self.assertEqual(hardware["clock_hz"], physical["clock_hz"])
        self.assertEqual(hardware["buffer_bytes"], physical["sram_bytes"])
        self.assertEqual(hardware["rates"]["pe"], physical["dense_macs_per_cycle"])
        self.assertEqual(hardware["rates"]["hbm"], physical["hbm_bytes_per_cycle"])

    def test_figure_6_and_7_rounding(self):
        harp = read(ROOT / "reference/harp-speedups.json")["values"]
        timing = read(ROOT / "reference/e2e-speedups.json")["workloads"]
        self.assertAlmostEqual(geometric_mean(harp.values()), self.paper["figure_6"]["harp_vs_h100"], places=2)
        for method in ("BitRed", "DSTAR"):
            ratio = geometric_mean(harp[n] / timing[n][method + "-Adapt"]["speedup_vs_h100"]
                                   for n in harp)
            self.assertAlmostEqual(ratio, self.paper["figure_6"]["harp_vs_" + method.lower()], places=2)
        modes = read(ROOT / "reference/harp-ablation.json")["workloads"]
        for mode in ("residual", "planeclear", "combined_generic", "combined_harp"):
            mean = geometric_mean(row[mode]["speedup"] for row in modes.values())
            self.assertAlmostEqual(mean, self.paper["figure_7"][mode], places=2)
        gain = geometric_mean(row["combined_harp"]["speedup"] / row["combined_generic"]["speedup"]
                              for row in modes.values())
        self.assertTrue(isclose(gain, self.paper["figure_7"]["hardware_gain"], abs_tol=.005))

