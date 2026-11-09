from copy import deepcopy
import unittest

from hardware.baselines import component_compare
from hardware.baselines.components.common import model_inputs, workload_profile
from hardware.baselines.common import read
from hardware.baselines.kernel_latencies import execute
from hardware.dataflow import build, read_workloads

from hardware.baselines.kernel_latencies import fit_latencies, predict_ns


class KernelLatencyTest(unittest.TestCase):
    def test_shared_kernel_times_reproduce_two_trajectories(self):
        counts = [{"shared": 3, "image": 10}, {"shared": 2, "video": 30}]
        table = fit_latencies(counts, [100, 300], {"shared": 4, "image": 20, "video": 30})
        for row, target in zip(counts, [100, 300]):
            self.assertAlmostEqual(predict_ns(row, table), target, delta=1e-5)
        self.assertTrue(all(value > 0 for value in table.values()))

    def test_execution_recomputes_changed_task_counts(self):
        latencies = {"projection": 12, "attention": 20}
        first = predict_ns({"projection": 10, "attention": 5}, latencies)
        second = predict_ns({"projection": 20, "attention": 10}, latencies)
        self.assertEqual(second, first * 2)

    def test_unknown_shapes_require_explicit_kernel_entry(self):
        with self.assertRaises(KeyError):
            predict_ns({"new_kernel": 1}, {"old_kernel": 5})



    def test_stored_tables_replay_all_thirty_speedups(self):
        root = component_compare.ROOT
        graphs = {name: build(spec) for name, spec in read_workloads(root / "workloads/dataflow").items()}
        anchors = read(root / "reference/e2e-speedups.json")["workloads"]
        methods = read(root / "configs/baseline-comparison.json")["anchor_methods"]
        for method, module in component_compare.BASELINES.items():
            profile, _, config, _ = model_inputs(method, normalized=True)
            table = read(root / "configs/kernel-latencies" / f"{method}.json")
            for name, graph in graphs.items():
                with self.subTest(method=method, workload=name):
                    settings = workload_profile(profile, name)
                    result = execute(graph, config, module.phases(graph, settings), table)
                    speedup = anchors[name]["H100-Dense"]["latency_ms"] / result["latency_ms"]
                    expected = anchors[name][methods[method]]["speedup_vs_h100"]
                    self.assertAlmostEqual(speedup / expected, 1, delta=1e-7)

    def test_reconstructed_model_responds_to_changed_steps(self):
        root = component_compare.ROOT
        spec = read_workloads(root / "workloads/dataflow")["dit-xl2-512"]
        profile, _, config, _ = model_inputs("dstar", normalized=True)
        settings = workload_profile(profile, spec["id"])
        table = read(root / "configs/kernel-latencies/dstar.json")
        graph = build(spec)
        before = execute(graph, config, component_compare.dstar.phases(graph, settings), table)
        changed = deepcopy(spec)
        changed["steps"] *= 2
        graph = build(changed)
        after = execute(graph, config, component_compare.dstar.phases(graph, settings), table)
        self.assertGreater(after["latency_ms"], 1.5 * before["latency_ms"])
