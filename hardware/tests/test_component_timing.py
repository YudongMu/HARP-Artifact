import unittest

from hardware.component_timing import Plan, Stage, Task, run_layer, timeline
from hardware.dataflow import Graph


class ComponentTimingTest(unittest.TestCase):
    def test_ready_work_can_run_before_a_blocked_task(self):
        tasks = [
            Task("matrix", "pe", 10),
            Task("late_read", "hbm", 3, ("matrix",)),
            Task("prefetch", "hbm", 7),
        ]
        self.assertEqual(timeline(tasks)["cycles"], 13)

    def test_weights_shared_inside_cfg_batch(self):
        graph = Graph()
        x = graph.tensor("input", (4, 8), storage="layer_input")
        graph.linear("out", x, 8, "ffn_out")
        layer = graph.finish()
        spec = {"lanes_per_forward": 2}
        rates = {"pe": 8, "hbm": 16}
        plan = lambda node, layer, lanes: Plan((Stage("MAC", "pe", node["macs"] * lanes),))
        report = run_layer(layer, spec, rates, 10000, plan)
        self.assertEqual(report["hbm_bytes"], 128 + 128 + 128)
        self.assertEqual(report["component_cycles"]["MAC"], 64)

    def test_parallel_residual_read_can_overlap_matrix_work(self):
        graph = Graph()
        x = graph.tensor("input", (1, 1), storage="layer_input")
        graph.vector("out", "vector", (x,))
        rates = {"pe": 1, "hbm": 1}
        spec = {"lanes_per_forward": 1}
        serial = lambda *_: Plan((Stage("read", "hbm", 4), Stage("MAC", "pe", 10)))
        parallel = lambda *_: Plan((Stage("MAC", "pe", 10),), (Stage("read", "hbm", 4),))
        s = run_layer(graph.finish(), spec, rates, 1000, serial)
        p = run_layer(graph.finish(), spec, rates, 1000, parallel)
        self.assertEqual(s["hbm_bytes"], p["hbm_bytes"])
        self.assertEqual(s["cycles"] - p["cycles"], 4)


    def test_native_weight_precision_changes_transfer_without_changing_compute(self):
        graph = Graph()
        x = graph.tensor("input", (4, 8), storage="layer_input")
        graph.linear("out", x, 8, "ffn_out")
        plan = lambda node, layer, lanes: Plan(
            (Stage("A8W8", "pe", node["macs"] * lanes),),
            weight_bytes=node["weight_bytes"] // 2,
        )
        result = run_layer(graph.finish(), {"lanes_per_forward": 2},
                           {"pe": 8, "hbm": 16}, 10000, plan)
        self.assertEqual(result["hbm_bytes"], 64 + 128 + 128)
        self.assertEqual(result["component_cycles"]["A8W8"], 64)
