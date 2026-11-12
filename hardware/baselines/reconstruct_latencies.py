"""Evaluate baseline latency using released kernel service tables."""

import argparse
from math import exp, log
from pathlib import Path
from statistics import mean

from .common import PERFORMANCE_INPUTS, ROOT, digest, read, write
from .component_compare import BASELINES
from .components.common import model_inputs, workload_profile
from .kernel_latencies import execute
from ..dataflow import build, read_workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=ROOT / "configs/kernel-latencies")
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads/dataflow")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    specs = read_workloads(args.workloads)
    graphs = {name: build(spec) for name, spec in specs.items()}
    inputs = read(PERFORMANCE_INPUTS)["workloads"]
    if set(graphs) != set(inputs):
        raise ValueError("Timing inputs and workload coverage differ")
    paths = {method: args.tables / f"{method}.json" for method in BASELINES}
    result = {}
    for method, module in BASELINES.items():
        profile, _timing, config, _inputs = model_inputs(method, normalized=True)
        phases = {name: module.phases(graph, workload_profile(profile, name))
                  for name, graph in graphs.items()}
        table = read(paths[method])
        if table["method"] != method:
            raise ValueError("Kernel table baseline identity mismatch")
        estimates = {}
        for name, graph in graphs.items():
            outcome = execute(graph, config, phases[name], table)
            speedup = inputs[name]["h100_reference_ms"] / outcome["latency_ms"]
            estimates[name] = {**outcome, "speedup_vs_reference_h100": speedup}
        result[method] = {
            "workloads": estimates,
            "speedup_geomean": exp(mean(log(row["speedup_vs_reference_h100"]) for row in estimates.values())),
        }
    write(args.out, {"schema_version": 2, "model": "reconstructed_baseline_kernel_latencies_v0.3",
                     "timing": "sum_of_empirical_component_kernel_services",
                     "performance_inputs_sha256": digest(PERFORMANCE_INPUTS),
                     "baselines": result})


if __name__ == "__main__":
    main()

