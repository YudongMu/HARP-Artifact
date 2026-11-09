"""Rebuild baseline kernel service tables and replay historical speedup points."""

import argparse
from math import exp, log
from pathlib import Path
from statistics import mean

from .common import ROOT, read, write
from .component_compare import BASELINES
from .components.common import model_inputs, workload_profile
from .kernel_latencies import demands, execute, fit_latencies
from ..dataflow import build, read_workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=ROOT / "configs/kernel-latencies")
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads/dataflow")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    specs = read_workloads(args.workloads)
    graphs = {name: build(spec) for name, spec in specs.items()}
    mapping = read(ROOT / "configs/baseline-comparison.json")["anchor_methods"]
    anchor_data = read(ROOT / "reference/e2e-speedups.json")
    reference = anchor_data["workloads"]
    if set(graphs) != set(reference):
        raise ValueError("Reference and workload coverage differ")
    paths = {method: args.tables / f"{method}.json" for method in BASELINES}
    if args.rebuild and any(path.exists() for path in paths.values()):
        parser.error("Rebuilding requires a fresh kernel table directory")
    result = {}
    for method, module in BASELINES.items():
        profile, _timing, config, _inputs = model_inputs(method, normalized=True)
        phases = {name: module.phases(graph, workload_profile(profile, name))
                  for name, graph in graphs.items()}
        if args.rebuild:
            features, catalog, initial, targets = [], {}, {}, []
            for name, graph in graphs.items():
                counts, kernels, nominal = demands(graph, config, phases[name])
                features.append(counts)
                catalog.update(kernels)
                initial.update(nominal)
                anchors = reference[name]
                targets.append(anchors["H100-Dense"]["latency_ms"] * 1e6 /
                               anchors[mapping[method]]["speedup_vs_h100"])
            calibrated = fit_latencies(features, targets, initial)
            table = {"schema_version": 1, "model": "empirical_kernel_service_times_v0.3",
                     "method": method, "unit": "ns_per_tile",
                     "kernels": {key: {**catalog[key], "latency_ns": calibrated[key]}
                                 for key in sorted(calibrated)}}
            write(paths[method], table)
        else:
            table = read(paths[method])
            if table["method"] != method:
                raise ValueError("Kernel table baseline identity mismatch")
        estimates = {}
        for name, graph in graphs.items():
            outcome = execute(graph, config, phases[name], table)
            anchors = reference[name]
            speedup = anchors["H100-Dense"]["latency_ms"] / outcome["latency_ms"]
            target = anchors[mapping[method]]["speedup_vs_h100"]
            estimates[name] = {**outcome, "speedup_vs_reference_h100": speedup,
                               "target_speedup": target, "relative_error": speedup / target - 1}
        result[method] = {
            "workloads": estimates,
            "speedup_geomean": exp(mean(log(row["speedup_vs_reference_h100"]) for row in estimates.values())),
            "max_relative_error": max(abs(row["relative_error"]) for row in estimates.values()),
        }
    write(args.out, {"schema_version": 1, "model": "reconstructed_baseline_kernel_latencies_v0.3",
                     "timing": "sum_of_empirical_component_kernel_services",
                     "reference_evidence": anchor_data["evidence"], "baselines": result})


if __name__ == "__main__":
    main()

