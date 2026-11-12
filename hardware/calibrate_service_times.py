"""Regenerate a service-time table from the reconstruction calibration inputs."""

import argparse
from pathlib import Path

from .baselines.common import ROOT, read, write
from .baselines.component_compare import BASELINES
from .baselines.components.common import model_inputs, workload_profile
from .baselines.kernel_latencies import demands, fit_latencies
from .dataflow import build, read_workloads
from .harp_latency import load_cases


def calibration_cases(method, references):
    if method == "harp":
        cases, hardware = load_cases()
        speedups = read(ROOT / "reference/harp-speedups.json")["values"]
        targets = {
            name: (references[name]["H100-Dense"]["latency_ms"] / speedups[name])
            * (1 - case["formation_exposed_fraction"]) * 1e6
            for name, case in cases.items()
        }
    else:
        profile, _, hardware, _ = model_inputs(method, normalized=True)
        specs = read_workloads(ROOT / "workloads/dataflow")
        anchor = read(ROOT / "configs/baseline-comparison.json")["anchor_methods"][method]
        cases, targets = {}, {}
        for name, spec in specs.items():
            graph = build(spec)
            phases = BASELINES[method].phases(graph, workload_profile(profile, name))
            cases[name] = {"graph": graph, "phases": phases}
            row = references[name]
            targets[name] = row["H100-Dense"]["latency_ms"] * 1e6 / row[anchor]["speedup_vs_h100"]
    return cases, hardware, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("harp", *BASELINES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    references = read(ROOT / "reference/e2e-speedups.json")["workloads"]
    cases, hardware, targets = calibration_cases(args.method, references)
    examples, catalog, initial = [], {}, {}
    for case in cases.values():
        counts, kernels, nominal = demands(case["graph"], hardware, case["phases"])
        examples.append(counts)
        catalog.update(kernels)
        initial.update(nominal)
    fitted = fit_latencies(examples, [targets[name] for name in cases], initial,
                           cap_at_initial=args.method == "harp")
    write(args.out, {
        "schema_version": 1,
        "model": "empirical_kernel_service_times_v0.3",
        "method": args.method,
        "unit": "ns_per_tile",
        "kernels": {key: {**catalog[key], "latency_ns": value} for key, value in fitted.items()},
    })


if __name__ == "__main__":
    main()
