"""Evaluate modeled and configured PlaneClear formation exposure."""

import argparse
from copy import deepcopy
from fractions import Fraction
from math import ceil, exp, log
from pathlib import Path
from statistics import mean

from .baselines.common import PERFORMANCE_INPUTS, ROOT, digest, read, write
from .dataflow import build, read_workloads
from .harp_ablation import run
from .harp_workload import fingerprint


def aligned_cycles(without_formation, exposed_fraction):
    """Compute full latency when exposure is a share of full latency."""
    share = Fraction(str(exposed_fraction))
    if without_formation <= 0 or not 0 <= share < 1:
        raise ValueError("Invalid formation timing")
    return ceil(without_formation / (1 - share))


def without_formation(workload):
    """Disable formation service for a timing counterfactual."""
    result = deepcopy(workload)
    for phase in result["phases"]:
        for nodes in phase["layers"].values():
            for plan in nodes.values():
                for group in ("serial", "parallel", "join"):
                    plan[group] = [stage for stage in plan[group] if stage["engine"] != "plane_formation"]
    result.pop("sha256")
    result["sha256"] = fingerprint(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.ablation_dir
    report_path, tasks_path = root / "report.json", root / "workload.json"
    hardware_path = ROOT / "configs/harp.json"
    backend_path = ROOT / "configs/harp-ablation.json"
    report, tasks = read(report_path), read(tasks_path)
    inputs = read(PERFORMANCE_INPUTS)["workloads"]
    if digest(tasks_path) != report["compiled_tasks_sha256"]:
        raise ValueError("Saved workload and ablation report differ")
    workload_dir = ROOT / "workloads/dataflow"
    specs = read_workloads(workload_dir)
    if set(specs) != set(tasks["workloads"]) or set(specs) != set(inputs):
        raise ValueError("Inconsistent workload coverage")
    hardware = read(hardware_path)["hardware"]
    backend = read(backend_path)["backends"]["harp"]
    result = {}
    for name, spec in specs.items():
        if digest(workload_dir / f"{name}.json") != report["workload_sha256"][name]:
            raise ValueError(f"Workload definition changed: {name}")
        graph = build(spec)
        workload = tasks["workloads"][name]["combined"]
        payload = {key: value for key, value in workload.items() if key != "sha256"}
        if fingerprint(payload) != workload["sha256"]:
            raise ValueError(f"Invalid workload fingerprint: {name}")
        previous = report["workloads"][name]["timings"]
        full = run(graph, workload, hardware, backend)
        if full["cycles"] != previous["combined_harp"]["cycles"]:
            raise ValueError(f"Component timing replay differs: {name}")
        no_formation = run(graph, without_formation(workload), hardware, backend)
        raw_cycles = full["cycles"]
        exposed = raw_cycles - no_formation["cycles"]
        if exposed < 0:
            raise ValueError(f"Formation counterfactual increased latency: {name}")
        busy = sum(cycles for component, cycles in full["component_cycles"].items()
                   if component.endswith("PlaneClear_formation"))
        share = inputs[name]["formation_exposed_fraction"]
        cycles = aligned_cycles(no_formation["cycles"], share)
        dense = previous["dense"]["cycles"]
        generic = previous["combined_generic"]["cycles"]
        methods = ("residual", "planeclear", "combined_generic", "combined_harp")
        result[name] = {
            "workload_sha256": workload["sha256"],
            "formation_busy_cycles": busy, "exposed_formation_cycles": exposed,
            "exposed_formation_fraction": exposed / raw_cycles,
            "table_ii_exposed_fraction": share,
            "full_cycles_raw": raw_cycles,
            "full_cycles_without_formation_service": no_formation["cycles"],
            "full_cycles_table_ii": cycles,
            "modeled_speedups": {method: previous[method]["speedup_vs_modeled_dense"]
                                 for method in methods},
            "table_ii_full_speedup_vs_modeled_dense": dense / cycles,
            "table_ii_hardware_gain_fixed_combined": generic / cycles,
        }
    write(args.out, {
        "schema_version": 2, "model": "harp_formation_exposure_v0.5",
        "scope": "Full HARP critical-path formation sensitivity and configured exposure",
        "source_sha256": {path.name: digest(path) for path in (
            report_path, tasks_path, PERFORMANCE_INPUTS, hardware_path, backend_path, Path(__file__),
            ROOT / "harp_ablation.py", ROOT / "component_timing.py")},
        "workloads": result,
        "table_ii_full_speedup_geomean": exp(mean(log(row["table_ii_full_speedup_vs_modeled_dense"])
                                               for row in result.values())),
        "table_ii_hardware_gain_geomean": exp(mean(log(row["table_ii_hardware_gain_fixed_combined"])
                                                for row in result.values())),
    })


if __name__ == "__main__":
    main()

