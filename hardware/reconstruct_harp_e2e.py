"""Estimate end-to-end HARP latency and compare speedups."""

import argparse
from math import exp, log
from pathlib import Path
from statistics import mean

from .baselines.common import PERFORMANCE_INPUTS, digest, read, write
from .harp_latency import SERVICE_TABLE, estimate, load_cases


def geomean(values):
    return exp(mean(log(value) for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=SERVICE_TABLE)
    parser.add_argument("--workload-tasks", type=Path)
    parser.add_argument("--formation-report", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    cases, hardware = load_cases(args.workload_tasks)
    estimates = estimate(cases, hardware, read(args.table))
    baselines = read(args.baseline_report)["baselines"] if args.baseline_report else {}
    raw = read(args.formation_report)["workloads"] if args.formation_report else None
    if raw is not None and set(raw) != set(estimates):
        raise ValueError("Formation report workload coverage differs")
    result = {}
    for name, timing in estimates.items():
        h100 = cases[name]["h100_reference_ms"]
        speedup = h100 / timing["estimated_full_harp_ms"]
        row = {
            **timing, "h100_reference_ms": h100,
            "speedup_vs_reference_h100": speedup,
        }
        if raw is not None:
            if timing["workload_sha256"] != raw[name]["workload_sha256"]:
                raise ValueError(f"Formation report workload differs: {name}")
            row["unfitted_full_harp_ms"] = raw[name]["full_cycles_table_ii"] / hardware["clock_hz"] * 1000
        if baselines:
            baseline_speedups = {method: h100 / baseline["workloads"][name]["latency_ms"]
                                 for method, baseline in baselines.items()}
            row["reconstructed_baseline_speedups_vs_h100"] = baseline_speedups
            row["speedup_vs_reconstructed_dstar"] = speedup / baseline_speedups["dstar"]
        result[name] = row
    report = {
        "schema_version": 2, "model": "harp_e2e",
        "timing": "Released calibrated service times with configured formation exposure",
        "performance_inputs_sha256": digest(PERFORMANCE_INPUTS),
        "kernel_latency_table_sha256": digest(args.table),
        "workloads": result,
        "geomean_speedup_vs_reference_h100": geomean(row["speedup_vs_reference_h100"] for row in result.values()),
    }
    if baselines:
        report["geomean_speedup_vs_reconstructed_dstar"] = geomean(
            row["speedup_vs_reconstructed_dstar"] for row in result.values())
    write(args.out, report)


if __name__ == "__main__":
    main()

