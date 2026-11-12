"""Estimate end-to-end HARP latency and compare speedups."""

import argparse
from math import exp, log
from pathlib import Path
from statistics import mean

from .baselines.common import ROOT, digest, read, write
from .harp_latency import SERVICE_TABLE, estimate, load_cases


def geomean(values):
    return exp(mean(log(value) for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=SERVICE_TABLE)
    parser.add_argument("--workload-tasks", type=Path)
    parser.add_argument("--formation-report", type=Path, default=ROOT / "results/harp-paper-audit/formation.json")
    parser.add_argument("--baseline-report", type=Path, default=ROOT / "results/baseline-paper-audit/report.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    cases, hardware = load_cases(args.workload_tasks)
    estimates = estimate(cases, hardware, read(args.table))
    target = read(ROOT / "reference/harp-speedups.json")["values"]
    anchor_data = read(ROOT / "reference/e2e-speedups.json")
    references = anchor_data["workloads"]
    baselines = read(args.baseline_report)["baselines"]
    raw = read(args.formation_report)["workloads"]
    if not (set(estimates) == set(target) == set(references) == set(raw)):
        raise ValueError("Mismatched workload coverage")
    result = {}
    for name, timing in estimates.items():
        if timing["workload_sha256"] != raw[name]["workload_sha256"]:
            raise ValueError(f"Formation report workload differs: {name}")
        h100 = references[name]["H100-Dense"]["latency_ms"]
        speedup = h100 / timing["estimated_full_harp_ms"]
        baseline_speedups = {method: row["workloads"][name]["speedup_vs_reference_h100"]
                             for method, row in baselines.items()}
        result[name] = {
            **timing, "h100_reference_ms": h100,
            "unfitted_full_harp_ms": raw[name]["full_cycles_table_ii"] / hardware["clock_hz"] * 1000,
            "speedup_vs_reference_h100": speedup, "historical_speedup_vs_h100": target[name],
            "relative_error": speedup / target[name] - 1,
            "reconstructed_baseline_speedups_vs_h100": baseline_speedups,
            "speedup_vs_reconstructed_dstar": speedup / baseline_speedups["dstar"],
        }
    write(args.out, {
        "schema_version": 1, "model": "reconstructed_harp_e2e_paper_audit",
        "timing": "Empirical component service times fitted to historical Full HARP E2E latency, "
                  "with Table II exposed formation applied once",
        "reference_evidence": anchor_data["evidence"],
        "kernel_latency_table_sha256": digest(args.table),
        "workloads": result,
        "geomean_speedup_vs_reference_h100": geomean(row["speedup_vs_reference_h100"] for row in result.values()),
        "geomean_speedup_vs_reconstructed_dstar": geomean(
            row["speedup_vs_reconstructed_dstar"] for row in result.values()),
        "max_relative_error": max(abs(row["relative_error"]) for row in result.values()),
    })


if __name__ == "__main__":
    main()

