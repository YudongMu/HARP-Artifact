"""Replay Full HARP with hardware service times calibrated to historical E2E latency."""

import argparse
from math import exp, log
from pathlib import Path
from statistics import mean

from .baselines.common import ROOT, digest, read, write
from .baselines.kernel_latencies import demands, execute, fit_latencies
from .dataflow import build, read_workloads
from .formation_ablation import without_formation
from .harp_ablation import execution_phases
from .harp_workload import fingerprint


def with_exposed_formation(core_ms, share):
    if core_ms <= 0 or not 0 <= share < 1:
        raise ValueError("Invalid formation exposure")
    full = core_ms / (1 - share)
    return full, full - core_ms


def geomean(values):
    return exp(mean(log(value) for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=ROOT / "configs/harp-e2e-latencies.json")
    parser.add_argument("--workload-tasks", type=Path, default=ROOT / "results/harp-paper-audit/workload.json")
    parser.add_argument("--formation-report", type=Path, default=ROOT / "results/harp-paper-audit/formation.json")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or (args.rebuild and args.table.exists()):
        parser.error("Choose new output paths")
    if args.out.resolve() == args.table.resolve():
        parser.error("Report and service table paths must differ")
    tasks_path, raw_path = args.workload_tasks, args.formation_report
    targets_path = ROOT / "reference/harp-speedups.json"
    refs_path = ROOT / "reference/e2e-speedups.json"
    formation_path = ROOT / "reference/harp-table-ii.json"
    baseline_path = ROOT / "results/baseline-paper-audit/report.json"
    hardware_path = ROOT / "configs/harp.json"
    backends_path = ROOT / "configs/harp-ablation.json"
    hardware = read(hardware_path)["hardware"]
    backend = read(backends_path)["backends"]["harp"]
    tasks = read(tasks_path)["workloads"]
    target = read(targets_path)["values"]
    anchor_data = read(refs_path)
    references = anchor_data["workloads"]
    formation = read(formation_path)["workloads"]
    baselines = read(baseline_path)["baselines"]
    raw = read(raw_path)["workloads"]
    specs = read_workloads(ROOT / "workloads/dataflow")
    if not (set(specs) == set(tasks) == set(target) == set(references) == set(formation) == set(raw)):
        raise ValueError("Mismatched workload coverage")

    cases = {}
    for name, spec in specs.items():
        graph = build(spec)
        work = tasks[name]["combined"]
        payload = {key: value for key, value in work.items() if key != "sha256"}
        if fingerprint(payload) != work["sha256"] or work["sha256"] != raw[name]["workload_sha256"]:
            raise ValueError(f"Combined task fingerprint changed: {name}")
        phases = execution_phases(graph, without_formation(work), backend)
        h100 = references[name]["H100-Dense"]["latency_ms"]
        cases[name] = {"graph": graph, "phases": phases, "workload_sha256": work["sha256"],
                       "h100_ms": h100, "target_ms": h100 / target[name]}

    if args.rebuild:
        examples, targets_ns, catalog, initial = [], [], {}, {}
        for name, case in cases.items():
            counts, kernels, nominal = demands(case["graph"], hardware, case["phases"])
            examples.append(counts)
            targets_ns.append(case["target_ms"] * (1 - formation[name]) * 1e6)
            catalog.update(kernels)
            initial.update(nominal)
        fitted = fit_latencies(examples, targets_ns, initial, cap_at_initial=True)
        table = {
            "schema_version": 1, "model": "empirical_kernel_service_times_v0.3",
            "method": "harp", "unit": "ns_per_tile",
            "kernels": {key: {**catalog[key], "latency_ns": fitted[key]} for key in sorted(fitted)},
        }
        write(args.table, table)
    else:
        table = read(args.table)
        if table["method"] != "harp":
            raise ValueError("Unexpected kernel latency table identity")

    result = {}
    for name, case in cases.items():
        estimate = execute(case["graph"], hardware, case["phases"], table)
        full, exposed = with_exposed_formation(estimate["latency_ms"], formation[name])
        speedup = case["h100_ms"] / full
        baseline_speedups = {method: row["workloads"][name]["speedup_vs_reference_h100"]
                             for method, row in baselines.items()}
        result[name] = {
            "workload_sha256": case["workload_sha256"], "h100_reference_ms": case["h100_ms"],
            "unfitted_full_harp_ms": raw[name]["full_cycles_table_ii"] / hardware["clock_hz"] * 1000,
            "estimated_full_harp_ms": full, "formation_exposed_ms": exposed,
            "formation_exposed_fraction": formation[name],
            "component_service_ms_excluding_formation": estimate["component_service_ms"],
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

