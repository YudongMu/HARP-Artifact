"""Replay the historical analytical ablation on its Matched NPU baseline."""

import argparse
from pathlib import Path
from statistics import geometric_mean

from .baselines.common import PERFORMANCE_INPUTS, ROOT, digest, read, write
from .harp_latency import BACKENDS, HARDWARE, SERVICE_TABLE, estimate, load_cases


def dense_latency_ms(work, inputs):
    tokens, width = work["tokens"], work["hidden_dim"]
    layer_steps = work["layers"] * work["timesteps"]
    operations = layer_steps * (12.0 * tokens * width * width + 4.0 * tokens * tokens * width)
    compute_ms = operations / (inputs["dense_peak_tflops"] * 1e12 * 0.90) * 1000.0
    activation_bytes = tokens * width * layer_steps * 2.0 * 4.4
    memory_ms = activation_bytes / (inputs["hbm_bandwidth_tbps"] * 1e12) * 1000.0
    return max(compute_ms, memory_ms) * 1.025


def reconstruct(inputs, full_harp_ms):
    if set(inputs["workloads"]) != set(full_harp_ms):
        raise ValueError("Matched NPU and end-to-end workload coverage differ")
    results = {}
    for name, work in inputs["workloads"].items():
        dense = dense_latency_ms(work, inputs)
        hbm = work["hbm_pressure"]
        # Historical compute, memory, and control costs in units of Dense time.
        normalized = {
            "residual": work["rawa_work_fraction"] / 1.15 + (0.065 + 0.035 * hbm) + 0.035,
            "planeclear": (
                work["plane_work_fraction"] / 2.20
                + (0.055 + 0.025 * hbm)
                + (0.025 + 0.012 * work["fallback_rate"])
            ),
            "budget": (
                work["joint_work_fraction"] / 2.10
                + (0.045 + 0.025 * hbm)
                + (0.045 + 0.02 * work["operator_irregularity"])
            ),
        }
        latency = {mode: dense * factor for mode, factor in normalized.items()}
        latency["all"] = full_harp_ms[name]
        results[name] = {
            "matched_npu_dense_ms": dense,
            "latency_ms": latency,
            "speedup": {mode: dense / value for mode, value in latency.items()},
        }
    return {
        "schema_version": 1,
        "model": "historical_analytical_matched_npu",
        "workloads": results,
        "geomean_speedup": {
            mode: geometric_mean(row["speedup"][mode] for row in results.values())
            for mode in ("residual", "planeclear", "budget", "all")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=SERVICE_TABLE)
    parser.add_argument("--workload-tasks", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")
    inputs = ROOT / "workloads/harp/matched-npu.json"
    cases, hardware = load_cases(args.workload_tasks)
    timing = estimate(cases, hardware, read(args.table))
    report = reconstruct(read(inputs), {name: row["estimated_full_harp_ms"] for name, row in timing.items()})
    report["input_sha256"] = {path.relative_to(ROOT).as_posix(): digest(path)
                              for path in (inputs, HARDWARE, BACKENDS, PERFORMANCE_INPUTS)}
    report["service_table_sha256"] = digest(args.table)
    report["workload_sha256"] = {name: row["workload_sha256"] for name, row in timing.items()}
    write(args.out, report)


if __name__ == "__main__":
    main()
