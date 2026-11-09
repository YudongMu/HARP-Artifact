"""Replay the historical analytical ablation on its Matched NPU baseline."""

import argparse
from pathlib import Path
from statistics import geometric_mean

from .baselines.common import ROOT, digest, read, write


def dense_latency_ms(work, inputs):
    tokens, width = work["tokens"], work["hidden_dim"]
    layer_steps = work["layers"] * work["timesteps"]
    operations = layer_steps * (12.0 * tokens * width * width + 4.0 * tokens * tokens * width)
    compute_ms = operations / (inputs["dense_peak_tflops"] * 1e12 * 0.90) * 1000.0
    activation_bytes = tokens * width * layer_steps * 2.0 * 4.4
    memory_ms = activation_bytes / (inputs["hbm_bandwidth_tbps"] * 1e12) * 1000.0
    return max(compute_ms, memory_ms) * 1.025


def reconstruct(inputs, e2e):
    if set(inputs["workloads"]) != set(e2e["workloads"]):
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
        latency["all"] = e2e["workloads"][name]["estimated_full_harp_ms"]
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
    args = parser.parse_args()
    inputs = ROOT / "workloads/harp/matched-npu.json"
    frozen_e2e = ROOT / "results/harp-paper-audit/e2e.json"
    report = reconstruct(read(inputs), read(frozen_e2e))
    report["input_sha256"] = {path.name: digest(path) for path in (inputs, frozen_e2e)}
    write(args.out, report)


if __name__ == "__main__":
    main()
