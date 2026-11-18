"""Run component timing sensitivities with configured work and resource rates."""

import argparse
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
from statistics import geometric_mean

from .baselines.common import PERFORMANCE_INPUTS, ROOT, digest, read, write
from .baselines.component_compare import BASELINES
from .baselines.components.common import COMPONENT_CONFIGS, model_inputs, workload_profile
from .component_timing import run_graph
from .dataflow import build, read_workloads
from .harp_ablation import run
from .harp_workload import compile_workload, load_profiles
from .matched_npu_ablation import reconstruct as matched_npu_timing


def timing_metrics(outcome, config, h100_ms):
    """Report the schedule and two resource lower bounds on its latency."""
    clock_hz = config["clock_hz"]
    resource_ms = {engine: cycles / clock_hz * 1000
                   for engine, cycles in outcome["resource_cycles"].items()}
    # This bound allows arbitrary overlap within each layer but preserves layer order.
    layer_cycles = sum(max(row["resource_cycles"].values(), default=0) * row["repetitions"]
                       for row in outcome["layers"])
    trajectory_cycles = max(outcome["resource_cycles"].values(), default=0)
    if not 0 < trajectory_cycles <= layer_cycles <= outcome["cycles"]:
        raise ValueError("Resource lower bounds do not agree with the schedule")
    scheduled_ms = outcome["seconds"] * 1000
    return {
        "cycles": outcome["cycles"],
        "scheduled_ms": scheduled_ms,
        "layer_overlap_bound_ms": layer_cycles / clock_hz * 1000,
        "trajectory_overlap_bound_ms": trajectory_cycles / clock_hz * 1000,
        "hbm_bytes": outcome["hbm_bytes"],
        "resource_busy_ms": resource_ms,
        "bottleneck_resource": max(resource_ms, key=resource_ms.get),
        "speedup_vs_configured_h100": h100_ms / scheduled_ms,
        "speedup_ceiling_vs_configured_h100": h100_ms / (trajectory_cycles / clock_hz * 1000),
    }


def at_pe_rate(config, pe_rate):
    return {**config, "rates": {**config["rates"], "pe": pe_rate}}


def aggregate(rows):
    return {
        "workloads": rows,
        "geomean_speedup_vs_configured_h100": geometric_mean(
            row["speedup_vs_configured_h100"] for row in rows.values()),
        "geomean_speedup_ceiling_vs_configured_h100": geometric_mean(
            row["speedup_ceiling_vs_configured_h100"] for row in rows.values()),
    }


def component_methods(report):
    return {
        "harp": report["harp"]["current"]["workloads"],
        **{name: cases["current"]["workloads"] for name, cases in report["baselines"].items()},
    }


def scaled_timing(row, scale, clock_hz):
    if not isfinite(scale) or scale <= 0:
        raise ValueError("Cycle scale must be positive and finite")
    cycles = ceil(row["cycles"] * scale)
    return {"effective_cycles": cycles, "latency_ms": cycles / clock_hz * 1000}


def load_cycle_scales(methods, parameters_path):
    if parameters_path is not None:
        parameters = read(parameters_path)
        scales = parameters["cycle_scale"]
        if parameters["schema_version"] != 1 or set(scales) != set(methods):
            raise ValueError("Timing parameters and simulator coverage differ")
        return scales, digest(parameters_path)

    scales = {method: read(COMPONENT_CONFIGS / f"{method}.json")["cycle_scale"]
              for method in methods}
    # Keep the v1 parameter fingerprint comparable with archived evaluations.
    parameters = {"schema_version": 1, "cycle_scale": scales}
    encoded = json.dumps(parameters, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return scales, sha256(encoded.encode("utf-8")).hexdigest()


def fixed_parameter_timing(report, inputs, parameters_path, harp_clock, baseline_clock):
    methods = component_methods(report)
    scales, parameters_sha256 = load_cycle_scales(methods, parameters_path)
    results = {}
    for method, rows in methods.items():
        clock = harp_clock if method == "harp" else baseline_clock
        estimates = {}
        for name, row in rows.items():
            estimate = scaled_timing(row, scales[method], clock)
            estimates[name] = {
                **estimate,
                "speedup_vs_configured_h100": inputs[name]["h100_reference_ms"] / estimate["latency_ms"],
            }
        results[method] = {
            "workloads": estimates,
            "geomean_speedup_vs_configured_h100": geometric_mean(
                row["speedup_vs_configured_h100"] for row in estimates.values()),
        }
    harp_ms = {name: row["latency_ms"] for name, row in results["harp"]["workloads"].items()}
    matched = matched_npu_timing(read(ROOT / "workloads/harp/matched-npu.json"), harp_ms)
    return {"parameters_sha256": parameters_sha256, "methods": results,
            "matched_npu": matched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--fixed", action="store_true",
                        help="Replay the six updated profiles with the bundled timing parameters")
    timing.add_argument("--timing-parameters", type=Path,
                        help="Apply a timing parameter file to the original study profiles")
    args = parser.parse_args()
    inputs = read(PERFORMANCE_INPUTS)["workloads"]
    specs = read_workloads(ROOT / "workloads/dataflow")
    graphs = {name: build(spec) for name, spec in specs.items()}
    profile_dir = "component-profile-update" if args.fixed else "revised"
    profiles, profile_sha256 = load_profiles(ROOT / "workloads/harp" / profile_dir, set(specs))
    hardware_path = ROOT / "configs/harp.json"
    backends_path = ROOT / "configs/harp-ablation.json"
    legacy_path = ROOT / "workloads/harp/matched-npu.json"
    hardware = read(hardware_path)["hardware"]
    backends = read(backends_path)["backends"]
    current_issue = backends["harp"]["sparse_issue_utilization"]
    current_pe = hardware["rates"]["pe"]
    legacy_tflops = read(legacy_path)["dense_peak_tflops"]
    legacy_pe = round(legacy_tflops * 1e12 / (2 * hardware["clock_hz"]))
    scenarios = {
        "current": (None, current_issue, current_pe),
        "full_sparse_issue": (None, 1.0, current_pe),
        "half_active_density": (0.03125, current_issue, current_pe),
        "double_active_density": (0.125, current_issue, current_pe),
        "full_issue_half_density": (0.03125, 1.0, current_pe),
        "legacy_peak": (None, current_issue, legacy_pe),
    }
    results = {}
    for scenario, (active_density, utilization, pe_rate) in scenarios.items():
        config = at_pe_rate(hardware, pe_rate)
        backend = {**backends["harp"], "sparse_issue_utilization": utilization}
        rows = {}
        for name, graph in graphs.items():
            profile = dict(profiles[name])
            if active_density is not None:
                profile["active_component_fraction"] = active_density
            tasks = compile_workload(graph, profile, "combined")
            outcome = run(graph, tasks, config, backend)
            h100_ms = inputs[name]["h100_reference_ms"]
            dense = run(graph, compile_workload(graph, profile, "dense"), config, backends["generic"])
            dense_cycles = sum(cycles for stage, cycles in outcome["component_cycles"].items()
                               if stage == "Matrix" or stage.endswith(("_Dense_BF16", "_dense_fallback")))
            rows[name] = {
                **timing_metrics(outcome, config, h100_ms),
                "dense_bf16_busy_ms": dense_cycles / config["clock_hz"] * 1000,
                "workload_sha256": tasks["sha256"],
                "h100_reference_ms": h100_ms,
                "modeled_dense_ms": dense["seconds"] * 1000,
                "speedup_vs_modeled_dense": dense["cycles"] / outcome["cycles"],
            }
        results[scenario] = {
            **aggregate(rows),
            "parameters": {"pe_issues_per_cycle": pe_rate, "sparse_issue_utilization": utilization,
                           "active_component_fraction_override": active_density},
            "geomean_speedup_vs_modeled_dense": geometric_mean(
                row["speedup_vs_modeled_dense"] for row in rows.values()),
        }

    baselines = {}
    for method, module in BASELINES.items():
        profile, _, config, _ = model_inputs(method, normalized=True)
        baselines[method] = {}
        for label, peak in (("current", config["rates"]["pe"]), ("legacy_peak", legacy_pe)):
            resources = at_pe_rate(config, peak)
            rows = {}
            for name, graph in graphs.items():
                phases = module.phases(graph, workload_profile(profile, name))
                outcome = run_graph(graph, resources, phases)
                rows[name] = timing_metrics(outcome, resources, inputs[name]["h100_reference_ms"])
            baselines[method][label] = aggregate(rows)

    report = {
        "schema_version": 1,
        "model": "component_timing_sensitivity",
        "scope": "Unfitted component estimates and optimistic resource bounds",
        "algorithm_profiles_sha256": profile_sha256,
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest(path)
                          for path in (PERFORMANCE_INPUTS, hardware_path, backends_path, legacy_path)},
        "harp": results,
        "baselines": baselines,
    }
    if args.fixed or args.timing_parameters is not None:
        baseline_clock = read(ROOT / "configs/baseline-comparison.json")["normalized"]["clock_hz"]
        fixed = fixed_parameter_timing(report, inputs, args.timing_parameters,
                                       hardware["clock_hz"], baseline_clock)
        report = {"schema_version": 1, "model": "fixed_component_timing",
                  "algorithm_profiles_sha256": profile_sha256,
                  "fixed_parameter_timing": fixed}
    write(args.out, report)


if __name__ == "__main__":
    main()
