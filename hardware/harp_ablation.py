"""Measure HARP workload changes and replay fixed work on alternative hardware."""

import argparse
from math import ceil, exp, log
from pathlib import Path
from statistics import mean

from .baselines.common import ROOT, digest, read, write
from .component_timing import Plan, Stage, run_graph
from .dataflow import build, read_workloads
from .harp_workload import compile_workload, fingerprint, load_profiles


MODELS = ("dense", "residual", "planeclear", "combined")


def execution_phases(graph, workload, backend):
    if fingerprint(graph) != workload["graph_sha256"]:
        raise ValueError("Compiled HARP workload and dataflow graph differ")
    utilization = backend["sparse_issue_utilization"]
    if not 0 < utilization <= 1:
        raise ValueError("Invalid sparse issue utilization")

    def execute(layers):
        def plan(node, layer, _lanes):
            original = layers[layer["id"]][node["id"]]

            def stages(group):
                result = []
                for raw in original[group]:
                    stage = Stage(**raw)
                    units = stage.units
                    if stage.name.endswith(("_Plane_Plane", "_Plane_BF16")):
                        units = ceil(units / utilization)
                    engine, cost = backend["engine_mapping"].get(stage.engine, (stage.engine, 1))
                    result.append(Stage(stage.name, engine, ceil(units * cost)))
                return tuple(result)

            serial, parallel, join = (stages(name) for name in ("serial", "parallel", "join"))
            if not backend["overlap_residual"]:
                serial, parallel = parallel + serial, ()
            return Plan(serial, parallel, join, weight_bytes=original["weight_bytes"])
        return plan

    return [(row["phase"], row["forwards"], execute(row["layers"])) for row in workload["phases"]]


def run(graph, workload, hardware, backend):
    phases = execution_phases(graph, workload, backend)
    return {**run_graph(graph, hardware, phases), "workload_sha256": workload["sha256"]}


def fraction(removed, before):
    return removed / before if before else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    task_args = parser.add_mutually_exclusive_group()
    task_args.add_argument("--tasks-out", type=Path)
    task_args.add_argument("--tasks-in", type=Path)
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads/dataflow")
    parser.add_argument("--profiles", type=Path, default=ROOT / "workloads/harp/revised")
    parser.add_argument("--reference", type=Path, default=ROOT / "reference/harp-ablation.json")
    parser.add_argument("--backends", type=Path, default=ROOT / "configs/harp-ablation.json")
    args = parser.parse_args()
    if args.out.exists() or args.tasks_out and args.tasks_out.exists():
        parser.error("Choose new output files")
    if args.tasks_out and args.out.resolve() == args.tasks_out.resolve():
        parser.error("Report and compiled task paths must differ")
    specs = read_workloads(args.workloads)
    hardware_path = ROOT / "configs/harp.json"
    backend_path = args.backends
    hardware = read(hardware_path)["hardware"]
    backend = read(backend_path)
    reference = read(args.reference)
    if set(specs) != set(reference["workloads"]) or reference["denominator"] != "Matched NPU":
        raise ValueError("Reference ablation denominator or workload coverage differs")
    replay = read(args.tasks_in) if args.tasks_in else None
    if replay is not None and (replay["schema_version"] != 1 or set(replay["workloads"]) != set(specs)):
        raise ValueError("Compiled HARP workload coverage or schema differs")
    profiles, profile_sha256 = (load_profiles(args.profiles, set(specs)) if replay is None
                                else (None, replay["profiles_sha256"]))
    results, task_output = {}, {}
    for name, spec in specs.items():
        graph = build(spec)
        if replay is None:
            work = {mode: compile_workload(graph, profiles[name], mode) for mode in MODELS}
        else:
            work = replay["workloads"][name]
            if set(work) != set(MODELS):
                raise ValueError(f"Incomplete compiled ablations for {name}")
            for mode, workload in work.items():
                payload = {key: value for key, value in workload.items() if key != "sha256"}
                if workload["mode"] != mode or fingerprint(payload) != workload["sha256"]:
                    raise ValueError(f"Invalid compiled workload fingerprint for {name}/{mode}")
        timings = {
            "dense": run(graph, work["dense"], hardware, backend["backends"]["generic"]),
            "residual": run(graph, work["residual"], hardware, backend["backends"]["generic"]),
            "planeclear": run(graph, work["planeclear"], hardware, backend["backends"]["generic"]),
            "combined_generic": run(graph, work["combined"], hardware, backend["backends"]["generic"]),
            "combined_harp": run(graph, work["combined"], hardware, backend["backends"]["harp"]),
            "combined_dedicated_serial": run(graph, work["combined"], hardware,
                                             backend["backends"]["harp_dedicated_serial"]),
            "combined_unsplit": run(graph, work["combined"], hardware, backend["backends"]["harp_unsplit"]),
        }
        dense = timings["dense"]["cycles"]
        anchors = reference["workloads"][name]
        for mode, timing in timings.items():
            timing["speedup_vs_modeled_dense"] = dense / timing["cycles"]
            if mode in anchors:
                target = anchors[mode]["speedup"]
                timing["reference_speedup_vs_matched_npu"] = target
                timing["relative_speedup_error"] = timing["speedup_vs_modeled_dense"] / target - 1
        algorithms = {}
        for mode, workload in work.items():
            counts = workload["counts"]
            algorithms[mode] = {
                "workload_sha256": workload["sha256"], "counts": counts,
                "layer_work": workload["layer_work"],
                "net_qkpv_mac_equivalent_reduction": fraction(
                    counts["qkpv_mac_equivalent_removed_net"], counts["dense_qkpv_macs"]),
                **{f"{family}_active_reduction": fraction(
                    counts[f"{family}_removed_active"], counts.get(f"{family}_before_active", 0))
                   for family in ("qk_plane_pairs", "pv_plane_components", "linear_plane_components")},
            }
        results[name] = {
            "workload_evidence": graph["evidence"], "algorithm_work": algorithms, "timings": timings,
            "hardware_gain_fixed_combined": timings["combined_generic"]["cycles"] / timings["combined_harp"]["cycles"],
            "hardware_breakdown_fixed_combined": {
                "dedicated_services": timings["combined_generic"]["cycles"] /
                timings["combined_dedicated_serial"]["cycles"],
                "residual_overlap": timings["combined_dedicated_serial"]["cycles"] /
                timings["combined_unsplit"]["cycles"],
                "sparse_issue_utilization": timings["combined_unsplit"]["cycles"] /
                timings["combined_harp"]["cycles"],
            },
            "reference_hardware_gain": anchors["combined_harp"]["speedup"] / anchors["combined_generic"]["speedup"],
        }
        if args.tasks_out:
            task_output[name] = work

    def geomean(values):
        return exp(mean(log(value) for value in values))

    report = {
        "schema_version": 1, "model": "harp_workload_ablation_v0.4",
        "modeled_denominator": "dense generic substrate proxy",
        "reference": {key: reference[key] for key in ("file", "sheet", "sha256", "denominator")},
        "reference_evidence": reference.get("evidence", "Workbook numerical anchors; provenance unverified."),
        "backend_assumptions": backend,
        "workloads": results,
        "geomean": {
            mode: {"modeled_speedup": geomean(row["timings"][mode]["speedup_vs_modeled_dense"]
                                              for row in results.values()),
                   "reference_speedup": geomean(anchors[mode]["speedup"]
                                               for anchors in reference["workloads"].values())}
            for mode in ("residual", "planeclear", "combined_generic", "combined_harp")
        },
        "fixed_work_hardware_gain_geomean": geomean(row["hardware_gain_fixed_combined"]
                                                    for row in results.values()),
        "reference_hardware_gain_geomean": geomean(row["reference_hardware_gain"] for row in results.values()),
        "hardware_breakdown_geomean": {
            stage: geomean(row["hardware_breakdown_fixed_combined"][stage] for row in results.values())
            for stage in ("dedicated_services", "residual_overlap", "sparse_issue_utilization")
        },
        "source_sha256": {"algorithm_profiles": profile_sha256, **{
            path.name: digest(path) for path in (args.reference, hardware_path, backend_path,
                                                 Path(__file__), ROOT / "harp_workload.py",
                                                 ROOT / "harp_components.py", ROOT / "component_timing.py",
                                                 ROOT / "dataflow.py")
        }},
        "workload_sha256": {name: digest(args.workloads / f"{name}.json") for name in specs},
    }
    if args.tasks_out:
        write(args.tasks_out, {"schema_version": 1, "profiles_sha256": profile_sha256, "workloads": task_output})
    tasks_path = args.tasks_in or args.tasks_out
    if tasks_path:
        report["compiled_tasks_sha256"] = digest(tasks_path)
    write(args.out, report)


if __name__ == "__main__":
    main()

