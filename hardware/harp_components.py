"""Restored HARP component and algorithm parameter model."""

import argparse
from math import ceil, exp, log
from pathlib import Path
from statistics import mean, median

from .baselines.common import ROOT, digest, read, write
from .component_timing import Plan, Stage, dense_plan, run_graph
from .dataflow import build, read_workloads
from .trace import validate_trace


LINEARS = {"qkv", "attention_out", "ffn_in", "ffn_out", "packed_in", "packed_out"}
MODES = ("dense", "residual", "planeclear", "combined")
OBSERVED_SITE = {"qkv": "qkv", "attention_out": "out", "ffn_in": "ffn1", "ffn_out": "ffn2"}


def attention_pairs(dense, selected_fraction, phase):
    if not 0 <= selected_fraction <= 1:
        raise ValueError("Selected fraction outside [0, 1]")
    if phase == "plain":
        return dense, 0
    selected = ceil(dense * selected_fraction)
    if phase == "refresh":
        return dense, selected
    if phase == "steady":
        return 0, selected
    raise ValueError(f"Unknown phase {phase}")


def issue_split(work, fallback, active_ratio):
    if not 0 <= fallback <= 1 or active_ratio < 0:
        raise ValueError("Invalid PlaneClear issue parameters")
    eligible = round(work * (1 - fallback))
    return work - eligible, ceil(eligible * active_ratio)


def qk_issue_split(pairs, heads, head_width, fallback, retained_planes, slice_channels,
                   *, key_retained_planes=None):
    """A sparse QK issue counts one numeric plane pair on one channel Slice."""
    key_planes = retained_planes if key_retained_planes is None else key_retained_planes
    if not 0 <= fallback <= 1 or min(retained_planes, key_planes) < 0 or slice_channels <= 0:
        raise ValueError("Invalid QK plane-pair parameters")
    eligible = (1 - fallback)**2
    dense = pairs * heads * head_width
    sparse = pairs * heads * ceil(head_width / slice_channels) * (retained_planes * key_planes)
    return dense - round(dense * eligible), ceil(sparse * eligible)


def phase_counts(graph, interval, mode):
    if interval < 1 or mode not in MODES:
        raise ValueError("Invalid HARP refresh interval or mode")
    if mode not in {"residual", "combined"}:
        return [("plain", graph["forward_calls"])]
    refresh = ceil(graph["steps"] / interval) * graph["forwards_per_step"]
    return [("refresh", refresh), ("steady", graph["forward_calls"] - refresh)]


def sparse_stages(site, dense, plane, profile, balanced, kind):
    utilization = profile["issue_utilization_balanced" if balanced else "issue_utilization_unsplit"]
    if not 0 < utilization <= 1:
        raise ValueError("Invalid sustained issue utilization")
    return (
        Stage(site + "_dense_fallback", "pe", dense),
        Stage(site + "_" + kind, "pe", ceil(plane / utilization)),
    )


def retained_planes(profile, natural):
    skipped = profile["natural_skip_fraction"] if natural else profile["final_skip_fraction"]
    if not 0 <= skipped <= 1:
        raise ValueError("Invalid skipped plane fraction")
    return profile["numeric_planes"] * (1 - skipped)


def plane_site(profile, site):
    overrides = profile.get("plane_sites", {}).get(site)
    return {**profile, **overrides} if overrides else profile


def matrix_stages(work, site, profile, *, sparse, natural, balanced, counter=None):
    if not sparse:
        return (Stage(site + "_Dense_BF16", "pe", work),)
    fallback = profile["dense_fallback_fraction"]
    if counter is None:
        active = retained_planes(profile, natural) * profile["active_component_fraction"]
    else:
        fallback, active, _formed = counter
    dense, plane = issue_split(work, fallback, active)
    return sparse_stages(site, dense, plane, profile, balanced, "Plane_BF16")


def attention_plan(node, layer, lanes, profile, mode, phase, balanced, observed):
    attrs = node["attributes"]
    width = attrs["heads"] * attrs["head_width"]
    out_values = attrs["query_tokens"] * width * lanes
    full, selected = attention_pairs(attrs["pair_elements"], profile["selected_fraction"], phase)
    residual = mode in {"residual", "combined"}
    plane = mode in {"planeclear", "combined"} and not observed
    geometry = layer["region_geometry"]
    serial, parallel, join = [], [], []

    def pass_work(name, pairs, natural=False):
        work = pairs * width * lanes
        if plane:
            slices = (3 * geometry["query_regions"] * attrs["heads"] *
                      ceil(attrs["head_width"] / profile["slice_channels"]) * lanes)
            serial.append(Stage(name + "_PlaneClear_formation", "plane_formation", slices))
            if balanced:
                serial.append(Stage(name + "_array_local_packet_dispatch", "packet_dispatch", slices))
        if plane:
            dense_qk, plane_qk = qk_issue_split(
                pairs * lanes, attrs["heads"], attrs["head_width"], profile["dense_fallback_fraction"],
                retained_planes(plane_site(profile, "q"), natural), profile["slice_channels"],
                key_retained_planes=retained_planes(plane_site(profile, "k"), natural),
            )
            serial.extend(sparse_stages(name + "_QK", dense_qk, plane_qk, profile, balanced, "Plane_Plane"))
        else:
            serial.append(Stage(name + "_QK_Dense_BF16", "pe", work))
        serial.append(Stage(name + "_online_softmax", "vector", 5 * pairs * attrs["heads"] * lanes))
        serial.extend(matrix_stages(work, name + "_PV", plane_site(profile, "v"), sparse=plane,
                                    natural=natural, balanced=balanced))
        if plane:
            serial.append(Stage(name + "_shift_reduction", "shift_reduce", out_values))

    if full:
        pass_work("HARP_refresh_full" if phase == "refresh" else "HARP_full_attention",
                  full, natural=phase == "refresh")
    if phase == "refresh":
        serial.append(Stage("HARP_region_selection", "region_control",
                            geometry["region_pairs"] * lanes))
    if selected:
        key_tokens = ceil(geometry["key_tokens_per_query_region_sum"] * profile["selected_fraction"])
        multicast_bytes = key_tokens * lanes * width * 4
        serial.append(Stage("HARP_selected_KV_multicast", "multicast", multicast_bytes))
        pass_work("HARP_refresh_replay" if phase == "refresh" else "HARP_selected_attention",
                  selected, natural=phase == "refresh")
    if residual and phase == "refresh":
        join.extend((Stage("HARP_residual_form", "vector", out_values),
                     Stage("HARP_residual_write", "hbm", 2 * out_values)))
    if residual and phase == "steady":
        parallel.append(Stage("HARP_residual_prefetch", "hbm", 2 * out_values))
        join.append(Stage("HARP_residual_add", "vector", out_values))
    return Plan(tuple(serial), tuple(parallel), tuple(join))


def build_plan(workload_profile, mode, phase, balanced, observed_planes=None):
    def plan(node, layer, lanes):
        overrides = workload_profile.get("layers", {}).get(layer["id"], {})
        profile = {**workload_profile, **overrides} if overrides else workload_profile
        role = node["role"]
        if role == "self_attention":
            # The PV graph node schedules the complete attention micro-pipeline.
            return (attention_plan(node, layer, lanes, profile, mode, phase, balanced,
                                   observed_planes is not None)
                    if node["kind"] == "pv" else Plan())
        sparse = mode in {"planeclear", "combined"} and role in LINEARS
        if not sparse:
            return dense_plan(node, layer, lanes)
        counter = None
        if observed_planes is not None:
            counter = observed_planes.get(OBSERVED_SITE.get(role))
            if counter is None:
                return dense_plan(node, layer, lanes)
        values = layer["tensors"][node["inputs"][0]]["bytes"] * lanes // 2
        slices = ceil(values / profile["slice_values"] * (counter[2] if counter else 1))
        formation = (Stage("HARP_PlaneClear_formation", "plane_formation", slices),)
        dispatch = ((Stage("HARP_array_local_packet_dispatch", "packet_dispatch", slices),)
                    if balanced else ())
        site = {"packed_in": "qkv", "packed_out": "attention_out"}.get(role, role)
        issues = matrix_stages(node["macs"] * lanes, "HARP_linear", plane_site(profile, site),
                              sparse=True, natural=phase == "refresh" and role in
                              {"qkv", "attention_out", "packed_in", "packed_out"},
                              balanced=balanced, counter=counter)
        output = layer["tensors"][node["output"]]["bytes"] * lanes // 2
        return Plan(formation + dispatch + issues +
                    (Stage("HARP_shift_reduction", "shift_reduce", output),))
    return plan


def run_mode(graph, hardware, profile, mode, *, balanced=True, observed_planes=None):
    phases = [(phase, count, build_plan(profile, mode, phase, balanced, observed_planes))
              for phase, count in phase_counts(graph, profile["refresh_interval"], mode)]
    return run_graph(graph, hardware, phases)


def observed_modes(graph, hardware, profile, traces):
    modes = {}
    for mode in MODES:
        rows = []
        for trace in traces:
            if trace["mode"] != mode:
                continue
            settings = profile.copy()
            if residual := trace.get("residual"):
                tokens = graph["bundles"][0]["layers"][0]["tokens"]
                tile = residual["tile_tokens"]
                calls_per_step = (graph["forwards_per_step"] * graph["lanes_per_forward"] *
                                  sum(b["repeat"] * len(b["layers"]) for b in graph["bundles"]))
                refresh_calls = ceil(graph["steps"] / settings["refresh_interval"]) * calls_per_step
                if (tile != settings["region_tokens"] or residual["total_calls"] != calls_per_step * graph["steps"]
                        or residual["refresh_calls"] != refresh_calls):
                    raise ValueError("Observed and reconstructed call geometry differs")
                width = graph["bundles"][0]["layers"][0]["tensors"]["latent"]["shape"][1]
                per_call = tokens * width
                if (residual["read_elements"] != (residual["total_calls"] - refresh_calls) * per_call
                        or residual["write_elements"] != refresh_calls * per_call):
                    raise ValueError("Observed residual bytes differ from the graph geometry")
                pairs = residual["total_calls"] * graph["bundles"][0]["layers"][0]["region_geometry"]["region_pairs"]
                settings["selected_fraction"] = residual["selected_pairs"] / pairs
            counters = {
                site: (1 - counts["supported_slices"] / counts["opportunity_slices"],
                       counts["retained_planes"] / max(1, counts["eligible_planes"]),
                       counts["slices"] / counts["opportunity_slices"])
                for site, counts in trace.get("planes", {}).items()
            }
            result = run_mode(graph, hardware, settings, mode, observed_planes=counters)
            rows.append({"case": trace["item"]["id"], "cycles": result["cycles"],
                         "hbm_bytes": result["hbm_bytes"],
                         "selected_fraction": settings["selected_fraction"] if residual else None})
        if not rows:
            raise ValueError(f"No observed cases for {mode}")
        modes[mode] = {"cases": rows, "cycles_median": median(row["cycles"] for row in rows),
                       "hbm_bytes_median": median(row["hbm_bytes"] for row in rows)}
    return modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads/dataflow")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/harp.json")
    parser.add_argument("--observed-dit-trace", type=Path)
    args = parser.parse_args()
    specs = read_workloads(args.workloads)
    graphs = {name: build(spec) for name, spec in specs.items()}
    config = read(args.config)
    hardware, profile = config["hardware"], config["algorithm"]
    if profile["region_tokens"] != 16:
        raise ValueError("The graph currently describes 4x4 Regions")
    if not 0 <= profile["natural_skip_fraction"] <= profile["final_skip_fraction"] <= 1:
        raise ValueError("Invalid HARP skipped plane fractions")
    refs_path = ROOT / "reference/e2e-speedups.json"
    targets_path = ROOT / "reference/harp-speedups.json"
    references, targets = read(refs_path), read(targets_path)
    rows = {}
    for name, graph in graphs.items():
        h100 = references["workloads"][name]["H100-Dense"]["latency_ms"]
        settings = {**profile, **config.get("workload_overrides", {}).get(name, {})}
        result = {mode: run_mode(graph, hardware, settings, mode) for mode in MODES}
        result["combined_unsplit"] = run_mode(graph, hardware, settings, "combined", balanced=False)
        for estimate in result.values():
            estimate["speedup_vs_reference_h100"] = h100 / (1000 * estimate["seconds"])
            estimate["peak_array_only_ms"] = (1000 * estimate["resource_cycles"].get("pe", 0)
                                               / hardware["clock_hz"])
        rows[name] = {"workload_evidence": graph["evidence"], "modes": result,
                      "target_speedup": targets["values"][name],
                      "target_latency_ms": h100 / targets["values"][name],
                      "combined_vs_dense": result["dense"]["cycles"] / result["combined"]["cycles"],
                      "balanced_vs_unsplit": result["combined_unsplit"]["cycles"] /
                      result["combined"]["cycles"]}
    report = {
        "schema_version": 1, "model": "harp_graph_components_v0.2",
        "scope": "Restored HARP parameters with block-scope component timing",
        "config": config, "workloads": rows, "target_source": targets["source"],
        "combined_geomean_speedup_vs_h100": exp(mean(log(row["modes"]["combined"]["speedup_vs_reference_h100"])
                                                     for row in rows.values())),
        "source_sha256": {path.name: digest(path) for path in (
            args.config, refs_path, targets_path, Path(__file__), ROOT / "dataflow.py",
            ROOT / "component_timing.py")},
        "workload_sha256": {name: digest(args.workloads / f"{name}.json") for name in graphs},
    }
    if args.observed_dit_trace:
        dit = "dit-xl2-512"
        traces = read(args.observed_dit_trace)
        for trace in traces:
            validate_trace(trace, dit)
        report["observed_dit"] = {
            "trace_sha256": digest(args.observed_dit_trace),
            "scope": "same graph and service rates; observed linear plane counters and selected pairs; "
                     "attention plane acceleration omitted because occupancy is unavailable",
            "modes": observed_modes(graphs[dit], hardware, profile, traces),
        }
    write(args.out, report)


if __name__ == "__main__":
    main()

