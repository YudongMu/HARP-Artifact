"""Compile hardware-independent HARP work and count algorithmic eliminations."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json

from .baselines.common import digest, read
from .harp_components import attention_pairs, build_plan, phase_counts


PLANE_FIELDS = {
    "qk_plane_pairs": "_Plane_Plane",
    "pv_plane_components": "_PV_Plane_BF16",
    "linear_plane_components": "HARP_linear_Plane_BF16",
}


def fingerprint(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_profiles(path, workloads):
    if path.is_file():
        legacy = read(path)
        profiles = {name: {**legacy["defaults"], **legacy.get("workloads", {}).get(name, {})}
                    for name in workloads}
        return profiles, digest(path)

    defaults_path = path / "defaults.json"
    defaults = read(defaults_path)
    paths = {p.stem: p for p in path.glob("*.json") if p != defaults_path}
    if set(paths) != workloads:
        raise ValueError("HARP algorithm profile coverage differs")
    profiles = {}
    sites = {"qkv", "q", "k", "v", "attention_out", "ffn_in", "ffn_out"}
    for name, source in sorted(paths.items()):
        profile = read(source)
        if (profile["id"] != name or set(profile["plane_sites"]) != sites
                or not 0 <= profile["selected_fraction"] <= 1):
            raise ValueError(f"Invalid HARP algorithm profile: {name}")
        for rates in profile["plane_sites"].values():
            if not 0 <= rates["natural_skip_fraction"] <= rates["final_skip_fraction"] <= 1:
                raise ValueError(f"Invalid PlaneClear fractions: {name}")
        profiles[name] = {**defaults, **profile}
    return profiles, fingerprint({"defaults.json": digest(defaults_path),
                                  **{p.name: digest(p) for p in paths.values()}})


def plane_counts(stages):
    counts = Counter()
    for stage in stages:
        if stage.name.endswith("_dense_fallback"):
            counts["dense_fallback_macs"] += stage.units
        for family, suffix in PLANE_FIELDS.items():
            if stage.name.endswith(suffix):
                counts[family] += stage.units
    return counts


def compiler(profile, mode, phase, plane_baseline=None):
    settings = {**profile}
    layers = {name: {**overrides} for name, overrides in profile.get("layers", {}).items()}
    if plane_baseline == "all":
        settings.update(natural_skip_fraction=0, final_skip_fraction=0)
        for overrides in layers.values():
            overrides.update(natural_skip_fraction=0, final_skip_fraction=0)
    elif plane_baseline == "natural":
        settings["final_skip_fraction"] = settings["natural_skip_fraction"]
        for overrides in layers.values():
            overrides["final_skip_fraction"] = overrides.get("natural_skip_fraction",
                                                            settings["natural_skip_fraction"])
    for overrides in (settings, *layers.values()):
        if plane_baseline and "plane_sites" in overrides:
            overrides["plane_sites"] = {
                name: {**rates, "natural_skip_fraction": 0, "final_skip_fraction": 0}
                if plane_baseline == "all" else
                {**rates, "final_skip_fraction": rates["natural_skip_fraction"]}
                for name, rates in overrides["plane_sites"].items()
            }
        # Preserve issue counts. Backend utilization is applied only during execution.
        overrides.update(issue_utilization_balanced=1, issue_utilization_unsplit=1)
    settings["layers"] = layers
    return build_plan(settings, mode, phase, balanced=True)


def logical_attention(node, layer, lanes, selected_fraction, phase):
    attrs = node["attributes"]
    dense = attrs["pair_elements"]
    full, selected = attention_pairs(dense, selected_fraction, phase)
    dense_regions = layer["region_geometry"]["region_pairs"]
    full_regions, selected_regions = attention_pairs(dense_regions, selected_fraction, phase)
    width = attrs["heads"] * attrs["head_width"]
    counts = Counter({
        "dense_token_pairs": dense * lanes, "executed_token_pairs": (full + selected) * lanes,
        "dense_region_pairs": dense_regions * lanes,
        "executed_region_pairs": (full_regions + selected_regions) * lanes,
        "dense_qkpv_macs": 2 * dense * width * lanes,
        "executed_qkpv_mac_equivalent": 2 * (full + selected) * width * lanes,
    })
    if phase == "refresh":
        counts.update(refresh_token_pairs=full * lanes, replay_token_pairs=selected * lanes)
    elif phase == "steady":
        counts["steady_token_pairs"] = selected * lanes
    else:
        counts["plain_token_pairs"] = full * lanes
    return counts


def compile_workload(graph, profile, mode):
    lanes = graph["lanes_per_forward"]
    phases, totals, layer_work = [], Counter(), []
    for phase, forwards in phase_counts(graph, profile["refresh_interval"], mode):
        if not forwards:
            continue
        active = compiler(profile, mode, phase)
        natural = compiler(profile, mode, phase, "natural")
        all_planes = compiler(profile, mode, phase, "all")
        phase_layers = {}
        for bundle in graph["bundles"]:
            repeats = bundle["repeat"] * forwards
            for layer in bundle["layers"]:
                name = layer["id"]
                if name in phase_layers:
                    raise ValueError(f"Duplicate layer template: {name}")
                settings = {**profile, **profile.get("layers", {}).get(name, {})}
                plans, counts = {}, Counter()
                for node in layer["nodes"]:
                    compiled = active(node, layer, lanes)
                    plans[node["id"]] = asdict(compiled)
                    stages = (*compiled.serial, *compiled.parallel, *compiled.join)
                    if node["role"] == "self_attention" and node["kind"] == "pv":
                        counts.update(logical_attention(node, layer, lanes, settings["selected_fraction"], phase))
                    actual = plane_counts(stages)
                    counts["dense_fallback_macs"] += actual["dense_fallback_macs"]
                    for stage in stages:
                        if stage.name.endswith("_PlaneClear_formation"):
                            counts["formed_slices"] += stage.units
                        elif stage.name in {"HARP_residual_write", "HARP_residual_prefetch"}:
                            counts[stage.name.removeprefix("HARP_") + "_bytes"] += stage.units
                    if mode in {"planeclear", "combined"}:
                        no_active = plane_counts(natural(node, layer, lanes).serial)
                        no_skips = plane_counts(all_planes(node, layer, lanes).serial)
                        for family in PLANE_FIELDS:
                            counts[f"{family}_issued"] += actual[family]
                            counts[f"{family}_before_active"] += no_active[family]
                            counts[f"{family}_before_natural"] += no_skips[family]
                counts["token_pairs_removed_net"] = counts["dense_token_pairs"] - counts["executed_token_pairs"]
                counts["qkpv_mac_equivalent_removed_net"] = (
                    counts["dense_qkpv_macs"] - counts["executed_qkpv_mac_equivalent"])
                for family in PLANE_FIELDS:
                    counts[f"{family}_removed_active"] = (
                        counts[f"{family}_before_active"] - counts[f"{family}_issued"])
                    counts[f"{family}_removed_natural"] = (
                        counts[f"{family}_before_natural"] - counts[f"{family}_before_active"])
                expanded = {key: value * repeats for key, value in counts.items()}
                totals.update(expanded)
                layer_work.append({"layer": name, "phase": phase, "repetitions": repeats, "counts": expanded})
                phase_layers[name] = plans
        phases.append({"phase": phase, "forwards": forwards, "layers": phase_layers})
    workload = {"mode": mode, "graph_sha256": fingerprint(graph), "phases": phases,
                "counts": dict(totals), "layer_work": layer_work}
    workload["sha256"] = fingerprint(workload)
    return workload

