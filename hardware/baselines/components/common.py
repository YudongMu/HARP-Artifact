"""Helpers for baseline component phase definitions."""

import argparse
from math import ceil
from pathlib import Path

from ...component_timing import Plan, Stage, run_graph
from ...dataflow import build, read_workloads
from ..common import ROOT, digest, native_architecture, read, write


COMPONENT_CONFIGS = ROOT / "configs/baseline-components"
LINEAR_ROLES = {"qkv", "attention_out", "ffn_in", "ffn_out", "packed_in", "packed_out"}


def engine(node, lanes, name, vector_name, fraction=1):
    if node["macs"]:
        return Plan((Stage(name, "pe", ceil(node["macs"] * lanes * fraction)),))
    if node["vector_ops"]:
        return Plan((Stage(vector_name, "vector", ceil(node["vector_ops"] * lanes * fraction)),))
    return Plan()


def elements(layer, tensor, lanes):
    return layer["tensors"][tensor]["bytes"] * lanes // 2


def temporal_phases(graph, build_plan):
    calls = graph["forwards_per_step"]
    steps = graph["steps"]
    if steps == 1:
        return [("single", calls, build_plan(True, True))]
    return [
        ("initial", calls, build_plan(True, False)),
        ("steady", (steps - 2) * calls, build_plan(False, False)),
        ("final", calls, build_plan(False, True)),
    ]



def model_inputs(name, *, normalized=False):
    profile_path = ROOT / "configs" / f"{name}.json"
    timing_path = COMPONENT_CONFIGS / f"{name}.json"
    profile, timing = read(profile_path), read(timing_path)
    if normalized:
        peak = read(ROOT / "configs/baseline-comparison.json")["normalized"]
    elif name == "dstar":
        from ..dstar import ARCHITECTURE
        peak = ARCHITECTURE
    else:
        peak = native_architecture(profile)
    architecture = {
        "clock_hz": peak["clock_hz"], "buffer_bytes": timing["buffer_bytes"],
        "rates": {"pe": peak["macs_per_cycle"], "vector": peak["vector_ops_per_cycle"],
                  "hbm": peak["hbm_bytes_per_cycle"], **timing["rates"]},
    }
    return profile, timing, architecture, (profile_path, timing_path)


def workload_profile(profile, name):
    return profile["profiles"][name] if "profiles" in profile else profile["profile"]


def run_component(name, phases):
    parser = argparse.ArgumentParser(description=f"Run the {name} component timing model.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads/dataflow")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--kernel-latencies", type=Path)
    args = parser.parse_args()
    specs = read_workloads(args.workloads)
    profile, timing, architecture, inputs = model_inputs(
        name, normalized=args.normalized or args.kernel_latencies is not None)
    table = None
    if args.kernel_latencies is not None:
        from ..kernel_latencies import execute
        table = read(args.kernel_latencies)
        if table["method"] != name:
            raise ValueError("Kernel latency baseline identity mismatch")
    dependencies = (*inputs, Path(__file__), ROOT / "model.py", ROOT / "component_timing.py",
                    ROOT / "dataflow.py", ROOT / "baselines/components" / f"{name}.py")
    if args.normalized or table is not None:
        dependencies += (ROOT / "configs/baseline-comparison.json",)
    elif name == "dstar":
        dependencies += (ROOT / "baselines/dstar.py",)
    if table is not None:
        dependencies += (args.kernel_latencies, ROOT / "baselines/kernel_latencies.py")
    estimates = {}
    for key, spec in specs.items():
        graph = build(spec)
        settings = workload_profile(profile, key)
        estimate = (execute(graph, architecture, phases(graph, settings), table) if table is not None
                    else run_graph(graph, architecture, phases(graph, settings)))
        estimates[key] = {"workload_evidence": graph["evidence"], "profile": settings, **estimate}
    write(args.out, {
        "schema_version": 1, "model": name, "resources": ("empirical_kernel_latencies" if table is not None else
                      "normalized" if args.normalized else "native_peak"),
        "architecture": architecture, "cost_evidence": timing["evidence"], "workloads": estimates,
        "workload_sha256": {key: digest(args.workloads / f"{key}.json") for key in specs},
        "source_sha256": {path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name:
                          digest(path) for path in dependencies},
    })
