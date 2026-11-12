"""Estimate Full HARP latency for a compiled Combined workload."""

from .baselines.common import ROOT, read
from .baselines.kernel_latencies import execute
from .dataflow import build, read_workloads
from .formation_ablation import without_formation
from .harp_ablation import execution_phases
from .harp_workload import compile_workload, fingerprint, load_profiles


SERVICE_TABLE = ROOT / "configs/harp-e2e-latencies.json"
HARDWARE = ROOT / "configs/harp.json"
BACKENDS = ROOT / "configs/harp-ablation.json"
FORMATION = ROOT / "reference/harp-table-ii.json"


def with_exposed_formation(core_ms, share):
    if core_ms <= 0 or not 0 <= share < 1:
        raise ValueError("Invalid formation exposure")
    full = core_ms / (1 - share)
    return full, full - core_ms


def load_cases(tasks_path=None):
    specs = read_workloads(ROOT / "workloads/dataflow")
    formation = read(FORMATION)["workloads"]
    tasks = read(tasks_path)["workloads"] if tasks_path is not None else None
    if set(specs) != set(formation) or tasks is not None and set(specs) != set(tasks):
        raise ValueError("Mismatched HARP workload coverage")
    profiles = None
    if tasks is None:
        profiles, _ = load_profiles(ROOT / "workloads/harp/revised", set(specs))
    backend = read(BACKENDS)["backends"]["harp"]
    cases = {}
    for name, spec in specs.items():
        graph = build(spec)
        work = tasks[name]["combined"] if tasks is not None else compile_workload(
            graph, profiles[name], "combined")
        payload = {key: value for key, value in work.items() if key != "sha256"}
        if work["mode"] != "combined" or fingerprint(payload) != work["sha256"]:
            raise ValueError(f"Invalid Combined workload fingerprint: {name}")
        cases[name] = {
            "graph": graph,
            "phases": execution_phases(graph, without_formation(work), backend),
            "workload_sha256": work["sha256"],
            "formation_exposed_fraction": formation[name],
        }
    return cases, read(HARDWARE)["hardware"]


def estimate(cases, hardware, table):
    if table["method"] != "harp":
        raise ValueError("Unexpected kernel service table identity")
    result = {}
    for name, case in cases.items():
        outcome = execute(case["graph"], hardware, case["phases"], table)
        share = case["formation_exposed_fraction"]
        full, exposed = with_exposed_formation(outcome["latency_ms"], share)
        result[name] = {
            "workload_sha256": case["workload_sha256"],
            "estimated_full_harp_ms": full,
            "formation_exposed_ms": exposed,
            "formation_exposed_fraction": share,
            "component_service_ms_excluding_formation": outcome["component_service_ms"],
        }
    return result
