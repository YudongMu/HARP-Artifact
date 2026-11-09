"""Translate immutable DiT quality receipts into hardware workload counters."""

from hashlib import sha256
import json


SITES = {
    "qkv_input": ("qkv", 1),
    "o_proj_input": ("out", 1),
    "ffn1_input": ("ffn1", 1),
    "ffn2_input": ("ffn2", 4),
}


MODES = {"dense", "residual", "planeclear", "combined"}
DEPLOYMENT_KEYS = (
    "checkpoint_sha256", "noise_schedule_sha256", "sampler", "cfg_scale", "layout", "steps",
)


def validate_trace(trace, workload_id):
    mode = trace["mode"]
    if (trace["schema_version"] != 1 or trace["workload_id"] != workload_id
            or trace["granularity"] != "trajectory_aggregate" or mode not in MODES):
        raise ValueError("Incompatible trace identity or granularity")
    for field, enabled in (("residual", mode in {"residual", "combined"}),
                           ("planes", mode in {"planeclear", "combined"})):
        if enabled != bool(trace.get(field)):
            raise ValueError(f"Mode {mode} has inconsistent {field} counters")
    if residual := trace.get("residual"):
        if (residual["element_bytes"] != 2 or residual["tile_tokens"] <= 0
                or any(type(count) is not int or count < 0 for count in residual.values())):
            raise ValueError("Invalid BF16 residual counters")
    if trace.get("planes") and not trace.get("table_sha256"):
        raise ValueError("PlaneClear trace lacks table identity")


def import_receipt(receipt, workload):
    identity = receipt["identity"]
    mode = identity["mode"]
    counters = receipt["runtime_counters"]
    if (workload["id"] != "dit-xl2-512" or mode not in MODES
            or identity["steps"] != workload["steps"] or identity["resolution"] not in (512, [512, 512])
            or identity["batch_size"] != 1):
        raise ValueError("Unsupported receipt deployment")
    trace = {"schema_version": 1, "workload_id": workload["id"],
             "granularity": "trajectory_aggregate", "mode": mode,
             "item": receipt["item"], "noise_sha256": receipt["noise_sha256"],
             "execution_sha256": receipt["execution_sha256"],
             "table_sha256": receipt.get("source_table_sha256"),
             "deployment": {key: identity.get(key) for key in DEPLOYMENT_KEYS}}
    steps, lanes, n, width = (workload[k] for k in ("steps", "lanes", "tokens", "width"))
    layers = sum(group["count"] for group in workload["blocks"])
    calls = steps * lanes * layers
    if mode in {"residual", "combined"}:
        tile = 16
        interval = identity["refresh_interval"]
        attention = counters["attention_totals"]
        if (interval <= 0 or n % tile or counters["attention_calls"] * lanes != calls
                or attention["candidate_region_pairs"] != calls * (n // tile)**2):
            raise ValueError("Region pairs do not match the workload")
        refresh = ((steps + interval - 1) // interval) * lanes * layers
        trace["residual"] = {
            "tile_tokens": tile, "total_calls": calls, "refresh_calls": refresh,
            "selected_pairs": attention["selected_region_pairs"],
            "read_elements": attention["residual_elements_read"],
            "write_elements": attention["residual_elements_written"], "element_bytes": 2,
        }
    if mode in {"planeclear", "combined"}:
        planes = trace["planes"] = {}
        omitted = trace["unmapped_attention_plane_slices"] = {}
        fields = ("slices", "supported_slices", "retained_planes", "eligible_planes")
        for key, value in counters["plane_by_site"].items():
            layer, site = key.split(":")
            if not 0 <= int(layer) < layers:
                raise ValueError("Plane counter has invalid layer")
            if site not in SITES:
                if site not in {"q", "k", "v"}:
                    raise ValueError(f"Unknown plane site {site}")
                omitted[site] = omitted.get(site, 0) + value["slices"]
                continue
            operator, scale = SITES[site]
            row = planes.setdefault(operator, {field: 0 for field in fields})
            row["opportunity_slices"] = calls * (n // 16) * (width // 32) * scale
            for field in fields:
                row[field] += value[field]
        if not planes or not trace["table_sha256"]:
            raise ValueError("Missing PlaneClear evidence")
    validate_trace(trace, workload["id"])
    return trace


def import_case_tree(root, workload):
    """Load four-mode matched receipts, preserving source file fingerprints."""
    groups = {"dense": "dense", "residual": "residual", "pc": "planeclear", "combined": "combined"}
    case_ids = sorted(path.parent.name for path in (root / "dense" / "cases").glob("*/sample.json"))
    if not case_ids:
        raise ValueError("No dense receipts")
    for group in groups:
        ids = sorted(path.parent.name for path in (root / group / "cases").glob("*/sample.json"))
        if ids != case_ids:
            raise ValueError(f"Receipt coverage mismatch for {group}")
    traces = []
    for case_id in case_ids:
        pairs = set()
        identities = set()
        for group, mode in groups.items():
            path = root / group / "cases" / case_id / "sample.json"
            data = path.read_bytes()
            receipt = json.loads(data)
            trace = import_receipt(receipt, workload)
            if trace["mode"] != mode or trace["item"]["id"] != case_id:
                raise ValueError(f"Receipt identity mismatch at {path}")
            pairs.add((json.dumps(trace["item"], sort_keys=True), trace["noise_sha256"]))
            identities.add(json.dumps(trace["deployment"], sort_keys=True))
            trace["source"] = {"path": path.relative_to(root).as_posix(), "sha256": sha256(data).hexdigest()}
            traces.append(trace)
        if len(pairs) != 1 or len(identities) != 1:
            raise ValueError(f"Unpaired receipts for {case_id}")
    return traces




