"""Empirical kernel service times for reconstructed baseline latency."""

from collections import Counter
from hashlib import sha256
import json
from math import ceil, fsum, isfinite

from ..component_timing import Stage, memory_placement


TILE_UNITS = {"pe": 4096, "vector": 1024, "hbm": 4096}


def predict_ns(counts, latencies):
    return fsum(count * latencies[key] for key, count in counts.items())


def fit_latencies(examples, targets_ns, initial_ns, *, tolerance=1e-9, max_passes=20000,
                  cap_at_initial=False):
    """Fit positive per-kernel tile times against total trajectory durations."""
    if not examples or len(examples) != len(targets_ns):
        raise ValueError("Mismatched calibration examples")
    if any(not isfinite(value) or value <= 0 for value in targets_ns):
        raise ValueError("Target durations must be positive and finite")
    keys = sorted({key for row in examples for key in row})
    values = {key: float(initial_ns[key]) for key in keys}
    if any(not isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("Initial kernel times must be positive and finite")
    columns = {key: [row.get(key, 0) / target for row, target in zip(examples, targets_ns)]
               for key in keys}
    norms = {key: fsum(value**2 for value in column) for key, column in columns.items()}
    predicted = [predict_ns(row, values) / target for row, target in zip(examples, targets_ns)]
    for _ in range(max_passes):
        if max(abs(value - 1) for value in predicted) <= tolerance:
            return values
        for key in keys:
            column = columns[key]
            if not norms[key]:
                continue
            delta = fsum(weight * (1 - value) for weight, value in zip(column, predicted)) / norms[key]
            new = max(initial_ns[key] * 1e-9, values[key] + delta)
            if cap_at_initial:
                new = min(initial_ns[key], new)
            change = new - values[key]
            values[key] = new
            predicted = [value + weight * change for weight, value in zip(column, predicted)]
    error = max(abs(predict_ns(row, values) / target - 1)
                for row, target in zip(examples, targets_ns))
    raise ValueError(f"Kernel calibration did not converge: relative error {error:.6g}")


def kernel_spec(node, layer, lanes, stage):
    return {
        "component": stage.name, "engine": stage.engine, "operator": node["kind"],
        "role": node["role"], "lanes": lanes,
        "input_shapes": [layer["tensors"][name]["shape"] for name in node["inputs"]],
        "output_shape": layer["tensors"][node["output"]]["shape"],
        "attention_pattern": node["attributes"].get("pattern"),
        "head_width": node["attributes"].get("head_width"),
        "units_per_tile": TILE_UNITS.get(stage.engine, 1024),
    }


def demands(graph, config, phases):
    """Collect component tile counts from operator shapes and phase repetitions."""
    if sum(count for _, count, _ in phases) != graph["forward_calls"]:
        raise ValueError("Incomplete trajectory phase coverage")
    counts = Counter()
    kernels = {}
    nominal = {}
    lanes = graph["lanes_per_forward"]
    for _, forwards, plan in phases:
        if not forwards:
            continue
        for bundle in graph["bundles"]:
            repetitions = forwards * bundle["repeat"]
            for layer in bundle["layers"]:
                tensors = layer["tensors"]
                nodes = {node["output"]: node for node in layer["nodes"]}
                spilled = memory_placement(layer, lanes, config["buffer_bytes"])

                def spilled_input(name, demand):
                    tensor = tensors[name]
                    if tensor["storage"] == "view":
                        sources = nodes[name]["inputs"]
                        if len(sources) == 1:
                            return spilled_input(sources[0], demand)
                        return sum(spilled_input(source, tensors[source]["bytes"]) for source in sources)
                    return min(demand, tensor["bytes"]) if name in spilled else 0

                for node in layer["nodes"]:
                    output = node["output"]
                    if tensors[output]["storage"] == "view":
                        continue
                    service = plan(node, layer, lanes)
                    weights = node["weight_bytes"] if service.weight_bytes is None else service.weight_bytes
                    read = weights + lanes * sum(spilled_input(name, tensors[name]["bytes"])
                                                 for name in node["inputs"])
                    write = lanes * tensors[output]["bytes"] if output in spilled else 0
                    stages = (
                        Stage("HBM_operand_read", "hbm", read), *service.serial,
                        *service.parallel, *service.join,
                        Stage("HBM_output_write", "hbm", write),
                    )
                    for stage in stages:
                        if not stage.units:
                            continue
                        if stage.units < 0:
                            raise ValueError("Negative service demand")
                        signature = kernel_spec(node, layer, lanes, stage)
                        key = sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
                        kernels[key] = signature
                        tile = signature["units_per_tile"]
                        counts[key] += ceil(stage.units / tile) * repetitions
                        nominal[key] = tile / config["rates"][stage.engine] * 1e9 / config["clock_hz"]
    return dict(counts), kernels, nominal


def execute(graph, config, phases, table):
    if table["model"] != "empirical_kernel_service_times_v0.3":
        raise ValueError("Incompatible kernel latency table")
    counts, kernels, _ = demands(graph, config, phases)
    latencies = {key: row["latency_ns"] for key, row in table["kernels"].items()}
    total = predict_ns(counts, latencies)
    components = Counter()
    for key, count in counts.items():
        components[kernels[key]["component"]] += count * latencies[key]
    return {"latency_ms": total / 1e6, "component_service_ms": {
        name: value / 1e6 for name, value in sorted(components.items())},
        "kernel_count": len(counts)}

