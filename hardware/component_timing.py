"""Dependency-aware component timing over a repeated operator graph."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil

from .model import Task


@dataclass(frozen=True)
class Stage:
    name: str
    engine: str
    units: int


@dataclass(frozen=True)
class Plan:
    serial: tuple = ()
    parallel: tuple = ()
    join: tuple = ()
    weight_bytes: int | None = None


def timeline(tasks):
    """Dispatch ready work by earliest feasible start, then insertion order."""
    if len({task.name for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task identity")
    waiting = list(tasks)
    finish = {}
    free = defaultdict(int)
    busy = Counter()
    while waiting:
        ready = [(max(free[t.resource], *(finish[p] for p in t.after), 0), i)
                 for i, t in enumerate(waiting) if all(p in finish for p in t.after)]
        if not ready:
            raise ValueError("Cyclic or missing task dependency")
        start, index = min(ready)
        task = waiting.pop(index)
        finish[task.name] = free[task.resource] = start + task.cycles
        busy[task.resource] += task.cycles
    return {"cycles": max(finish.values(), default=0), "resource_cycles": dict(busy)}


def dense_plan(node, _layer, lanes):
    if node["macs"]:
        return Plan((Stage("Matrix", "pe", node["macs"] * lanes),))
    if node["vector_ops"]:
        return Plan((Stage(node["kind"], "vector", node["vector_ops"] * lanes),))
    return Plan()


def memory_placement(layer, lanes, capacity):
    """Allocate complete live tensors when they fit, otherwise stream from HBM."""
    tensors = layer["tensors"]
    end = {name: tensor["lifetime"][1] for name, tensor in tensors.items()}
    for node in reversed(layer["nodes"]):
        if tensors[node["output"]]["storage"] == "view":
            for source in node["inputs"]:
                end[source] = max(end[source], end[node["output"]])
    live = {}
    spilled = {name for name, tensor in tensors.items() if tensor["producer"] is None}
    for index, node in enumerate(layer["nodes"]):
        live = {name: size for name, size in live.items() if end[name] >= index}
        output = node["output"]
        tensor = tensors[output]
        if tensor["storage"] in {"view", "attention_stream"}:
            continue
        size = tensor["bytes"] * lanes
        if not tensor["consumers"] or sum(live.values()) + size > capacity:
            spilled.add(output)
        else:
            live[output] = size
    return spilled


def run_layer(layer, graph, rates, buffer_bytes, plan=dense_plan):
    lanes = graph["lanes_per_forward"]
    tensors = layer["tensors"]
    by_output = {node["output"]: node for node in layer["nodes"]}
    spilled = memory_placement(layer, lanes, buffer_bytes)
    tasks = []
    terminals = {}
    components = Counter()
    units = Counter()
    hbm_bytes = 0

    def spilled_input(name, demand):
        tensor = tensors[name]
        if tensor["storage"] == "view":
            sources = by_output[name]["inputs"]
            if len(sources) == 1:
                return spilled_input(sources[0], demand)
            return sum(spilled_input(source, tensors[source]["bytes"]) for source in sources)
        return min(demand, tensor["bytes"]) if name in spilled else 0

    for node in layer["nodes"]:
        output = node["output"]
        dependencies = tuple(dict.fromkeys(dep for name in node["inputs"] for dep in terminals.get(name, ())))
        if tensors[output]["storage"] == "view":
            terminals[output] = dependencies
            continue
        stages = plan(node, layer, lanes)
        weights = node["weight_bytes"] if stages.weight_bytes is None else stages.weight_bytes
        read = weights + lanes * sum(spilled_input(name, tensors[name]["bytes"]) for name in node["inputs"])
        write = lanes * tensors[output]["bytes"] if output in spilled else 0

        def chain(sequence, after):
            nonlocal hbm_bytes
            for stage in sequence:
                if stage.units < 0 or rates[stage.engine] <= 0:
                    raise ValueError(f"Invalid service cost {stage}")
                if stage.units:
                    cycles = ceil(stage.units / rates[stage.engine])
                    name = f"{node['id']}:{len(tasks)}"
                    tasks.append(Task(name, stage.engine, cycles, after))
                    components[stage.name] += cycles
                    units[stage.name] += stage.units
                    if stage.engine == "hbm":
                        hbm_bytes += stage.units
                    after = (name,)
            return after

        loaded = chain((Stage("HBM_operand_read", "hbm", read),), dependencies)
        work = chain(stages.serial, loaded)
        parallel = chain(stages.parallel, dependencies) if stages.parallel else ()
        joined = chain(stages.join, tuple(dict.fromkeys(work + parallel)))
        terminals[output] = chain((Stage("HBM_output_write", "hbm", write),), joined)
    return {**timeline(tasks), "hbm_bytes": hbm_bytes,
            "component_cycles": dict(components), "component_work": dict(units)}


def run_graph(graph, config, phases):
    """Run repeated layer bundles at each phase's explicit forward count."""
    if sum(forwards for _, forwards, _ in phases) != graph["forward_calls"]:
        raise ValueError("Phase counts do not cover every Transformer call")
    resources, components, work = Counter(), Counter(), Counter()
    cycles = hbm_bytes = 0
    layer_results = []
    for label, forwards, plan in phases:
        if not forwards:
            continue
        for bundle in graph["bundles"]:
            count = bundle["repeat"] * forwards
            for layer in bundle["layers"]:
                result = run_layer(layer, graph, config["rates"], config["buffer_bytes"], plan)
                cycles += result["cycles"] * count
                hbm_bytes += result["hbm_bytes"] * count
                for aggregate, field in ((resources, "resource_cycles"), (components, "component_cycles"),
                                         (work, "component_work")):
                    for key, value in result[field].items():
                        aggregate[key] += value * count
                layer_results.append({"phase": label, "layer": layer["id"], "repetitions": count, **result})
    return {"cycles": cycles, "seconds": cycles / config["clock_hz"], "hbm_bytes": hbm_bytes,
            "resource_cycles": dict(resources), "component_cycles": dict(components),
            "component_work": dict(work), "layers": layer_results}

