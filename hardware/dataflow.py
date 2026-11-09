"""Expand compact Transformer layer bundles into operator and tensor graphs."""

import argparse
from collections import Counter
from itertools import product
from math import ceil, prod
from pathlib import Path

from .baselines.common import ROOT, digest, read, write


def read_workloads(directory):
    entries = {path.stem: read(path) for path in sorted(directory.glob("*.json"))}
    if not entries or any(name != spec["id"] or spec["schema_version"] != 1
                          for name, spec in entries.items()):
        raise ValueError("Invalid workload descriptor identities")
    return entries


def window_histogram(grid, tile):
    """Count equal-volume windows, including short edge windows."""
    if len(grid) != len(tile) or min(*grid, *tile) <= 0:
        raise ValueError("Invalid window geometry")
    spans = [Counter(min(extent - start, length) for start in range(0, extent, length))
             for extent, length in zip(grid, tile)]
    result = Counter()
    for choice in product(*(axis.items() for axis in spans)):
        result[prod(size for size, _ in choice)] += prod(count for _, count in choice)
    return dict(sorted(result.items()))


class Graph:
    def __init__(self):
        self.nodes = []
        self.tensors = {}

    def tensor(self, name, shape, *, layout="token_major", storage="activation", bytes_per_element=2):
        if name in self.tensors or min(shape) <= 0:
            raise ValueError(f"Invalid tensor {name}: {shape}")
        self.tensors[name] = {"shape": tuple(shape), "layout": layout, "storage": storage,
                              "bytes": prod(shape) * bytes_per_element,
                              "producer": None, "consumers": []}
        return name

    def op(self, name, kind, inputs, shape, *, macs=0, vector=0, weights=0,
           layout="token_major", storage="activation", role=None, attrs=None):
        for source in inputs:
            self.tensors[source]["consumers"].append(name)
        self.tensor(name, shape, layout=layout, storage=storage)
        self.tensors[name]["producer"] = name
        self.nodes.append({"id": name, "kind": kind, "role": role or kind,
                           "inputs": tuple(inputs), "output": name, "macs": macs,
                           "vector_ops": vector, "weight_bytes": 2 * weights,
                           "attributes": attrs or {}})
        return name

    def linear(self, name, source, out, role):
        tokens, features = self.tensors[source]["shape"]
        return self.op(name, "linear", (source,), (tokens, out), macs=tokens * features * out,
                       weights=features * out, role=role)

    def vector(self, name, kind, inputs, per_element=1):
        shape = self.tensors[inputs[0]]["shape"]
        return self.op(name, kind, inputs, shape, vector=per_element * prod(shape))

    def finish(self):
        index = {node["id"]: i for i, node in enumerate(self.nodes)}
        for tensor in self.tensors.values():
            start = index[tensor["producer"]] if tensor["producer"] else -1
            tensor["lifetime"] = [start, max((index[x] for x in tensor["consumers"]), default=start)]
        return {"nodes": self.nodes, "tensors": self.tensors}


def attention_geometry(spec, layer, tokens):
    pattern = layer.get("attention", "global")
    if pattern == "global":
        return {tokens: 1}
    t, h, w = spec["grid"]
    if pattern == "spatial":
        return {h * w: t}
    if pattern == "temporal":
        return {t: h * w}
    if pattern == "window":
        return window_histogram((t, h, w), layer["window"])
    raise ValueError(f"Unknown attention pattern {pattern}")


def region_geometry(spec, layer):
    """Count 4×4 frame-local Region intersections in each attention group."""
    grid = spec["grid"]
    pattern = layer.get("attention", "global")
    tile = (4, 4) if len(grid) == 2 else (1, 4, 4)

    def group(shape):
        return prod(shape), prod(ceil(n / s) for n, s in zip(shape, tile))

    if layer["type"] in {"joint", "packed"}:
        tokens, regions = group(grid)
        context = spec["context"]
        groups = [(tokens + context, regions + ceil(context / 16), 1)]
    elif pattern == "spatial":
        frames, h, w = grid
        groups = [(h * w, ceil(h / 4) * ceil(w / 4), frames)]
    elif pattern == "temporal":
        frames, h, w = grid
        groups = [(frames, frames, h * w)]
    elif pattern == "window":
        window = layer["window"]
        spans = [Counter(min(n - i, s) for i in range(0, n, s)) for n, s in zip(grid, window)]
        groups = [(*group(tuple(n for n, _ in dims)), prod(count for _, count in dims))
                  for dims in product(*(axis.items() for axis in spans))]
    elif pattern == "global":
        groups = [(*group(grid), 1)]
    else:
        raise ValueError(f"Unknown attention pattern {pattern}")
    return {
        "query_regions": sum(r * count for _, r, count in groups),
        "region_pairs": sum(r**2 * count for _, r, count in groups),
        "key_tokens_per_query_region_sum": sum(n * r * count for n, r, count in groups),
    }


def attention(graph, prefix, q, k, v, q_tokens, pair_elements, width, heads, pattern):
    # Ragged windows are represented as a packed logical score stream.
    shape = (heads, pair_elements)
    attrs = {"query_tokens": q_tokens, "pair_elements": pair_elements, "pattern": pattern,
             "heads": heads, "head_width": width // heads}
    role = "self_attention" if prefix == "self" else "cross_attention"
    score = graph.op(prefix + ".qk", "qk", (q, k), shape, macs=pair_elements * width,
                     storage="attention_stream", layout="head_pair_stream", role=role, attrs=attrs)
    probs = graph.op(prefix + ".softmax", "softmax", (score,), shape, vector=5 * prod(shape),
                     storage="attention_stream", layout="head_pair_stream", role=role, attrs=attrs)
    return graph.op(prefix + ".pv", "pv", (probs, v), (q_tokens, width),
                    macs=pair_elements * width, role=role, attrs=attrs)


def ffn(graph, prefix, source, width, hidden):
    normal = graph.vector(prefix + "ffn_norm", "layer_norm", (source,), 8)
    first = graph.linear(prefix + "ffn_in", normal, hidden, "ffn_in")
    act = graph.vector(prefix + "activation", "gelu", (first,), 8)
    out = graph.linear(prefix + "ffn_out", act, width, "ffn_out")
    return graph.vector(prefix + "ffn_residual", "residual_add", (source, out))


def project_qkv(graph, prefix, source, width, spec):
    qkv = graph.linear(prefix + "qkv", source, 3 * width, "qkv")
    shape = (graph.tensors[source]["shape"][0], width)
    q, k, v = [graph.op(prefix + part, "view", (qkv,), shape, storage="view",
                        layout="token_head_channel") for part in ("q", "k", "v")]
    if spec.get("qk_norm"):
        q = graph.vector(prefix + "q_norm", "rms_norm", (q,), 6)
        k = graph.vector(prefix + "k_norm", "rms_norm", (k,), 6)
    if spec.get("rope"):
        q = graph.vector(prefix + "q_rope", "rotary", (q,), 3)
        k = graph.vector(prefix + "k_rope", "rotary", (k,), 3)
    return q, k, v


def conditioned(graph, prefix, source, cond, width, conditioning):
    if conditioning:
        coeff = (graph.linear(prefix + "adaln", cond, 6 * width, "conditioning")
                 if conditioning == "linear6" else cond)
        return graph.vector(prefix + "attn_norm", "adaptive_norm", (source, coeff), 11)
    return graph.vector(prefix + "attn_norm", "layer_norm", (source,), 8)


def packed_layer(graph, latent, context, cond, tokens, width, hidden, heads):
    total = tokens + graph.tensors[context]["shape"][0]
    joined = graph.op("join_streams", "concat", (latent, context), (total, width), storage="view")
    mod = graph.linear("packed_modulation", cond, 3 * width, "conditioning")
    normal = graph.vector("packed_norm", "adaptive_norm", (joined, mod), 11)
    packed = graph.linear("packed_in", normal, 3 * width + hidden, "packed_in")
    q, k, v = [graph.op(part, "view", (packed,), (total, width), storage="view")
               for part in ("q", "k", "v")]
    q = graph.vector("q_rope", "rotary_qk_norm", (q,), 9)
    k = graph.vector("k_rope", "rotary_qk_norm", (k,), 9)
    act = graph.op("packed_activation", "gelu", (packed,), (total, hidden), vector=8 * total * hidden)
    attn = attention(graph, "self", q, k, v, total, total**2, width, heads, "joint")
    both = graph.op("parallel_join", "concat", (attn, act), (total, width + hidden), storage="view")
    out = graph.linear("packed_out", both, width, "packed_out")
    graph.vector("packed_residual", "residual_add", (joined, out))


def build_layer(spec, layer):
    graph = Graph()
    n, d, heads, hidden = prod(spec["grid"]), spec["width"], spec["heads"], spec["ffn"]
    latent = graph.tensor("latent", (n, d), storage="layer_input")
    cond_width = 6 * d if spec.get("conditioning") == "shared6" else d
    cond = graph.tensor("conditioning", (1, cond_width), storage="forward_shared")
    kind = layer["type"]
    if kind in {"joint", "packed", "cross"}:
        context = graph.tensor("context", (spec["context"], d),
                               storage="layer_input" if kind != "cross" else "forward_shared")
    if kind == "packed":
        packed_layer(graph, latent, context, cond, n, d, hidden, heads)
        return {"tokens": n, "attention": "joint", "region_geometry": region_geometry(spec, layer),
                **graph.finish()}

    normal = conditioned(graph, "", latent, cond, d, spec.get("conditioning"))
    q, k, v = project_qkv(graph, "", normal, d, spec)
    histogram = attention_geometry(spec, layer, n)
    if kind == "joint":
        c = spec["context"]
        text_norm = conditioned(graph, "text.", context, cond, d,
                                None if layer.get("context_pre_only") else spec.get("conditioning"))
        text = project_qkv(graph, "text.", text_norm, d, spec)
        q, k, v = [graph.op("joint." + part, "concat", (image, txt), (n + c, d), storage="view")
                   for part, image, txt in zip(("q", "k", "v"), (q, k, v), text)]
        attn = attention(graph, "self", q, k, v, n + c, (n + c)**2, d, heads, "joint")
        histogram = {n + c: 1}
        if not layer.get("context_pre_only"):
            txt = graph.op("text.attention_slice", "view", (attn,), (c, d), storage="view")
            txt = graph.linear("text.out", txt, d, "attention_out")
            txt = graph.vector("text.attention_residual", "residual_add", (context, txt))
            ffn(graph, "text.", txt, d, hidden)
        attn = graph.op("image_attention_slice", "view", (attn,), (n, d), storage="view")
    else:
        attn = attention(graph, "self", q, k, v, n, sum(size**2 * count for size, count in histogram.items()),
                         d, heads, layer.get("attention", "global"))
    attn = graph.linear("out", attn, d, "attention_out")
    latent = graph.vector("attention_residual", "residual_add", (latent, attn))
    if kind == "cross":
        norm = graph.vector("cross.norm", "layer_norm", (latent,), 8)
        query = graph.linear("cross.q", norm, d, "cross_projection")
        key = graph.linear("cross.k", context, d, "cross_projection")
        value = graph.linear("cross.v", context, d, "cross_projection")
        if spec.get("qk_norm"):
            query = graph.vector("cross.q_norm", "rms_norm", (query,), 6)
            key = graph.vector("cross.k_norm", "rms_norm", (key,), 6)
        out = attention(graph, "cross", query, key, value, n, n * spec["context"], d, heads, "cross")
        out = graph.linear("cross.out", out, d, "cross_projection")
        latent = graph.vector("cross.residual", "residual_add", (latent, out))
    ffn(graph, "", latent, d, hidden)
    return {"tokens": n, "attention": "joint" if kind == "joint" else layer.get("attention", "global"),
            "window_histogram": histogram, "region_geometry": region_geometry(spec, layer), **graph.finish()}


def build(spec):
    if spec["width"] % spec["heads"] or min(spec["steps"], spec["forwards"], spec["lanes"]) <= 0:
        raise ValueError(f"Invalid model geometry: {spec['id']}")
    forwards = spec["steps"] * spec["forwards"]
    lane_evaluations = forwards * spec["lanes"]
    bundles = []
    macs = self_macs = 0
    for bundle in spec["bundles"]:
        repeat = bundle["repeat"]
        if repeat <= 0:
            raise ValueError("Nonpositive layer repetition")
        layers = []
        for layer in bundle["layers"]:
            graph = build_layer(spec, layer)
            graph["id"] = layer["id"]
            layers.append(graph)
            for node in graph["nodes"]:
                work = node["macs"] * repeat * lane_evaluations
                macs += work
                if node["role"] == "self_attention":
                    self_macs += work
        bundles.append({"repeat": repeat, "layers": layers})
    n, d = prod(spec["grid"]), spec["width"]
    return {"id": spec["id"], "evidence": spec["evidence"], "source": spec["source"],
            "assumptions": spec.get("assumptions", []), "bundles": bundles,
            "steps": spec["steps"], "forwards_per_step": spec["forwards"],
            "lanes_per_forward": spec["lanes"], "forward_calls": forwards,
            "cfg_lane_evaluations": lane_evaluations, "macs": macs, "self_attention_macs": self_macs,
            "boundary_projection_macs": n * d * (spec["patch_in"] + spec["patch_out"]) * lane_evaluations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads/dataflow")
    args = parser.parse_args()
    workloads = read_workloads(args.workloads)
    write(args.out, {"schema_version": 1, "workloads": {name: build(spec) for name, spec in workloads.items()},
                     "source_sha256": {"dataflow.py": digest(Path(__file__)),
                                       **{name: digest(args.workloads / f"{name}.json") for name in workloads}}})


if __name__ == "__main__":
    main()

