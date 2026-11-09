"""Compare independent baseline component models on six workload graphs."""

import argparse
from math import exp, log, sqrt
from pathlib import Path
from statistics import mean

from ..component_timing import run_graph
from ..dataflow import build, read_workloads
from .common import ROOT, digest, read, write
from .components import adaptiv, bitred, ditto, dstar, exion
from .components.common import model_inputs, workload_profile


BASELINES = {module.__name__.rsplit(".", 1)[-1]: module
             for module in (adaptiv, bitred, ditto, dstar, exion)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, default=ROOT / "workloads/dataflow")
    args = parser.parse_args()
    specs = read_workloads(args.workloads)
    graphs = {name: build(spec) for name, spec in specs.items()}
    comparison_path = ROOT / "configs/baseline-comparison.json"
    targets_path = ROOT / "reference/e2e-speedups.json"
    comparison = read(comparison_path)
    reference = read(targets_path)
    baselines = {}
    dependencies = [comparison_path, targets_path, Path(__file__), ROOT / "dataflow.py",
                    ROOT / "component_timing.py", ROOT / "baselines/components/common.py"]
    for method, module in BASELINES.items():
        profile, timing, config, inputs = model_inputs(method, normalized=True)
        dependencies.extend((*inputs, Path(module.__file__)))
        rows = {}
        for name, graph in graphs.items():
            settings = workload_profile(profile, name)
            outcome = run_graph(graph, config, module.phases(graph, settings))
            anchors = reference["workloads"][name]
            target = anchors[comparison["anchor_methods"][method]]["speedup_vs_h100"]
            reference_ms = anchors["H100-Dense"]["latency_ms"]
            target_ms = reference_ms / target
            speedup = reference_ms / (1000 * outcome["seconds"])
            pe_only_ms = 1000 * outcome["resource_cycles"].get("pe", 0) / config["clock_hz"]
            rows[name] = {**outcome, "speedup_vs_reference_h100": speedup, "target_speedup": target,
                          "log_speedup_ratio_error": log(speedup / target),
                          "target_latency_ms": target_ms,
                          "peak_array_only_ms": pe_only_ms,
                          "target_below_array_only_time": target_ms < pe_only_ms,
                          "workload_evidence": graph["evidence"], "profile": settings}
        errors = [row["log_speedup_ratio_error"] for row in rows.values()]
        bias = mean(errors)
        baselines[method] = {"workloads": rows, "config": config, "cost_evidence": timing["evidence"],
                             "speedup_geomean": exp(mean(log(r["speedup_vs_reference_h100"])
                                                       for r in rows.values())),
                             "target_geomean": exp(mean(log(r["target_speedup"]) for r in rows.values())),
                             "log_speedup_bias": bias,
                             "cross_model_log_rmse_after_global_scale": sqrt(mean((x-bias)**2 for x in errors))}
    write(args.out, {
        "schema_version": 1, "model": "baseline_graph_components_v0.2",
        "scope": "Normalized block-scope component estimates with approximate timing constants",
        "reference_evidence": reference["evidence"],
        "baselines": baselines,
        "workload_sha256": {name: digest(args.workloads / f"{name}.json") for name in graphs},
        "source_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in dependencies},
    })


if __name__ == "__main__":
    main()

