"""Validate matched DiT generations and collect their ADM FID scores."""

import argparse
import hashlib
import json
import math
from pathlib import Path


MODES = {
    "dense": "Dense BF16",
    "residual": "HARP-Residual",
    "planeclear": "HARP-PlaneClear",
    "combined": "HARP-Budget / HARP",
}
COUNT = 5000


def paired_items(count, seed_base=100000):
    if count <= 0:
        raise ValueError("Positive sample count required")
    return [
        {"index": index, "class_id": index % 1000, "seed": seed_base + index}
        for index in range(count)
    ]


def evaluation_items(count, seed_base=100000):
    if count < 2:
        raise ValueError("Paired FID requires at least two samples")
    complete, extra = divmod(count, 1000)
    items = paired_items(complete * 1000, seed_base) if complete else []
    for index in range(extra):
        class_id = index * 1000 // extra
        items.append({"index": len(items), "class_id": class_id,
                      "seed": seed_base + complete * 1000 + class_id})
    return items


def comparison_modes(value):
    modes = tuple(value.split(","))
    if "dense" not in modes or len(modes) < 2 or len(set(modes)) != len(modes) or not set(modes) <= MODES.keys():
        raise ValueError("Select Dense and at least one distinct HARP mode")
    return tuple(mode for mode in MODES if mode in modes)


def compare_generations(runs, count, modes=tuple(MODES)):
    if set(runs) != set(modes) or "dense" not in runs:
        raise ValueError("Generation modes differ from the requested comparison")
    contract = runs["dense"]["paired_contract"]
    if contract.get("cohort") == "class-balanced-v1":
        expected = evaluation_items(count, contract.get("seed_base", 100000))
    else:
        expected = paired_items(count, contract.get("seed_base", 100000))
    paired = None
    for mode, run in runs.items():
        if (run["mode"] != mode or run["status"] != "complete"
                or run["count"] != count or run["completed"] != count
                or len(run["samples"]) != count):
            raise ValueError(f"Incomplete or mismatched generation for {mode}")
        if run["paired_contract"] != contract:
            raise ValueError(f"Sampling contract differs for {mode}")
        rows = [
            {key: sample[key] for key in ("index", "class_id", "seed", "noise_sha256")}
            for sample in run["samples"]
        ]
        if any(
            {key: row[key] for key in item} != item or not row["noise_sha256"]
            for row, item in zip(rows, expected)
        ) or (paired is not None and rows != paired):
            raise ValueError(f"Paired inputs differ for {mode}")
        paired = rows
    return paired


def compare_scores(scores, count):
    reference = tuple(scores["dense"][key] for key in ("reference_sha256", "inception_graph_sha256"))
    results = {}
    for mode in scores:
        score = scores[mode]
        if (score["generated_samples"] != count
                or tuple(score[key] for key in ("reference_sha256", "inception_graph_sha256")) != reference):
            raise ValueError(f"ADM reference or sample count differs for {mode}")
        fid = score["fid_by_sample_count"][str(count)]
        if not math.isfinite(fid) or fid < 0:
            raise ValueError(f"Invalid FID for {mode}")
        results[mode] = {"label": MODES[mode], "fid": fid}
    dense = results["dense"]["fid"]
    if dense <= 0:
        raise ValueError("Dense FID must be positive for a relative comparison")
    for row in results.values():
        row["delta_fid"] = row["fid"] - dense
        row["delta_percent_dense"] = 100 * row["delta_fid"] / dense
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args()
    modes = comparison_modes(args.modes)
    manifests = {mode: (args.output / mode / "generation.json").read_bytes() for mode in modes}
    runs = {mode: json.loads(data) for mode, data in manifests.items()}
    paired = compare_generations(runs, args.count, modes)
    if args.phase == "check":
        print(f"Verified {len(paired)} matched inputs in each of {len(modes)} modes.")
        return
    scores = {mode: json.loads((args.output / mode / "fid-pilot.json").read_text()) for mode in modes}
    for mode, score in scores.items():
        if score["generation_manifest_sha256"] != hashlib.sha256(manifests[mode]).hexdigest():
            raise ValueError(f"ADM scores do not match the generation manifest for {mode}")
    report = {
        "metric": "ADM FID",
        "samples_per_mode": args.count,
        "class_count": len({row["class_id"] for row in paired}),
        "paired_inputs_sha256": hashlib.sha256(json.dumps(paired, sort_keys=True).encode()).hexdigest(),
        "sampling_contract": runs["dense"]["paired_contract"],
        "reference_sha256": scores["dense"]["reference_sha256"],
        "inception_graph_sha256": scores["dense"]["inception_graph_sha256"],
        "modes": compare_scores(scores, args.count),
    }
    temporary = args.output / "fid-comparison.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output / "fid-comparison.json")
    print(json.dumps(report["modes"], indent=2))


if __name__ == "__main__":
    main()
