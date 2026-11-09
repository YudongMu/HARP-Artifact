"""Stage frozen Wan selection evidence into an independent reviewer run."""

import argparse
import json
from pathlib import Path

from quality import wan_stage4 as stage
from quality.wan_stage4_score import validate_selection


def prepare(reference, run, source):
    def read(path):
        return json.loads(path.read_bytes())

    def stage_bytes(path, data):
        dest = run / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.read_bytes() != data:
                raise ValueError(f"Frozen Wan reviewer input changed: {dest}")
        else:
            with dest.open("xb") as stream:
                stream.write(data)

    decision = read(reference / "selection.json")
    policy = decision["selected"]
    if policy["id"] != "mixed" or decision["heldout_used_for_selection"] is not False:
        raise ValueError("Expected the frozen Wan mixed development selection")
    shared = read(source)
    if shared["allowances"] != policy["allowances"]:
        raise ValueError("Shared Wan λ values differ from the frozen selection")
    binding = decision["binding"]
    for mode, row in binding["tables"].items():
        name = Path(row["path"]).relative_to(run)
        frozen = read(reference / name)
        identity = stage.table_identity(shared["identity"], mode)
        derived = dict(frozen, identity={key: identity[key] for key in frozen["identity"]},
                       allowances=shared["allowances"], provenance=shared["provenance"])
        if identity != frozen["identity"] or derived != frozen:
            raise ValueError(f"Wan deployment binding differs for {mode}")
        data = json.dumps(derived, indent=2, allow_nan=False).encode()
        stage_bytes(name, data)
        if stage.sha(run / name) != row["sha256"]:
            raise ValueError(f"Wan historical table SHA differs for {mode}")

    files = ["manifest.json", "selection.json", "candidates.json", "calibration-scoring-context.json"]
    for label in binding["scores"]:
        files.extend(f"calibration-scores/{label}/{name}" for name in ("scores.json", "score-execution.json"))
    candidate = Path(policy["calibration_run_directory"]).relative_to(run)
    files.extend(candidate / name for name in ("run.json", "stage4-execution.json", "initial-noise.pt", "latent.pt", "sample.mp4"))
    files.append(Path(policy["candidate_tables"]["combined"]).relative_to(run))
    for path in files:
        stage_bytes(path, (reference / path).read_bytes())
    stage.verify_selection(run, decision)

    manifest = read(run / "manifest.json")
    stage.verified_case(run / candidate, manifest["inputs"]["calibration"][0], "combined",
                        stage.Protocol(9), policy["candidate_tables"]["combined"], manifest["generator_source_sha256"])
    target = run / "generation-index.json"
    if not target.exists():
        index = read(reference / target.name)
        candidates = [case for case in index["cases"] if case["split"] == "calibration"
                      and case["kind"] == "candidate" and case["run_directory"] == policy["calibration_run_directory"]]
        if len(candidates) != 1 or candidates[0]["run_sha256"] != stage.sha(run / candidate / "run.json"):
            raise ValueError("Missing frozen Wan calibration candidate receipt")
        index = dict(index, status="running", cases=candidates)
        validate_selection(decision, index, run)
        stage.write_json(target, index)
    stage.immutable_json(run / "source-budget.json", {
        "source": str(source), "sha256": stage.sha(source), "inputs": manifest["inputs"]["heldout"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    prepare(args.reference, args.run, args.source)


if __name__ == "__main__":
    main()
