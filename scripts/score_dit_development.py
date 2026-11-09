"""Score the 16 fixed DiT inputs from the generation entry."""

import argparse
import json
import runpy
import sys
from pathlib import Path

from quality import dit_stage4 as stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    record = json.loads((root / "source-budget.json").read_text())
    items = stage.heldout_items()[::16]
    table = root / "tables/source-budget-table.json"
    if record["inputs"] != items or stage.file_sha(table) != record["sha256"]:
        raise ValueError("DiT scoring requires the 16 fixed inputs and the recorded budget")
    stage.heldout_items = lambda: items
    stage.CANDIDATES = ("risk-cap-0", "source-budget")
    stage.NONZERO = ("source-budget",)
    score = Path(__file__).with_name("dit_stage4_score.py")
    sys.argv = [str(score), "--phase", "heldout", "--output", str(root / "heldout"),
                "--candidates", str(root / "tables"), "--selection", str(root / "selection.json")]
    runpy.run_path(str(score), run_name="__main__")
    report = json.loads((root / "heldout/quality.json").read_text())
    if not report["official_adm_all_groups_passed"]:
        raise ValueError("Official ADM FID did not pass for all four modes; see quality.json")


if __name__ == "__main__":
    main()
