"""Extract HARP ablation anchors from the reference workbook."""

import argparse
from hashlib import sha256
from math import isfinite
from pathlib import Path

from .baselines.common import write
from .import_speedup_targets import MODELS


METHODS = {
    "HARP-Residual": "residual", "HARP-PlaneClear": "planeclear",
    "HARP-Budget": "combined_generic", "HARP-All": "combined_harp",
}


def main():
    from openpyxl import load_workbook

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    book = load_workbook(args.workbook, read_only=True, data_only=True)
    rows = {name: {} for name in MODELS.values()}
    try:
        sheet = book["Sheet1"]
        values = sheet.iter_rows(min_col=1, max_col=3, values_only=True)
        if next(values) != ("Model", "Method", "Speedup_vs_MatchedNPU"):
            raise ValueError("Unexpected ablation workbook header")
        for index, (name, method, speedup) in enumerate(values, 2):
            if method not in METHODS:
                continue
            results = rows[MODELS[name]]
            mode = METHODS[method]
            if mode in results or not isinstance(speedup, (int, float)) or not isfinite(speedup) or speedup <= 0:
                raise ValueError(f"Duplicate or invalid HARP anchor at row {index}")
            results[mode] = {"speedup": speedup, "range": f"A{index}:C{index}"}
    finally:
        book.close()
    if any(set(workload) != set(METHODS.values()) for workload in rows.values()):
        raise ValueError("Incomplete HARP ablation anchors")
    write(args.out, {"schema_version": 1, "file": args.workbook.name, "sheet": "Sheet1",
                     "sha256": sha256(args.workbook.read_bytes()).hexdigest(),
                     "denominator": "Matched NPU", "workloads": rows})


if __name__ == "__main__":
    main()

