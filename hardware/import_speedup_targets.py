"""Export the Figure 6 latency/speedup anchors using a read-only workbook."""

import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path


MODELS = {
    "DiT-XL/2-512": "dit-xl2-512", "SD3 Medium": "sd3-medium", "FLUX.1-dev": "flux1-dev",
    "Latte-1": "latte-1", "Wan2.1": "wan21-14b-81f", "Open-Sora v1.3": "opensora-v13",
}
METHODS = {"H100-Dense", "BitRed-Adapt", "Ditto-Adapt", "EXION-Adapt", "AdapTiV-Adapt", "DSTAR-Adapt"}


def main():
    from openpyxl import load_workbook

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    book = load_workbook(args.workbook, read_only=True, data_only=True)
    try:
        sheet = book["Sheet1"]
        rows = sheet.iter_rows(min_col=1, max_col=4, values_only=True)
        if next(rows) != ("Model", "Method", "Latency_ms", "Speedup_vs_H100"):
            raise ValueError("Unexpected workbook header")
        targets = {model: {} for model in MODELS.values()}
        for index, (name, method, latency, speedup) in enumerate(rows, 2):
            if name not in MODELS or method not in METHODS:
                continue
            group = targets[MODELS[name]]
            if method in group or not (latency > 0 and speedup > 0):
                raise ValueError(f"Duplicate or invalid anchor at row {index}")
            group[method] = {"latency_ms": latency, "speedup_vs_h100": speedup,
                             "range": f"A{index}:D{index}"}
    finally:
        book.close()
    for group in targets.values():
        if set(group) != METHODS:
            raise ValueError("Incomplete workbook anchors")
        h100 = group["H100-Dense"]["latency_ms"]
        for row in group.values():
            if not isclose(h100 / row["latency_ms"], row["speedup_vs_h100"], rel_tol=1e-7):
                raise ValueError(f"Inconsistent H100 speedup at {row['range']}")
    result = {"schema_version": 1, "sheet": "Sheet1", "file": args.workbook.name,
              "sha256": sha256(args.workbook.read_bytes()).hexdigest(), "workloads": targets}
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

