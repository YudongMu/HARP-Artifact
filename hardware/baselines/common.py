"""Shared configuration and report I/O for hardware models."""

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def native_architecture(spec):
    return {"clock_hz": spec["clock_hz"], "macs_per_cycle": spec["macs_per_cycle"],
            "vector_ops_per_cycle": spec["vector_ops_per_cycle"],
            "hbm_bytes_per_cycle": spec["bandwidth_gbps"] * 1e9 / spec["clock_hz"]}
