"""Verify computed performance metrics against the archived reports."""

import argparse
from pathlib import Path

from .baselines.common import ROOT, read


REFERENCES = {
    "baseline.json": "baseline-paper-audit/report.json",
    "workload.json": "harp-paper-audit/workload.json",
    "report.json": "harp-paper-audit/report.json",
    "formation.json": "harp-paper-audit/formation.json",
    "e2e.json": "harp-paper-audit/e2e.json",
    "matched-npu-ablation.json": "harp-paper-audit/matched-npu-ablation.json",
}

# Only report metadata and comparisons to external targets differ across schemas.
REPORT_METADATA = {
    "schema_version", "model", "timing", "scope", "source_sha256", "input_sha256",
    "performance_inputs_sha256", "reference", "reference_evidence",
}
REFERENCE_METRICS = {
    "baseline.json": {"max_relative_error", "target_speedup", "relative_error"},
    "e2e.json": {"max_relative_error", "historical_speedup_vs_h100", "relative_error"},
    "report.json": {"reference_hardware_gain_geomean", "reference_hardware_gain",
                    "reference_speedup", "reference_speedup_vs_matched_npu", "relative_speedup_error"},
    "formation.json": {"historical_speedups_vs_matched_npu"},
}


def without_reference_metrics(value, excluded):
    if isinstance(value, dict):
        return {key: without_reference_metrics(item, excluded) for key, item in value.items()
                if key not in excluded}
    if isinstance(value, list):
        return [without_reference_metrics(item, excluded) for item in value]
    return value


def computed_report(name, report):
    if name == "workload.json":
        return report
    values = {key: value for key, value in report.items() if key not in REPORT_METADATA}
    return without_reference_metrics(values, REFERENCE_METRICS.get(name, set()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    for name, reference in REFERENCES.items():
        generated = computed_report(name, read(args.output_directory / name))
        expected = computed_report(name, read(ROOT / "results" / reference))
        if generated != expected:
            parser.exit(1, f"Computed metrics differ from the archived report: {name}\n")
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
