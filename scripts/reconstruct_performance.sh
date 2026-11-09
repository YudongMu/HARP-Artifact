#!/usr/bin/env bash
set -euo pipefail
if (( $# > 1 )); then
  echo "Usage: bash scripts/reconstruct_performance.sh [new-output-directory]" >&2
  exit 2
fi
if (( $# == 1 )); then
  mkdir -- "$1"
  out=$(cd -- "$1" && pwd)
else
  out=$(mktemp -d)
fi
cd "$(dirname "$0")/.."
printf 'Reports and logs: %s\n' "$out"

{
  python3 -B -m unittest discover -s hardware/tests
  python3 -B -m hardware.baselines.reconstruct_latencies --out "$out/baseline.json"
  python3 -B -m hardware --out "$out/report.json" --tasks-out "$out/workload.json"
  python3 -B -m hardware.formation_ablation --ablation-dir "$out" --out "$out/formation.json"
  python3 -B -m hardware.reconstruct_harp_e2e \
    --workload-tasks "$out/workload.json" --formation-report "$out/formation.json" --out "$out/e2e.json"
  cmp "$out/baseline.json" hardware/results/baseline-paper-audit/report.json
  for name in workload report formation e2e; do
    cmp "$out/$name.json" "hardware/results/harp-paper-audit/$name.json"
  done
  echo 'All five reports match the frozen references byte for byte.'
} 2>&1 | tee "$out/reconstruction.log"

python3 -B - "$out" <<'PY' > "$out/summary.md"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
e2e = json.loads((out / 'e2e.json').read_text())
abl = json.loads((out / 'report.json').read_text())
formation = json.loads((out / 'formation.json').read_text())
baseline = json.loads((out / 'baseline.json').read_text())['baselines']
methods = ('bitred', 'ditto', 'exion', 'adaptiv', 'dstar')
print('# Six-workload calibrated performance reconstruction\n')
print('Latency covers denoising. Speedups use the historical H100 latency for each workload.\n')
print('| Workload | H100 reference (ms) | HARP (ms) | BitRed | Ditto | EXION | AdapTiV | DSTAR | HARP |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for workload, row in e2e['workloads'].items():
    values = row['reconstructed_baseline_speedups_vs_h100']
    factors = [values[m] for m in methods] + [row['speedup_vs_reference_h100']]
    print(f"| {workload} | {row['h100_reference_ms']:.3f} | {row['estimated_full_harp_ms']:.3f} | "
          + ' | '.join(f'{value:.3f}×' for value in factors) + ' |')
means = [baseline[m]['speedup_geomean'] for m in methods] + [e2e['geomean_speedup_vs_reference_h100']]
print('| Geometric mean | — | — | ' + ' | '.join(f'{value:.3f}×' for value in means) + ' |')
configurations = ('residual', 'planeclear', 'combined_generic', 'combined_harp')
for title, denominator, field, aggregate in (
    ('Historical algorithm and hardware ablation', 'historical Matched NPU',
     'reference_speedup_vs_matched_npu', 'reference_speedup'),
    ('Modeled algorithm and hardware ablation', 'modeled dense generic substrate',
     'speedup_vs_modeled_dense', 'modeled_speedup'),
):
    print(f'\n## {title}\n')
    print(f'Speedup is relative to the {denominator}. Budget and All use the same combined algorithm on the generic and HARP substrates, respectively.\n')
    print('| Workload | Residual | PlaneClear | HARP-Budget | HARP-All |')
    print('|---|---:|---:|---:|---:|')
    for workload, row in abl['workloads'].items():
        values = [row['timings'][mode][field] for mode in configurations]
        print(f'| {workload} | ' + ' | '.join(f'{v:.3f}×' for v in values) + ' |')
    values = [abl['geomean'][mode][aggregate] for mode in configurations]
    print('| Geometric mean | ' + ' | '.join(f'{v:.3f}×' for v in values) + ' |')

print('\n## Fixed-work hardware gain and formation overhead\n')
print('The hardware ratios compare HARP-All with HARP-Budget using the same Combined work. Formation columns report the exposed share of full execution time.\n')
print('| Workload | Historical All / Budget | Modeled All / Budget | Modeled exposed formation | Table II exposed formation |')
print('|---|---:|---:|---:|---:|')
for workload, row in abl['workloads'].items():
    form = formation['workloads'][workload]
    print(f"| {workload} | {row['reference_hardware_gain']:.3f}× | {row['hardware_gain_fixed_combined']:.3f}× "
          f"| {form['exposed_formation_fraction']:.3%} | {form['table_ii_exposed_fraction']:.3%} |")
print(f"| Geometric mean | {abl['reference_hardware_gain_geomean']:.3f}× "
      f"| {abl['fixed_work_hardware_gain_geomean']:.3f}× | — | — |")
print('\nSee report.json and formation.json for per-workload counters and model assumptions.')
PY
printf 'Summary: %s/summary.md\n' "$out"
