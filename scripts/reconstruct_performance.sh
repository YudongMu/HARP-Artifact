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
  python3 -B -m hardware.matched_npu_ablation --out "$out/matched-npu-ablation.json"
} 2>&1 | tee "$out/reconstruction.log"

python3 -B - "$out" <<'PY' > "$out/summary.md"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
e2e = json.loads((out / 'e2e.json').read_text())
abl = json.loads((out / 'report.json').read_text())
matched_npu = json.loads((out / 'matched-npu-ablation.json').read_text())
baseline = json.loads((out / 'baseline.json').read_text())['baselines']
methods = ('bitred', 'ditto', 'exion', 'adaptiv', 'dstar')
print('# Six-workload calibrated performance reconstruction\n')
print('The calibrated model estimates denoising latency from compiled workloads and fitted service times. '
      'Speedups use historical H100 latency inputs. The run checks all five generated reports against frozen outputs.\n')
print('| Workload | H100 reference (ms) | Estimated HARP (ms) | BitRed | Ditto | EXION | AdapTiV | DSTAR | HARP |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for workload, row in e2e['workloads'].items():
    values = row['reconstructed_baseline_speedups_vs_h100']
    factors = [values[m] for m in methods] + [row['speedup_vs_reference_h100']]
    print(f"| {workload} | {row['h100_reference_ms']:.3f} | {row['estimated_full_harp_ms']:.3f} | "
          + ' | '.join(f'{value:.3f}×' for value in factors) + ' |')
means = [baseline[m]['speedup_geomean'] for m in methods] + [e2e['geomean_speedup_vs_reference_h100']]
print('| Geometric mean | — | — | ' + ' | '.join(f'{value:.3f}×' for value in means) + ' |')
print('\n## Ablation results\n')
print('The workload compiler computes net QK/PV MAC-equivalent reductions relative to dense attention and '
      'reductions in active linear bit-plane components relative to the counts before plane clearing.\n')
print('| Workload | Residual QK/PV reduction | PlaneClear linear plane reduction | Combined QK/PV reduction | Combined linear plane reduction |')
print('|---|---:|---:|---:|---:|')
matched_workloads = 0
for workload, row in abl['workloads'].items():
    work = row['algorithm_work']
    values = (
        work['residual']['net_qkpv_mac_equivalent_reduction'],
        work['planeclear']['linear_plane_components_active_reduction'],
        work['combined']['net_qkpv_mac_equivalent_reduction'],
        work['combined']['linear_plane_components_active_reduction'],
    )
    print(f'| {workload} | ' + ' | '.join(f'{value:.2%}' for value in values) + ' |')
    timings = row['timings']
    matched_workloads += (
        timings['combined_generic']['workload_sha256'] == timings['combined_harp']['workload_sha256']
    )
print(f"\nThe generic and HARP hardware models used identical Combined task lists in "
      f"{matched_workloads} of {len(abl['workloads'])} workloads.")
print('\n## Speedup Ablation relative to Matched NPU\n')
print('We compare HARP-Residual, HARP-PlaneClear, HARP-Budget (Combined), and HARP-All '
      'against the dense algorithm on Matched NPU. The end-to-end and ablation comparisons '
      'use the same HARP-All latency.\n')
print('| Workload | Matched NPU Dense (ms) | HARP-Residual | HARP-PlaneClear | HARP-Budget | HARP-All |')
print('|---|---:|---:|---:|---:|---:|')
modes = ('residual', 'planeclear', 'budget', 'all')
for workload, row in matched_npu['workloads'].items():
    factors = ' | '.join(f"{row['speedup'][mode]:.3f}×" for mode in modes)
    print(f"| {workload} | {row['matched_npu_dense_ms']:.3f} | {factors} |")
factors = ' | '.join(f"{matched_npu['geomean_speedup'][mode]:.3f}×" for mode in modes)
print(f'| Geometric mean | — | {factors} |')
print('\nPer-configuration latencies and calculated speedups are in `matched-npu-ablation.json`.')
PY
printf 'Summary: %s/summary.md\n' "$out"
