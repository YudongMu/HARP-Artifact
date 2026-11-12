# Six-workload performance reconstruction

The evaluator applies the released service-time parameters to compiled workloads. Speedups use historical H100 reference latencies. See the hardware documentation for calibration provenance.

| Workload | H100 reference (ms) | Estimated HARP (ms) | BitRed | Ditto | EXION | AdapTiV | DSTAR | HARP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dit-xl2-512 | 1067.983 | 97.705 | 2.254× | 1.976× | 2.225× | 2.332× | 3.359× | 10.931× |
| flux1-dev | 15831.380 | 1473.841 | 2.216× | 2.149× | 2.310× | 2.381× | 3.659× | 10.742× |
| latte-1 | 42271.261 | 2999.434 | 2.246× | 2.342× | 2.366× | 2.461× | 4.009× | 14.093× |
| opensora-v13 | 1162178.060 | 89820.278 | 2.194× | 2.499× | 2.410× | 2.480× | 4.206× | 12.939× |
| sd3-medium | 1991.942 | 171.284 | 2.293× | 2.104× | 2.297× | 2.406× | 3.564× | 11.629× |
| wan21-14b-81f | 524488.869 | 38777.651 | 2.222× | 2.429× | 2.387× | 2.467× | 4.197× | 13.526× |
| Geometric mean | — | — | 2.238× | 2.242× | 2.332× | 2.420× | 3.818× | 12.243× |

## Ablation results

The workload compiler computes net QK/PV MAC-equivalent reductions relative to dense attention and reductions in active linear bit-plane components relative to the counts before plane clearing.

| Workload | Residual QK/PV reduction | PlaneClear linear plane reduction | Combined QK/PV reduction | Combined linear plane reduction |
|---|---:|---:|---:|---:|
| dit-xl2-512 | 40.67% | 45.50% | 40.67% | 43.02% |
| flux1-dev | 44.58% | 38.80% | 44.58% | 33.05% |
| latte-1 | 51.73% | 43.03% | 51.73% | 40.61% |
| opensora-v13 | 46.75% | 38.81% | 46.75% | 36.55% |
| sd3-medium | 41.70% | 44.72% | 41.70% | 42.33% |
| wan21-14b-81f | 58.61% | 40.63% | 58.61% | 37.57% |

The generic and HARP hardware models used identical Combined task lists in 6 of 6 workloads.

## Speedup Ablation relative to Matched NPU

We compare HARP-Residual, HARP-PlaneClear, HARP-Budget (Combined), and HARP-All against the dense algorithm on Matched NPU. Both reports compute HARP-All latency with the same evaluator and inputs.

| Workload | Matched NPU Dense (ms) | HARP-Residual | HARP-PlaneClear | HARP-Budget | HARP-All |
|---|---:|---:|---:|---:|---:|
| dit-xl2-512 | 642.988 | 1.103× | 3.761× | 3.584× | 6.581× |
| flux1-dev | 9077.531 | 1.119× | 3.471× | 3.362× | 6.159× |
| latte-1 | 23136.134 | 1.570× | 3.368× | 4.095× | 7.713× |
| opensora-v13 | 608433.490 | 1.564× | 2.979× | 3.694× | 6.774× |
| sd3-medium | 1142.157 | 1.202× | 3.568× | 3.625× | 6.668× |
| wan21-14b-81f | 280686.828 | 1.550× | 3.177× | 3.863× | 7.238× |
| Geometric mean | — | 1.335× | 3.378× | 3.697× | 6.838× |

Per-configuration latencies and calculated speedups are in `matched-npu-ablation.json`.
