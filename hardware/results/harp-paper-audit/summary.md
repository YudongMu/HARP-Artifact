# Six-workload calibrated performance reconstruction

The calibrated model estimates denoising latency from compiled workloads and fitted service times. Speedups use historical H100 latency inputs. The run checks all five generated reports against frozen outputs.

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

Figure 7 compares speedups against the original dense algorithm on the Matched NPU. The component run uses a dense generic timing proxy. The historical Matched NPU values in `report.json` are comparison inputs. Reproducing the figure numerically requires timing the variants with the Matched NPU baseline.
