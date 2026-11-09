# Six-workload calibrated performance reconstruction

Latency covers denoising. Speedups use the historical H100 latency for each workload.

| Workload | H100 reference (ms) | HARP (ms) | BitRed | Ditto | EXION | AdapTiV | DSTAR | HARP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dit-xl2-512 | 1067.983 | 97.705 | 2.254× | 1.976× | 2.225× | 2.332× | 3.359× | 10.931× |
| flux1-dev | 15831.380 | 1473.841 | 2.216× | 2.149× | 2.310× | 2.381× | 3.659× | 10.742× |
| latte-1 | 42271.261 | 2999.434 | 2.246× | 2.342× | 2.366× | 2.461× | 4.009× | 14.093× |
| opensora-v13 | 1162178.060 | 89820.278 | 2.194× | 2.499× | 2.410× | 2.480× | 4.206× | 12.939× |
| sd3-medium | 1991.942 | 171.284 | 2.293× | 2.104× | 2.297× | 2.406× | 3.564× | 11.629× |
| wan21-14b-81f | 524488.869 | 38777.651 | 2.222× | 2.429× | 2.387× | 2.467× | 4.197× | 13.526× |
| Geometric mean | — | — | 2.238× | 2.242× | 2.332× | 2.420× | 3.818× | 12.243× |

## Historical algorithm and hardware ablation

Speedup is relative to the historical Matched NPU. Budget and All use the same combined algorithm on the generic and HARP substrates, respectively.

| Workload | Residual | PlaneClear | HARP-Budget | HARP-All |
|---|---:|---:|---:|---:|
| dit-xl2-512 | 1.103× | 3.761× | 3.817× | 6.581× |
| flux1-dev | 1.119× | 3.471× | 3.627× | 6.159× |
| latte-1 | 1.570× | 3.368× | 4.095× | 7.713× |
| opensora-v13 | 1.564× | 2.979× | 3.694× | 6.774× |
| sd3-medium | 1.202× | 3.568× | 3.625× | 6.668× |
| wan21-14b-81f | 1.550× | 3.177× | 3.863× | 7.238× |
| Geometric mean | 1.335× | 3.378× | 3.783× | 6.838× |

## Modeled algorithm and hardware ablation

Speedup is relative to the modeled dense generic substrate. Budget and All use the same combined algorithm on the generic and HARP substrates, respectively.

| Workload | Residual | PlaneClear | HARP-Budget | HARP-All |
|---|---:|---:|---:|---:|
| dit-xl2-512 | 1.039× | 0.993× | 0.967× | 1.539× |
| flux1-dev | 1.062× | 0.869× | 0.839× | 1.358× |
| latte-1 | 1.025× | 1.016× | 1.007× | 1.636× |
| opensora-v13 | 1.045× | 0.814× | 0.837× | 1.258× |
| sd3-medium | 1.116× | 0.767× | 0.778× | 1.218× |
| wan21-14b-81f | 1.370× | 0.523× | 0.658× | 1.019× |
| Geometric mean | 1.104× | 0.812× | 0.840× | 1.322× |

## Fixed-work hardware gain and formation overhead

The hardware ratios compare HARP-All with HARP-Budget using the same Combined work. Formation columns report the exposed share of full execution time.

| Workload | Historical All / Budget | Modeled All / Budget | Modeled exposed formation | Table II exposed formation |
|---|---:|---:|---:|---:|
| dit-xl2-512 | 1.724× | 1.591× | 0.027% | 1.870% |
| flux1-dev | 1.698× | 1.619× | 0.004% | 0.000% |
| latte-1 | 1.884× | 1.624× | 0.087% | 2.930% |
| opensora-v13 | 1.834× | 1.502× | 0.008% | 0.000% |
| sd3-medium | 1.839× | 1.566× | 0.008% | 3.240% |
| wan21-14b-81f | 1.874× | 1.550× | 0.002% | 0.000% |
| Geometric mean | 1.807× | 1.575× | — | — |

See report.json and formation.json for per-workload counters and model assumptions.
