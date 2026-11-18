# HARP component timing study

The component sweep evaluates graph workloads, existing algorithm profiles and configured resource rates. The release reports are loaded separately for the comparisons below.

Reproduce the sweep from the study checkout.

    python3 -B -m hardware.component_study --out /tmp/harp-component-sensitivity.json

The sweep retains the revised 32.768 TFLOP/s dense BF16 peak at 2 GHz for the first five scenarios. The existing within-plane density proxy is 2/32. The 1/32 and 4/32 cases bracket the effect of this unmeasured input. Full issue assumes 100% utilization of sparse issue lanes. The trajectory overlap bound lets every resource overlap fully for the same compiled work. It is a lower bound on latency.

| HARP scenario | PE issues/cycle | Sparse issue | Active fraction | H100 speedup | H100 speedup ceiling | Modeled Dense speedup |
|---|---:|---:|---:|---:|---:|---:|
| Current component model | 8,192 | 86.83% | 2/32 | 0.200× | 0.225× | 1.322× |
| Full sparse issue | 8,192 | 100.00% | 2/32 | 0.225× | 0.257× | 1.487× |
| Half active fraction | 8,192 | 86.83% | 1/32 | 0.277× | 0.328× | 1.833× |
| Double active fraction | 8,192 | 86.83% | 4/32 | 0.130× | 0.140× | 0.858× |
| Full issue and half active fraction | 8,192 | 100.00% | 1/32 | 0.309× | 0.374× | 2.041× |
| Legacy peak sensitivity | 65,525 | 86.83% | 2/32 | 0.896× | 1.801× | 1.118× |

Speedups are geometric means over six workloads. H100 denominators are the configured historical comparison latencies. The legacy peak sensitivity changes the PE rate to the 262.1 TFLOP/s value recorded by the historical analytical model. This exceeds the revised hardware specification.

| Workload | Release estimate (ms) | Current (ms) | Full issue and half active fraction (ms) | Perfect overlap bound (ms) | Dense BF16 alone (ms) |
|---|---:|---:|---:|---:|---:|
| dit-xl2-512 | 97.705 | 2,233.354 | 1,402.199 | 1,111.528 | 17.131 |
| flux1-dev | 1,473.841 | 149,344.528 | 87,968.320 | 75,192.326 | 949.952 |
| latte-1 | 2,999.434 | 91,741.952 | 50,284.707 | 40,306.169 | 699.816 |
| opensora-v13 | 89,820.278 | 3,254,878.598 | 2,163,502.772 | 1,871,779.869 | 555,802.950 |
| sd3-medium | 171.284 | 24,703.686 | 17,608.902 | 13,624.687 | 134.221 |
| wan21-14b-81f | 38,777.651 | 5,488,782.885 | 4,217,259.163 | 3,679,812.717 | 485,983.566 |

All six workloads remain PE-bound in the full-issue and half-density case. The dense BF16 stages alone exceed the released latency on Wan and Open-Sora. These bounds apply to the current graph and modeled architecture. The legacy analytical configuration also used different workload accounting. In particular, its DiT configuration specifies 250 steps while the revised DiT graph specifies 50 with two CFG lanes.

| Baseline | Release speedup vs H100 | Normalized component speedup | Component ceiling | Legacy peak sensitivity |
|---|---:|---:|---:|---:|
| adaptiv | 2.420× | 0.334× | 0.352× | 1.917× |
| bitred | 2.238× | 0.190× | 0.200× | 1.089× |
| ditto | 2.242× | 0.240× | 0.259× | 1.244× |
| dstar | 3.818× | 0.229× | 0.241× | 1.337× |
| exion | 2.332× | 0.287× | 0.308× | 1.536× |

The normalized baseline sweep uses each method's existing component costs and the shared 8,192 MAC/cycle peak. The legacy peak sensitivity changes the PE rate to the same historical value for each method. No candidate in this study supports substituting its parameters for the release table.
