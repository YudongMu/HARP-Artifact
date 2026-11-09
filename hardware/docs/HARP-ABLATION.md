# HARP workload and ablation

HARP combines the six Table I constrained operator graphs with model and site profiles from Figures 8 and 9(a). The profiles retain a shared fallback and within-plane activity estimate. FLUX packed linear operators use QKV and output projection site proxies. The full assumptions and evidence are described in [Paper consistency audit](PAPER-AUDIT.md).

The compiler separates dense attention, refresh, replay, selected work, active and natural plane skips, fallback MACs and formation. Hardware comparisons replay the same Combined task list with matched modeled compute, capacity and HBM. The component run computes speedup relative to its generic dense proxy. The report also carries the historical Matched NPU values as comparison inputs. Its `reference_hardware_gain` is the ratio of the input HARP-All and HARP-Budget speedups.

The [Matched NPU replay](../matched_npu_ablation.py) calculates Dense, Residual, PlaneClear, and Budget latency using the historical analytical cycle model and [its workload parameters](../workloads/harp/matched-npu.json). The workload parameters come from the original synthetic-simulated tile and region profiles. This path preserves the historical work proxy and its 262.1 TFLOP/s throughput parameter. The separate revised component model uses the 32.768 TFLOP/s dense BF16 peak. The Matched NPU replay uses the frozen HARP-All latency from the end-to-end comparison against H100. The replay derives speedups from latencies without loading the Figure 7 ratios.

Run from `AE/`.

```bash
python3 -B -m hardware --out /tmp/harp/report.json --tasks-out /tmp/harp/workload.json
python3 -B -m hardware.formation_ablation --ablation-dir /tmp/harp --out /tmp/harp/formation.json
python3 -B -m hardware.reconstruct_harp_e2e \
  --workload-tasks /tmp/harp/workload.json --formation-report /tmp/harp/formation.json \
  --out /tmp/harp/e2e.json
python3 -B -m hardware.matched_npu_ablation --out /tmp/harp/matched-npu-ablation.json
```

Table II formation exposure is applied to the Full HARP timing. For the E2E reconstruction, fitted component service times and the Table II formation fraction are evaluated on the compiled Combined work. Changing the workload geometry or enabling previously unseen component shapes may require rebuilding the empirical latency tables.

The independent DiT development diagnostic remains available through `python3 -B -m hardware.harp_components --observed-dit-trace hardware/reference/dit-development-trace.json --out /tmp/harp-development.json`.

