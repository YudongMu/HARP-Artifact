# HARP workload and ablation

HARP combines the six Table I constrained operator graphs with model and site profiles from Figures 8 and 9(a). The profiles retain a shared fallback and within-plane activity estimate. FLUX packed linear operators use QKV and output projection site proxies. The full assumptions and evidence are described in [Paper consistency audit](PAPER-AUDIT.md).

The compiler separates dense attention, refresh, replay, selected work, active and natural plane skips, fallback MACs and formation. Hardware comparisons replay the same Combined task list with matched modeled compute, capacity and HBM. The component run computes speedup relative to its generic dense proxy. The report also carries the historical Matched NPU values as comparison inputs. Its `reference_hardware_gain` is the ratio of the input HARP-All and HARP-Budget speedups.

The [Matched NPU replay](../matched_npu_ablation.py) calculates Dense, Residual, PlaneClear, and Budget latency using the historical analytical cycle model and [its workload parameters](../workloads/harp/matched-npu.json). The workload parameters come from the original synthetic-simulated tile and region profiles. This path preserves the historical work proxy and its 262.1 TFLOP/s throughput parameter. The separate revised component model uses the 32.768 TFLOP/s dense BF16 peak. Both reports compute HARP-All latency using the [shared evaluator](../harp_latency.py) with the compiled Combined tasks, released service times, and formation exposure fractions. Matched NPU speedups are calculated from the resulting latencies.

Run from `AE/`.

```bash
python3 -B -m hardware --out /tmp/harp/report.json --tasks-out /tmp/harp/workload.json
python3 -B -m hardware.formation_ablation --ablation-dir /tmp/harp --out /tmp/harp/formation.json
python3 -B -m hardware.reconstruct_harp_e2e \
  --workload-tasks /tmp/harp/workload.json --formation-report /tmp/harp/formation.json \
  --out /tmp/harp/e2e.json
python3 -B -m hardware.matched_npu_ablation \
  --workload-tasks /tmp/harp/workload.json --out /tmp/harp/matched-npu-ablation.json
```

The Full HARP evaluator applies formation exposure from Table II once. The released service-time tables cover the compiled workload shapes. See the [provenance audit](PAPER-AUDIT.md#reference-provenance) for how the tables were prepared.

The independent DiT development diagnostic remains available through `python3 -B -m hardware.harp_components --observed-dit-trace hardware/reference/dit-development-trace.json --out /tmp/harp-development.json`.

