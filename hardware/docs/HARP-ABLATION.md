# HARP workload and ablation

HARP combines the six Table I constrained operator graphs with model and site profiles from Figures 8 and 9(a). The profiles retain a shared fallback and within-plane activity estimate. FLUX packed linear operators use QKV and output projection site proxies. The full assumptions and evidence are described in [Paper consistency audit](PAPER-AUDIT.md).

The compiler separates dense attention, refresh, replay, selected work, active and natural plane skips, fallback MACs and formation. Hardware comparisons replay the same Combined task list with matched modeled compute, capacity and HBM. The component run computes speedup relative to its generic dense proxy. The archived report also records the historical Matched NPU comparison values.

The [Matched NPU replay](../matched_npu_ablation.py) calculates Dense, Residual, PlaneClear, and Budget latency using the historical analytical cycle model and [its workload parameters](../workloads/harp/matched-npu.json). The workload parameters come from the original synthetic-simulated tile and region profiles. This path preserves the historical work proxy and its 262.1 TFLOP/s throughput parameter. The separate revised component model uses the 32.768 TFLOP/s dense BF16 peak. Both reports compute HARP-All latency using the [shared evaluator](../harp_latency.py) with the compiled Combined tasks, released service times, and formation exposure fractions. Matched NPU speedups are calculated from the resulting latencies.

Run from `AE/` to generate fresh workload, component, formation, end-to-end and Matched NPU reports.

```bash
bash scripts/reconstruct_performance.sh /tmp/harp-performance
```

The Full HARP evaluator applies the formation exposure in `configs/performance-inputs.json` once. Formation report keys with `table_ii` retain their names for compatibility with existing consumers. The released service-time tables cover the compiled workload shapes. See the [provenance audit](PAPER-AUDIT.md#reference-provenance) for how the tables and inputs were prepared.

The independent DiT development diagnostic remains available through `python3 -B -m hardware.harp_components --observed-dit-trace hardware/reference/dit-development-trace.json --out /tmp/harp-development.json`.

