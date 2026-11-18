# HARP hardware artifact

Run `python3 -B -m hardware.component_study --fixed --out results/component-rtl/evaluation.json` from the repository root for the six-workload component simulation. The [timing method](docs/COMPONENT-TIMING-METHOD.md) describes the fixed parameters and workload inputs.

## Archived release reconstruction

The active workload descriptions are in `workloads/dataflow/`. Table I supplies the reported widths, head counts and denoising steps. Each graph lists the assumed structure and any discrepancy with the upstream implementation. HARP-specific settings are in `workloads/harp/revised/` and use the model and site values shown in Figures 8 and 9(a).

The end-to-end and Matched NPU reports compute Full HARP latency with the same evaluator. It applies the released service-time table and configured formation exposure to each compiled Combined workload. The [performance inputs](configs/performance-inputs.json) specify formation exposure and the H100 latency denominators by workload. Matched NPU Dense, Residual, PlaneClear, and Budget use the analytical timing model documented in the [ablation guide](docs/HARP-ABLATION.md).

The workload compiler computes work reductions and supplies identical Combined tasks to the two component models. In the component report, `speedup_vs_modeled_dense` uses the generic dense proxy. The [paper consistency audit](docs/PAPER-AUDIT.md#reference-provenance) describes the service-table calibration and the independent DiT development H100 timing.

Run from `AE/` and choose new output files.

```bash
bash scripts/reconstruct_performance.sh /tmp/harp-performance
bash scripts/reconstruct_performance.sh --verify /tmp/harp-verified
```

Each baseline has its own entry. For example, run `python3 -B -m hardware.baselines.dstar --out /tmp/dstar.json` for the component model. Add `--kernel-latencies hardware/configs/kernel-latencies/dstar.json` to use its calibrated service times. The separate `hardware.baselines.component_compare` diagnostic compares component estimates with historical targets.

The archived reports for verification are in `results/baseline-paper-audit/` and `results/harp-paper-audit/`. See the [reference summary](results/harp-paper-audit/summary.md), [Matched NPU timing report](results/harp-paper-audit/matched-npu-ablation.json), and [HARP workload and ablation guide](docs/HARP-ABLATION.md).

