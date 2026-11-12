# HARP hardware artifact

The active workload descriptions are in `workloads/dataflow/`. Table I supplies the reported widths, head counts and denoising steps. Each graph lists the assumed structure and any discrepancy with the upstream implementation. HARP-specific settings are in `workloads/harp/revised/` and use the model and site values shown in Figures 8 and 9(a).

The end-to-end and Matched NPU reports compute Full HARP latency with the same evaluator. It applies the released service-time table and formation exposure fractions to each compiled Combined workload. The end-to-end report uses historical H100 latencies for comparison. Matched NPU Dense, Residual, PlaneClear, and Budget use the analytical timing model documented in the [ablation guide](docs/HARP-ABLATION.md).

The workload compiler computes work reductions and supplies identical Combined tasks to the two component models. In the component report, `speedup_vs_modeled_dense` uses the generic dense proxy, and `reference_speedup_vs_matched_npu` records the historical comparison value. The [paper consistency audit](docs/PAPER-AUDIT.md#reference-provenance) describes the service-table calibration and the independent DiT development H100 timing.

Run from `AE/` and choose new output files.

```bash
python3 -B -m unittest discover -s hardware/tests
python3 -B -m hardware.baselines.reconstruct_latencies --out /tmp/baselines.json
python3 -B -m hardware --out /tmp/harp-ablation.json --tasks-in hardware/results/harp-paper-audit/workload.json
python3 -B -m hardware.reconstruct_harp_e2e --out /tmp/harp-e2e.json
python3 -B -m hardware.matched_npu_ablation --out /tmp/harp-matched-npu-ablation.json
```

Each baseline has its own entry. For example, run `python3 -B -m hardware.baselines.dstar --out /tmp/dstar.json` for the component model. Add `--kernel-latencies hardware/configs/kernel-latencies/dstar.json` to use its calibrated service times. The common comparison is `python3 -B -m hardware.baselines.component_compare --out /tmp/component-comparison.json`.

The active reports are in `results/baseline-paper-audit/` and `results/harp-paper-audit/`. See the [performance summary](results/harp-paper-audit/summary.md), [Matched NPU timing report](results/harp-paper-audit/matched-npu-ablation.json), and [HARP workload and ablation guide](docs/HARP-ABLATION.md).

