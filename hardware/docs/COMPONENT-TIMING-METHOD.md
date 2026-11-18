# Fixed component timing

The original HARP simulator was calibrated against RTL according to the authors. For replay, the evaluator reads one fixed cycle scale per simulator from [component-model-timing.json](../configs/component-model-timing.json). HARP uses `0.016346771973846656` across all six workloads.

Run from the repository root with a new output path.

```bash
python3 -B -m hardware.component_study --fixed --out results/component-rtl/evaluation.json
```

The evaluator compiles the workload graph with the [updated profiles](../workloads/harp/component-profile-update/), schedules the resulting work on the configured resources, and computes `effective_cycles = ceil(scheduled_cycles * cycle_scale)`. Latency in milliseconds is `1000 * effective_cycles / clock_hz`. Dividing the configured H100 latency by that latency gives the reported speedup. The report includes the profile and parameter hashes.
