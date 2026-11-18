# Fixed component timing

The original HARP simulator was calibrated against RTL according to the authors. For replay, each accelerator has one fixed `cycle_scale` in its own [component configuration](../configs/baseline-components/). HARP reads [harp.json](../configs/baseline-components/harp.json). Each accelerator uses its scale for all six workloads.

Run from the repository root with a new output path.

```bash
python3 -B -m hardware.component_study --fixed --out results/component-rtl/evaluation.json
```

The evaluator compiles the workload graph with the [updated profiles](../workloads/harp/component-profile-update/), schedules the resulting work on the configured resources, and computes `effective_cycles = ceil(scheduled_cycles * cycle_scale)`. Latency in milliseconds is `1000 * effective_cycles / clock_hz`. Dividing the configured H100 latency by that latency gives the reported speedup. The report includes the profile hash and a fingerprint of the six cycle scales. The [source manifest](../../manifests/source-tree.sha256) records the hash of each configuration file.
