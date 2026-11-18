# Updated component workload profiles

The six trial profiles are in hardware/workloads/harp/component-profile-update.
Region retention and within-plane active fractions use the supplied values.
Each supplied final skip range is applied by mapping the previous site values
affinely onto its endpoints. Interior site values are derived from this mapping.
Natural skip fractions describe planes skipped before active clearing. The model uses
them in refresh-phase attention and to count natural and active plane eliminations.
Wan inherits the original site values of 13% to 25%. The supplied final skip range
describes planes skipped after active clearing.

[evaluation.json](evaluation.json) records the effective cycles, latency and speedup for each workload using the fixed timing parameters.

To replay this evaluation, run from the repository root with a new output path.

```bash
python3 -B -m hardware.component_study --fixed --out results/component-rtl/evaluation.json
```

See the [timing method](../../hardware/docs/COMPONENT-TIMING-METHOD.md) for the inputs and cycle calculation.
