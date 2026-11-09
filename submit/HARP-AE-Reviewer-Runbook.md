# HARP Reviewer Runbook

Run the commands from the artifact root. The [README](../README.md#environment-requirements-and-setup) lists the host, Docker, CDI and asset setup.

## Performance and ablation

~~~bash
bash scripts/reviewer_cpu.sh results/reviewer-performance-check
~~~

Expect 35 passing tests and byte-for-byte agreement for five JSON files: `baseline.json`, `workload.json`, `report.json`, `formation.json` and `e2e.json`. The output directory also contains `summary.md` and `reconstruction.log`. The performance and ablation denominators are described in the [hardware guide](../hardware/README.md).

## DiT paired FID

~~~bash
bash scripts/reviewer_dit_fid.sh run primary 100
bash scripts/reviewer_dit_fid.sh run extension 5000 dense,residual
~~~

The primary run compares Dense BF16, HARP-Residual, HARP-PlaneClear and HARP-Budget / HARP on the same 100 inputs. The extension compares Dense BF16 and HARP-Residual on 5,000 matched inputs. Each run writes per-mode FID and the Dense-relative changes to `results/reviewer-dit-fid-<run>/fid-comparison.json`. Repeat the command with the same run name to resume. Use `generate` and `score` to split the phases. Run `bash scripts/clear_dit_fid.sh primary` to remove the primary run's generated results.

## Wan paired VBench

~~~bash
bash scripts/reviewer_quality.sh wan development dev-check
~~~

The script generates Dense, Residual, PlaneClear and Combined for 22 paired inputs and scores all 16 VBench dimensions. Inspect `results/reviewer-wan-dev-check/heldout-scores/summary.json` and `paired-uncertainty.json`. The totals are **80.370 / 80.153 / 80.430 / 80.274** in mode order. The shared Wan λ table is under `src/Wan2.1/harp/`. The script reads the frozen input and selection records from `results/wan-stage4-9f-v1/`. [Wan scoring reference](../results/wan-stage4-9f-v1/heldout-scores/summary.json)

Use `generate` and `score` with the same run name to split or resume the workflow. Generation and scoring logs are written under the run directory. The scoring command returns a failure status if a dimension is undefined or fails. The H100 rehearsal generated one fresh four-mode sample in about 28 minutes and scored the 88 frozen held-out videos in about 7 minutes.

## One-input generation check

~~~bash
bash scripts/reviewer_quality.sh wan smoke check
~~~

This command writes four paired Wan generation receipts under `results/source-budget-wan-smoke-check/`.

[DiT primary results](../results/dit-fid-reference/primary.json) and [DiT extension results](../results/dit-fid-reference/extension.json).
