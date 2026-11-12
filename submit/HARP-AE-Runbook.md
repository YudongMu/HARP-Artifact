# HARP Runbook

Run the commands from the artifact root. The [README](../README.md#environment-requirements-and-setup) lists the host, Docker, CDI and asset setup.

## Performance and ablation

~~~bash
bash scripts/reconstruct_performance.sh results/performance-check
~~~

Expect passing tests and byte-for-byte agreement with the six reference reports. Open `summary.md` for the end-to-end comparison against H100, work reductions, and the four ablation speedups relative to Matched NPU Dense. Both comparisons compute Full HARP latency with the same evaluator. The underlying Matched NPU timings are in `matched-npu-ablation.json`. This run takes a few seconds on the reference host. See the [hardware guide](../hardware/README.md) for the timing inputs and calibration.

## DiT paired FID

~~~bash
bash scripts/run_dit_fid.sh run primary 100
bash scripts/run_dit_fid.sh run extension 5000 dense,residual
~~~

The primary run compares Dense BF16, HARP-Residual, HARP-PlaneClear and HARP-Budget / HARP on the same 100 inputs. The extension compares Dense BF16 and HARP-Residual on 5,000 matched inputs. Each run writes per-mode FID and the Dense-relative changes to `results/dit-fid-<run>/fid-comparison.json`. Repeat the command with the same run name to resume. Use `generate` and `score` to split the phases. Run `bash scripts/clear_dit_fid.sh primary` to remove the primary run's generated results.

## Wan paired VBench

~~~bash
bash scripts/run_quality.sh wan development dev-check
~~~

The script generates Dense, Residual, PlaneClear and Combined for 22 paired inputs and scores all 16 VBench dimensions. Inspect `results/wan-dev-check/heldout-scores/summary.json` and `paired-uncertainty.json`. The recorded H100 run yielded totals of **80.370 / 80.153 / 80.430 / 80.274** in mode order. The shared Wan λ table is under `src/Wan2.1/harp/`. The script reads the frozen input and selection records from `results/wan-stage4-9f-v1/`. [Wan scoring reference](../results/wan-stage4-9f-v1/heldout-scores/summary.json)

Use `generate` and `score` with the same run name to split or resume the workflow. Generation and scoring logs are written under the run directory. The scoring command returns a failure status if a dimension is undefined or fails. The H100 rehearsal generated one fresh four-mode sample in about 28 minutes and scored the 88 frozen held-out videos in about 7 minutes.

## One-input generation check

~~~bash
bash scripts/run_quality.sh wan smoke check
~~~

This command writes four paired Wan generation receipts under `results/source-budget-wan-smoke-check/`.

[DiT primary results](../results/dit-fid-reference/primary.json) and [DiT extension results](../results/dit-fid-reference/extension.json).
