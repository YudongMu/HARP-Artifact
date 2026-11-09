# HARP Artifact Delivery Catalog

| ID | Workflow | Reference evidence |
|---|---|---|
| Q1 | DiT paired ADM FID, 100 inputs for four configurations and 5,000 inputs for Dense and Residual | Per-configuration scores and Dense-relative changes in each run's `fid-comparison.json` |
| Q2 | Wan paired quality, 22 inputs per mode | [Four-mode VBench and all 16 dimensions](../results/wan-stage4-9f-v1/heldout-scores/summary.json) |
| P1 | Six-model calibrated performance reconstruction | [End-to-end latency and speedup](../hardware/results/harp-paper-audit/e2e.json) and [baseline comparison](../hardware/results/baseline-paper-audit/report.json) |
| A1 | Algorithm and hardware ablation and formation | [Workload](../hardware/results/harp-paper-audit/workload.json), [ablation](../hardware/results/harp-paper-audit/report.json), [formation](../hardware/results/harp-paper-audit/formation.json) |
| R1 | Independent DiT H100 development timing | [Measurements](../hardware/results/h100-dit-v0.1/measurements.json) and [log](../hardware/results/h100-dit-v0.1/run.log) |

The [reviewer runbook](HARP-AE-Reviewer-Runbook.md) maps each workflow to its commands and outputs. The six model source snapshots carry one λ table per model. The four models outside the DiT/Wan quality workflows include their source snapshots, adapter interfaces and λ tables.

## Container images

| Role | Local tag | Rehearsed image ID |
|---|---|---|
| Generation | `harp-ae-gpu:unified-v0.2` | `sha256:5c8a1141d32b78a6c4ba551014503ca57511c477effbeea49f25402f47c8e51a` |
| Scoring | `harp-ae-gpu:quality-v0.4` | `sha256:e3193d4e976c6fb81a926bf8901e5c650f2784594a02eb363a1150cc83890868` |

The assembled archive `containers/docker-images.tar.gz` contains both Linux amd64 images and restores both tags with `docker load`. The [Docker recipes](../docker/) and the exact [generation](../env/unified-resolved-lock.txt) and [scoring](../env/quality-semantic-resolved-lock.txt) dependency lists document the build environment.

## Asset inventory

The asset inventory paths are relative to `HARP_ASSETS`. Follow the [asset setup](../README.md#3-prepare-assets-for-the-workflow-you-want-to-run) to download verified DiT or Wan weights and unpack the matching archive from `scoring-assets/`.

| Scoring archive | Contents |
|---|---|
| `scoring-assets/dit-fid.tar.gz` | ADM scorer source, Inception graph and ImageNet 512 reference moments |
| `scoring-assets/wan-vbench.tar.gz` | Pinned VBench source and scoring dependencies, including the deployed 12-checkpoint inventory |

| Asset | Required path | Source and verification |
|---|---|---|
| DiT-XL/2-512 checkpoint | `weights/dit/DiT-XL-2-512x512.pt` | [Upstream checkpoint](https://github.com/facebookresearch/DiT#pre-trained-dit-checkpoints), [SHA-256](../manifests/dit-checkpoint.sha256) |
| DiT VAE configuration and safetensors | `weights/dit-vae/` | [Pinned upstream revision](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/31f26fdeee1355a5c34592e401dd41e45d25a493), [SHA-256](../manifests/dit-vae.sha256) |
| Wan transformer shards, VAE and text encoder | `weights/wan2.1-t2v-14b/` | [Pinned upstream revision](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/a064a6c71f5be440641209c07bf2a5ce7a2ff5e4), [required file inventory and hashes](../manifests/wan-download-plan.json) |
| DiT scoring | `quality/source/guided-diffusion/` and `quality/fid/` | [Scoring source](../assets/quality/source-manifest.json), [Inception graph](../assets/quality/fid/inception-graph.json), [ImageNet reference moments](../assets/quality/fid/imagenet512-moments.json) |
| Wan scoring | `quality/` | [VBench source](../assets/quality/source-manifest.json), [quality assets](../manifests/quality-scorer-assets.json), [semantic assets](../assets/quality/wan-semantic-assets.json) and [deployed inventory](../assets/quality/wan-semantic-deployed-inventory.json) |

The Wan verifier checks the exact checkpoint inventory. Keep download caches outside the Wan weight directory. For the DiT SHA-256 files, match the listed hashes to the corresponding filenames under your asset directory. Use the licensing terms that accompany each upstream asset. The package retains the [DiT license](../src/dit-original/LICENSE.txt), [Wan license](../src/Wan2.1/LICENSE.txt), and [VBench license](../assets/quality/source/VBench/LICENSE).

## Frozen budget tables

| Model | Table |
|---|---|
| DiT | [lambda.json](../src/dit-original/harp/lambda.json) |
| Wan | [lambda.json](../src/Wan2.1/harp/lambda.json) |
| SD3 | [lambda.json](../src/sd3-medium/harp/lambda.json) |
| FLUX | [lambda.json](../src/flux-dev/harp/lambda.json) |
| Latte | [lambda.json](../src/latte-1/harp/lambda.json) |
| Open-Sora | [lambda.json](../src/open-sora-v1.3/harp/lambda.json) |

The [package manifest](../manifests/rehearsal-package.sha256.json) lists the checksums of the budget tables, code and frozen records.

[DiT primary results](../results/dit-fid-reference/primary.json) and [DiT extension results](../results/dit-fid-reference/extension.json).
