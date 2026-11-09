# HARP Artifact Evaluation

HARP combines residual-compensated attention windows with selective removal of BF16 bit planes. The artifact reconstructs performance across six models and provides generation and quality evaluation for DiT and Wan.

| Workflow | Entry point | Reference result |
|---|---|---|
| Six-model performance reconstruction | [Quick start](#performance-reconstruction) | 12.243× geometric mean against the historical H100 latency references |
| DiT paired quality | [DiT paired FID](#dit-paired-fid) | Four configurations on 100 matched inputs, plus a 5,000-image Dense and Residual comparison |
| Wan paired quality | [Wan VBench](#wan-vbench) | **80.274 VBench** for Combined and **80.370** for Dense on the same 22 inputs |

## Environment requirements and setup

| Component | Rehearsed setup |
|---|---|
| Host tools | x86-64 Linux, Git, Bash, Python 3 (rehearsed with 3.12.3), tar and sha256sum. Asset setup also uses curl. Quality runs use flock |
| GPU | One NVIDIA H100 PCIe 80 GB, NVIDIA driver 570.195.03 |
| Container runtime | Docker Engine 29.1.1, NVIDIA Container Toolkit 1.20.1, working NVIDIA CDI and access to the Docker daemon as your user |
| Generation | `harp-ae-gpu:unified-v0.2` with Python 3.10.19, CUDA 12.1.1, PyTorch 2.5.1+cu121, Diffusers 0.32.2, Transformers 4.49.0 and FlashAttention 2.7.4.post1 |
| Scoring | `harp-ae-gpu:quality-v0.4` with Python 3.10.19 and the pinned ADM and VBench stack, including `tensorflow-cpu==2.17.1` for DiT FID |

The exact Python dependency lists are [env/unified-resolved-lock.txt](env/unified-resolved-lock.txt) for generation and [env/quality-semantic-resolved-lock.txt](env/quality-semantic-resolved-lock.txt) for scoring. The [Dockerfiles](docker/) document the build inputs. The steps below load both built images with their dependencies already installed. The [delivery catalog](submit/HARP-AE-Stage6.1-Review-Catalog.md#container-images) records the image IDs.

### 1. Get the source and assets

Clone the repository and run the commands from its root.

~~~bash
git clone https://github.com/YudongMu/HARP-Artifact.git
cd HARP-Artifact
~~~

You can now run [performance reconstruction](#performance-reconstruction) using the host Python standard library. For quality evaluation, open [Releases](https://github.com/YudongMu/HARP-Artifact/releases) and select **HARP Artifact Assets**. Its instructions give the commands to assemble the container archive in `containers/` and the scoring archives in `scoring-assets/`.

### 2. Load the images and check GPU access

~~~bash
docker version --format '{{.Server.Version}}'
nvidia-smi -L
nvidia-ctk cdi list
(cd containers && sha256sum -c docker-images.tar.gz.sha256)
docker load -i containers/docker-images.tar.gz
docker image inspect --format '{{.Id}}' \
  harp-ae-gpu:unified-v0.2 harp-ae-gpu:quality-v0.4
export HARP_GPU=0
docker run --rm --pull=never --device="nvidia.com/gpu=${HARP_GPU}" \
  harp-ae-gpu:unified-v0.2 nvidia-smi -L
~~~

Use a free GPU index from `nvidia-ctk cdi list`. For example, use `HARP_GPU=1` for `nvidia.com/gpu=1`. The final command should show the selected H100 inside the generation container. If the CDI list is empty, follow the [NVIDIA CDI setup guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html) before continuing.

### 3. Prepare assets for the workflow you want to run

From the repository root, create an asset directory on the host. Bash expands `$PWD` to the absolute path of this checkout. Use the same shell for setup and subsequent runs. The scripts mount the `weights/` and `quality/` directories into the containers.

~~~bash
export HARP_ASSETS="$PWD/harp-assets"
mkdir -p "$HARP_ASSETS/weights" "$HARP_ASSETS/quality"
~~~

**For DiT FID**, download the DiT checkpoint and VAE (about 3 GB), then extract the ADM scoring files.

~~~bash
python3 scripts/prepare_reviewer_weights.py dit --assets "$HARP_ASSETS"
(cd scoring-assets && sha256sum -c dit-fid.tar.gz.sha256)
tar -xzf scoring-assets/dit-fid.tar.gz -C "$HARP_ASSETS"
~~~

**For Wan VBench**, download the pinned Wan checkpoint, VAE and text encoder (about 69 GB), then extract the VBench scoring files. Allow at least 120 GB of free space for the archives, extracted files, weight downloads and outputs when running both quality workflows. Docker also needs free space for the loaded images on its storage filesystem.

~~~bash
python3 scripts/prepare_reviewer_weights.py wan --assets "$HARP_ASSETS"
(cd scoring-assets && sha256sum -c wan-vbench.tar.gz.sha256)
tar -xzf scoring-assets/wan-vbench.tar.gz -C "$HARP_ASSETS"
~~~

The downloader checks upstream file hashes before installing the files. Run the same command again to reuse verified files and resume partial downloads. If your H100 server has no outbound Internet access, run the downloader from the extracted code on an Internet-connected machine. Copy the resulting `weights/` directory into `$HARP_ASSETS` on the H100 host, and add `--verify-only` to the downloader command on that host. Use the same option to check an existing weight installation. See the [asset inventory](submit/HARP-AE-Stage6.1-Review-Catalog.md#asset-inventory) for the required layout and provenance.

## Performance reconstruction

Run from the directory containing this README. Choose a fresh output directory for each run.

~~~bash
bash scripts/reviewer_cpu.sh results/reviewer-performance-check
~~~

The command runs 35 tests and checks five reports against the frozen references byte for byte. Open `results/reviewer-performance-check/summary.md` for the six-model latency and speedup table. The report gives a **12.243×** geometric mean against the historical H100 references and a **1.807×** geometric mean for the historical All/Budget ablation. The [hardware guide](hardware/README.md) describes the reports and the independent DiT timing record.

## DiT paired FID

The primary experiment compares Dense BF16, HARP-Residual, HARP-PlaneClear and HARP-Budget / HARP on the same 100 class, seed and initial noise inputs. The class list samples every tenth ImageNet class. All configurations use 512×512 images, 50 denoising steps, and the same ADM evaluator and reference statistics. PlaneClear and HARP-Budget use the released DiT budget table.

~~~bash
bash scripts/reviewer_dit_fid.sh run primary 100
~~~

The primary table reports each configuration's percentage change relative to Dense BF16. The output at `results/reviewer-dit-fid-primary/fid-comparison.json` includes the underlying FID scores. The reference PlaneClear implementation takes longer to generate samples, so the four-configuration experiment uses a fixed subset.

| Configuration | FID change vs. Dense BF16 |
|---|---:|
| Dense BF16 | +0.000% |
| HARP-Residual | +0.899% |
| HARP-PlaneClear | +0.016% |
| HARP-Budget / HARP | +1.114% |

[Primary result record](results/dit-fid-reference/primary.json)

The extension compares Dense BF16 and HARP-Residual on 5,000 matched inputs, with five images per ImageNet class. Its absolute scores show how the FID scale changes when more images and classes contribute to the estimate.

~~~bash
bash scripts/reviewer_dit_fid.sh run extension 5000 dense,residual
~~~

| Configuration | ADM FID | FID change vs. Dense BF16 |
|---|---:|---:|
| Dense BF16 | 17.766 | +0.000% |
| HARP-Residual | 17.815 | +0.277% |

[Extension result record](results/dit-fid-reference/extension.json)

The output is `results/reviewer-dit-fid-extension/fid-comparison.json`. The count and comma-separated configuration list are configurable. For example, `bash scripts/reviewer_dit_fid.sh run larger 1000 dense,residual,planeclear,combined` covers every ImageNet class once. Use a fresh run name when changing settings, or repeat the same command to resume. The `generate` and `score` phases can run separately. To remove the generated results for a finished run, use `bash scripts/clear_dit_fid.sh primary`.

## Wan VBench

Use 22 paired inputs at 832×480, 9 frames and 50 denoising steps. PlaneClear and Combined share one λ table.

~~~bash
bash scripts/reviewer_quality.sh wan development dev-check
~~~

The expected 16-dimension VBench totals are **80.370 / 80.153 / 80.430 / 80.274** in Dense, Residual, PlaneClear and Combined order. The new scores are saved to `results/reviewer-wan-dev-check/heldout-scores/summary.json`. [Reference scoring record](results/wan-stage4-9f-v1/heldout-scores/summary.json)

For a one-input Wan generation check, run `bash scripts/reviewer_quality.sh wan smoke check`. The [reviewer runbook](submit/HARP-AE-Reviewer-Runbook.md) gives the output paths and how to resume Wan scoring.

## Package map

| Directory | Contents |
|---|---|
| `containers/` | Docker image archive assembled from Release assets |
| `scoring-assets/` | DiT FID and Wan VBench archives assembled from Release assets |
| [submit/](submit/README.md) | Reviewer runbook, delivery catalog and evaluation evidence |
| [scripts/](scripts/) | Performance, generation and scoring entry points |
| [hardware/](hardware/) | Workload profiles, performance models, tests and frozen reports |
| [harp/](harp/) and [quality/](quality/) | HARP reference code, model adapters and quality scoring |
| [src/](src/) | Model source snapshots and one λ table per model |
| [manifests/](manifests/) | Asset and source versions and checksums |
| [results/](results/) | Quality references, frozen Wan inputs and new workflow outputs |
