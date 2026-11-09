# HARP Artifact Evaluation

HARP combines residual-compensated attention windows with selective removal of BF16 bit planes. To keep the quality evaluation practical, we use DiT and Wan as representative workloads for class-conditioned image generation and text-to-video generation, respectively. The repository also provides HARP adapters and λ tables for SD3 Medium, FLUX.1-dev, Latte-1, and Open-Sora v1.3. The performance reconstruction covers all six models.

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

You can now run [performance reconstruction](#performance-reconstruction) using the host Python standard library. For quality evaluation, open [Releases](https://github.com/YudongMu/HARP-Artifact/releases) and select **HARP Artifact Assets**. Create the download directories from the repository root.

~~~bash
mkdir -p containers scoring-assets
~~~

Save the three `docker-images.tar.gz.part-*` files and `docker-images.tar.gz.sha256` in `containers/`. For DiT, save `dit-fid.tar.gz` and `dit-fid.tar.gz.sha256` in `scoring-assets/`. For Wan, save the five `wan-vbench.tar.gz.part-*` files and `wan-vbench.tar.gz.sha256` in `scoring-assets/`.

### 2. Load the images and check GPU access

~~~bash
docker version --format '{{.Server.Version}}'
nvidia-smi -L
nvidia-ctk cdi list
(cd containers && cat docker-images.tar.gz.part-{01..03} > docker-images.tar.gz &&
  sha256sum -c docker-images.tar.gz.sha256) &&
docker load -i containers/docker-images.tar.gz
docker image inspect --format '{{.Id}}' \
  harp-ae-gpu:unified-v0.2 harp-ae-gpu:quality-v0.4
export HARP_GPU=0
docker run --rm --pull=never --device="nvidia.com/gpu=${HARP_GPU}" \
  harp-ae-gpu:unified-v0.2 nvidia-smi -L
~~~

Use a free GPU index from `nvidia-ctk cdi list`. For example, use `HARP_GPU=1` for `nvidia.com/gpu=1`. The final command should show the selected H100 inside the generation container. If the CDI list is empty, follow the [NVIDIA CDI setup guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html) before continuing.

### 3. Prepare assets for the workflow you want to run

From the repository root, copy the entire block for your chosen workflow. Use the same shell for setup and subsequent runs. The Python command downloads and verifies the model weights. The `tar` command installs the scoring assets. The parentheses run the archive checks in `scoring-assets/` in a Bash subshell. The remaining commands run from the repository root.

**For DiT FID**, `prepare_weights.py dit` downloads the DiT checkpoint and VAE (about 3 GB).

~~~bash
export HARP_ASSETS="${HARP_ASSETS:-$PWD/harp-assets}"
mkdir -p "$HARP_ASSETS/weights" "$HARP_ASSETS/quality" &&
(cd scoring-assets && sha256sum -c dit-fid.tar.gz.sha256) &&
python3 scripts/prepare_weights.py dit --assets "$HARP_ASSETS" &&
tar -xzf scoring-assets/dit-fid.tar.gz -C "$HARP_ASSETS"
~~~

**For Wan VBench**, `prepare_weights.py wan` downloads the pinned Wan checkpoint, VAE and text encoder (about 69 GB). Allow at least 120 GB of free space for the archives, extracted files, weight downloads and outputs when running both quality workflows. Docker also needs free space for the loaded images on its storage filesystem.

~~~bash
export HARP_ASSETS="${HARP_ASSETS:-$PWD/harp-assets}"
mkdir -p "$HARP_ASSETS/weights" "$HARP_ASSETS/quality" &&
(cd scoring-assets && cat wan-vbench.tar.gz.part-{01..05} > wan-vbench.tar.gz &&
  sha256sum -c wan-vbench.tar.gz.sha256) &&
python3 scripts/prepare_weights.py wan --assets "$HARP_ASSETS" &&
tar -xzf scoring-assets/wan-vbench.tar.gz -C "$HARP_ASSETS"
~~~

Run the same command again to reuse verified files and resume partial downloads. If your H100 server has no outbound Internet access, run the Python download command from the checkout on an Internet-connected machine and copy the resulting `weights/` directory into `$HARP_ASSETS` on the H100 host. Verify the installed weights for your selected workflow with the corresponding command.

~~~bash
python3 scripts/prepare_weights.py dit --assets "$HARP_ASSETS" --verify-only
python3 scripts/prepare_weights.py wan --assets "$HARP_ASSETS" --verify-only
~~~

The [asset inventory](submit/HARP-AE-Stage6.1-Review-Catalog.md#asset-inventory) records the required layout and provenance.

## Performance reconstruction

Run from the directory containing this README. Choose a fresh output directory for each run. The complete reconstruction and its checks take a few seconds on the reference host.

~~~bash
bash scripts/reconstruct_performance.sh results/performance-check
~~~

The command compiles the workloads, runs the timing models, and checks five generated reports against the frozen outputs byte for byte. Open `results/performance-check/summary.md` for the end-to-end comparison, work reductions, and ablation speedups. The end-to-end geometric mean is **12.243×** relative to the historical H100 Dense latencies.

The ablation replay computes geometric means of **1.335×, 3.378×, 3.697×, and 6.838×** for HARP-Residual, HARP-PlaneClear, HARP-Budget (Combined), and HARP-All, respectively, relative to the historical analytical Matched NPU Dense baseline. HARP-All shares its latency with the frozen end-to-end reconstruction. The run writes the underlying timings to `matched-npu-ablation.json`. See the [reference summary](hardware/results/harp-paper-audit/summary.md) and [hardware guide](hardware/README.md).

## DiT paired FID

The primary experiment compares Dense BF16, HARP-Residual, HARP-PlaneClear and HARP-Budget / HARP on the same 100 class, seed and initial noise inputs. The class list samples every tenth ImageNet class. All configurations use 512×512 images, 50 denoising steps, and the same ADM evaluator and reference statistics. PlaneClear and HARP-Budget use the released DiT budget table.

Allow about **3 hours on one H100** for generation and scoring at the default batch size of 2.

~~~bash
bash scripts/run_dit_fid.sh run primary 100
~~~

The table reports the percentage changes measured on H100 relative to Dense BF16. The output at `results/dit-fid-primary/fid-comparison.json` includes the underlying FID scores. The reference PlaneClear implementation takes longer to generate samples, so the four-configuration experiment uses a fixed subset.

| Configuration | FID change vs. Dense BF16 |
|---|---:|
| Dense BF16 | +0.000% |
| HARP-Residual | +0.899% |
| HARP-PlaneClear | +0.016% |
| HARP-Budget / HARP | +1.114% |

[Primary result record](results/dit-fid-reference/primary.json)

The extension compares Dense BF16 and HARP-Residual on 5,000 matched inputs, with five images per ImageNet class. Its absolute scores show how the FID scale changes when more images and classes contribute to the estimate.

Allow about **8 hours on one H100** at batch size 2. Running generation for the two modes on separate H100s takes about **6.5 hours** including scoring.

~~~bash
bash scripts/run_dit_fid.sh run extension 5000 dense,residual
~~~

| Configuration | ADM FID | FID change vs. Dense BF16 |
|---|---:|---:|
| Dense BF16 | 17.766 | +0.000% |
| HARP-Residual | 17.815 | +0.277% |

[Extension result record](results/dit-fid-reference/extension.json)

The output is `results/dit-fid-extension/fid-comparison.json`. The count and comma-separated configuration list are configurable. For example, `bash scripts/run_dit_fid.sh run larger 1000 dense,residual,planeclear,combined` covers every ImageNet class once. Use a fresh run name when changing settings, or repeat the same command to resume. The `generate` and `score` phases can run separately. To remove the generated results for a finished run, use `bash scripts/clear_dit_fid.sh primary`.

## Wan VBench

Use 22 paired inputs at 832×480, 9 frames and 50 denoising steps. PlaneClear and Combined share one λ table.

Allow about **7 hours on one H100** for the full run with the assets already prepared. The 88 generation records total about 6.5 hours, and a scoring pass takes about 7–12 minutes.

~~~bash
bash scripts/run_quality.sh wan development dev-check
~~~

The 22-input run on H100 yielded 16-dimension VBench totals of **80.370 / 80.153 / 80.430 / 80.274** in Dense, Residual, PlaneClear and Combined order. Compare your new scores at `results/wan-dev-check/heldout-scores/summary.json` with the [recorded scores](results/wan-stage4-9f-v1/heldout-scores/summary.json).

For a one-input Wan generation check, run `bash scripts/run_quality.sh wan smoke check`. The [runbook](submit/HARP-AE-Runbook.md) gives the output paths and how to resume Wan scoring.

## Package map

| Directory | Contents |
|---|---|
| `containers/` | Docker image archive assembled from Release assets |
| `scoring-assets/` | DiT FID and Wan VBench archives assembled from Release assets |
| [submit/](submit/README.md) | Runbook, delivery catalog and evaluation evidence |
| [scripts/](scripts/) | Performance, generation and scoring entry points |
| [hardware/](hardware/) | Workload profiles, performance models, tests and frozen reports |
| [harp/](harp/) and [quality/](quality/) | HARP reference code, model adapters and quality scoring |
| [src/](src/) | Model source snapshots and one λ table per model |
| [manifests/](manifests/) | Asset and source versions and checksums |
| [results/](results/) | Quality references, frozen Wan inputs and new workflow outputs |
