# HARP AE submission fields — working draft

Updated 2026-11-08.

## 1. Submission title

HARP: A Hierarchical Attention and Representation Sparsity Accelerator for Diffusion Transformers

## 2. Intent to Submit

Yes, subject to final registration through the HPCA27 AE submission system.

## 3. Authors

Use the accepted paper's author list and affiliations. Enter contact information and conflicts in the submission system.

## 4. DOI URL / Artifact access

The HARP-Artifact GitHub repository contains the source, model adapters, λ tables, build recipes, and evaluation instructions. Start with its README to obtain the container images and scoring assets. The included downloader verifies the official DiT and Wan model weights against the pinned manifests.

## 5. Badges Applied For

Proposed targets are Open Research Objects, Research Objects Reviewed, and Results Reproduced. Use the badge names and criteria in the HPCA27 submission form.

## 6. Key Results to be Reproduced

1. **Generation quality on DiT and Wan.** For DiT-XL/2-512, the primary workflow compares Dense BF16, HARP-Residual, HARP-PlaneClear, and HARP-Budget / HARP using ADM FID on 100 matched inputs at 512×512 and 50 denoising steps. An extension reports absolute FID for Dense BF16 and HARP-Residual on 5,000 matched inputs to show the change in score scale. Wan2.1-T2V-14B uses 22 paired inputs at 832×480, 9 frames, and 50 steps. Across all 16 VBench dimensions, Combined scores 80.274 versus 80.370 for Dense. PlaneClear and Combined share one Wan λ table.
2. **Calibrated performance reconstruction across six workloads.** The reconstruction workflow runs HARP and the BitRed, Ditto, EXION, AdapTiV, and DSTAR models using frozen workload profiles and service-time inputs. It reconstructs denoising latency and speedup against the historical H100 reference for each workload. The six-workload geometric mean for HARP is 12.243×. The package reports the independent DiT H100 development timing separately, along with its own configuration and timing boundary.
3. **Algorithm and hardware ablation.** The reconstruction workflow reconstructs workload, ablation, and formation reports. The ablation report separates comparisons calibrated to historical references from unfitted component timing. The historical All/Budget geometric mean is 1.807×. The five frozen reports provide byte-level targets for the standard reconstruction.

SD3 Medium, FLUX.1-dev, Latte-1, and Open-Sora v1.3 are distributed as fixed source snapshots, adapter interfaces, and formal λ tables. Model execution, weights, and generation-quality validation in this artifact are scoped to DiT and Wan. The performance results cover latency and speedup under the calibrated reconstruction protocol.

## 7. Abstract

HARP combines residual-compensated attention windows with trajectory-aware removal of BF16 numeric planes. This artifact provides the HARP reference implementation and paired generation and scoring evidence for DiT and Wan. The performance workflow reconstructs the calibrated performance comparison, algorithm and hardware ablations, and formation overhead across six image and video workloads. The archive includes source snapshots, adapter interfaces, and formal λ tables for the four additional models.

## 8. Hardware Dependencies

Performance reconstruction uses Python 3. The reference generation and scoring environments were developed on NVIDIA H100 PCIe 80 GB hardware. The runbook gives the commands and measured runtimes from the H100 rehearsal.

## 9. Software Dependencies

The performance reconstruction uses the Python 3 standard library. The GPU workflow uses Docker with NVIDIA CDI. The image archive supplies `harp-ae-gpu:unified-v0.2` for generation and `harp-ae-gpu:quality-v0.4` for scoring. The repository includes the build recipes and dependency locks.

## 10. Topic

Domain-Specific Accelerators, Accelerator Architectures, and ML/Diffusion Inference.

## 11. Data Dependencies

The quality workflow uses the official DiT checkpoint and VAE, the ADM Inception graph and ImageNet 512 reference statistics, and the Wan2.1 model, VAE, text encoder, and VBench scoring assets. Source, revisions, and checksums are indexed in the [6.1 review catalog](HARP-AE-Stage6.1-Review-Catalog.md). The reconstruction workflow consumes the released dataflow, HARP profiles, component configurations, calibration inputs, and historical H100 references.

## 12. Evaluation catalog

| ID | Workflow | Existing evidence |
|---|---|---|
| Q1 | DiT paired generation and ADM FID | Four relative changes on 100 inputs and two absolute scores on 5,000 inputs |
| Q2 | Wan paired generation and 16-dimension VBench | [22-input four-mode result](../results/wan-stage4-9f-v1/heldout-scores/summary.json) |
| P1 | Six-workload calibrated denoising latency and speedup | [HARP end-to-end report](../hardware/results/harp-paper-audit/e2e.json) and [baseline report](../hardware/results/baseline-paper-audit/report.json) |
| A1 | Workload, ablation, and formation reconstruction | [Workload](../hardware/results/harp-paper-audit/workload.json), [ablation](../hardware/results/harp-paper-audit/report.json), and [formation](../hardware/results/harp-paper-audit/formation.json) |
| R1 | Independent DiT H100 development timing, provided as reference evidence | [Measurements](../hardware/results/h100-dit-v0.1/measurements.json) and [log](../hardware/results/h100-dit-v0.1/run.log) |

The [runbook](HARP-AE-Runbook.md) gives the commands and acceptance checks for Q1, Q2, P1, and A1. The [6.1 review catalog](HARP-AE-Stage6.1-Review-Catalog.md) indexes their inputs and reference evidence.

[DiT primary results](../results/dit-fid-reference/primary.json) and [DiT extension results](../results/dit-fid-reference/extension.json).
