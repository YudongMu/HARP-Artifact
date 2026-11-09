# HARP Independent Rehearsal

The artifact was rehearsed on an NVIDIA H100 PCIe 80 GB host with driver 570.195.03, Docker Engine 29.1.1, and the generation and scoring images recorded in the [delivery catalog](HARP-AE-Stage6.1-Review-Catalog.md). Model weights and scoring assets were mounted read-only.

| Workflow | Rehearsal outcome |
|---|---|
| Six-model performance reconstruction | 35 tests passed. Five reconstructed reports matched the frozen references byte for byte. |
| Wan four-mode generation | A fresh one-input check completed in about 28 minutes. |
| Wan VBench | The 88 verified held-out videos were rescored in a fresh output directory in about 7 minutes. All 16 dimensions returned finite scores, and the same inputs passed the resume check. |

Peak device memory sampled at five-second intervals was 59,479 MiB for Wan. [Wan reference](../results/wan-stage4-9f-v1/heldout-scores/summary.json)
