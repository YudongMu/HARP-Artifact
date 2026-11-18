# HARP paper and artifact cross-check

This audit maps the performance reconstruction to the [revised paper](../../HARP-326-revised.pdf). The paper's workload and hardware fields are recorded in [harp-revised.json](../reference/harp-revised.json).

| Paper location | Artifact evidence |
|---|---|
| Abstract, Sections I and VIII | The calibrated latency reconstruction yields geometric means of 12.243× against H100 and 3.206× against DSTAR. Generation quality is evaluated separately. |
| Sections II and III, Figures 1–2 | The dataflow represents projections, attention, feed-forward layers, modulation, and tensor dependencies. The published workload statistics supply the reconstruction inputs. |
| Section IV, Figures 3–4 | The model uses 4×4×32 tiles and a refresh interval of five steps. The workload encodes the paper's aggregate region-selection and bit-plane statistics and distinguishes refresh, replay and steady-state selection. |
| Section V, Figure 5 | The model uses the reported 2 GHz, 8,192 BF16 MAC/cycle, 96 MiB SRAM and 2.4 TB/s configuration. Component replay approximates buffering and resource timing. |
| Section VI-A, Table I | The six workload profiles match the reported widths, head counts and denoising steps. FLUX uses 4,096 width and 64 heads. Latte uses 1,408 width and 16 heads. The profiles document the graph and conditioning approximations. |
| Section VI-B, Figure 6 and Table II | The calibrated reconstruction matches the six latency targets at the precision shown in Table II. The formation fractions are calibration inputs. The reconstructed HARP/BitRed latency ratio is 5.47×. |
| Section VI-C, Figure 7 | The replay yields geometric means of 1.335×, 3.378×, 3.697×, and 6.838× for Residual, PlaneClear, Budget, and All against its Matched NPU Dense baseline. Budget is 2.3% lower than the figure aggregate, with differences of 6.1% for DiT and 7.3% for FLUX. Residual, PlaneClear, and Budget use the historical analytical model. Both reports compute All through the shared Full HARP evaluator. The separate component run uses a generic dense proxy. |
| Section VI-D, Figures 8–9 | The six region-selection input rates average 32.66%. The reconstruction uses the 42 model/site entries displayed in Figure 8. |
| Sections VI-E and VI-F | The reconstruction reports latency and speedup. The paper provides the area, power and sensitivity results. |

## Reference provenance

- The historical end-to-end workbook matches all 84 model/latency/speedup rows of the development CSV labeled `synthetic-simulated`. During artifact reconstruction, the historical H100 latency and speedup references supplied aggregate latency targets for fitting the HARP and baseline service-time tables. The runtime [performance configuration](../configs/performance-inputs.json) preserves the H100 denominators from that workbook and the formation exposure fractions from Table II. The reported latencies are model estimates under that calibration.
- The standard workflow evaluates the released configuration, service-time tables and workload profiles. With `--verify`, it runs the paper-alignment tests and compares computed metrics against the archived reports after evaluation. The [archived HARP service-table generator](https://github.com/YudongMu/HARP-Artifact/blob/56387b79b1e238dc799f05c02631da9d93abf809/hardware/calibrate_service_times.py) documents how the release table was built. For a baseline table in this checkout, run `python3 -B -m hardware.calibrate_service_times dstar --out /tmp/dstar-service-times.json`. The ablation workbook matches 34 of 42 rows in the development tables using the same model and denominator and is used for comparison.
- Historical and unfitted component results are reported separately. The independent DiT H100 median is **987.967 ms** under the development timing protocol. The DiT value of **1067.983 ms** in Table II is a historical calibration target. [DiT timing record](../results/h100-dit-v0.1/measurements.json)
- The reference workbook and paper use different energy-efficiency values. The calibrated entry uses latency and speedup targets. Figure 8 cell averages are 21.55% and 51.81%, while the prose reports 21.46% and 50.68%. The reconstruction uses the displayed cells.
