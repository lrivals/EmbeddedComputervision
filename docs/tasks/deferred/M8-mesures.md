# M8 — Measurements and report: deferred items

Parent: [M8-mesures.md](../M8-mesures.md). On the PC, the tools and the "C-sim projection"
row exist and the FPGA stage in C-sim equals the integer model (4,952 / 4,952 images). What
follows needs the board, with the scopes of `results/protocole.md`. Index:
[README.md](README.md).

### [ ] T8.1 — Performance, power and resources measured
- **Condition**: Board, Vivado · **Needs first**: T7.4, T7.1
- **Done on the PC**: `yolo_bench` (stages, mean, p99, `--power` INA260),
  `tools/perf_model.py`, `tools/bench_report.py`; projection row 206.5 ms, 4.8 img/s.
- **To do**: measurements on the KV260 (≥ 1,000 images, SOM power); LUT/DSP/BRAM and chip
  power from `make vivado-build`.
- **Unlocks**: T8.2, T10.5, T10.6, T20.9, T30.2.

### [ ] T8.2 — FPGA stage of the mAP on the board
- **Condition**: Board · **Needs first**: T8.1
- **Done on the PC**: float 56.30, integer 55.66, C-sim stage 55.66 (`make bench-sim`).
- **To do**: the same `yolo_bench --inputs` on the KV260 (`board/dets_*.jsonl`), then
  `tools/map_stades.py`.
- **Acceptance**: board detections identical to the integer model, image by image.

### [ ] T8.3 — Report on measurements
- **Condition**: Board · **Needs first**: T8.2
- **Done on the PC**: `results/rapport.md` written against the projection.
- **To do**: replace the projection with the board measurements.
- **Unlocks**: T17.19.
