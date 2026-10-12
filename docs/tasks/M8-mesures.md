# M8 — Measurements and report

Goal: figures comparable with those of the base (§10.4), with an explicit measurement
scope.

### [ ] T8.1 — Performance, power, resources
- **Spec**: §10.4, §10.5 step 6 · **Depends on**: T7.4 · **Size**: M
- **Deliverables**: row(s) in `results/benchmarks.csv`; protocol in
  `results/protocole.md`
- **Acceptance**: FPS and latency (mean and p99 over ≥ 1,000 images, separating
  preprocessing, accelerator and post-processing); GOPS = 2 × MACs / time; power with its
  **scope stated** (chip estimated by Vivado *and* board measured if possible); LUT/DSP/BRAM
  from the implementation report
- **Notes**: efficiency = theoretical cycles ($\text{MACs}/(T_mT_n)$) / measured cycles
- **Done**: `sw/app/yolo_bench` (time per image and per stage: preprocessing, DDR copy,
  accelerator, post-processing; mean and p99; `--images` JPEG or `--inputs` int8;
  `--power`: INA260 of the SOM through hwmon, idle then under load; JSONL detections;
  batches `--start/--count`). `tools/perf_model.py`: cycle model **equal cycle for cycle**
  to the C-sim counters (`make perf-model`), optimisation scenarios.
  `tools/bench_report.py`: `results/benchmarks.csv` (8 published rows of §10.4 + this
  work), `results/mesures.md` (stages, efficiency per layer), reading of
  `utilization.rpt` / `power.rpt`. `results/protocole.md`: scopes (stages, chip power
  estimated / SOM measured), conditions, board commands. Without a board: a "C-sim
  projection" row (Tiny-YOLOv2: 206.5 ms accelerator, 4.8 img/s, 33.8 GOPS, efficiency
  11.0 %).
- **Remaining**: measurements on the KV260 (≥ 1,000 images, SOM power); LUT/DSP/BRAM and
  chip power from `make vivado-build`.

### [ ] T8.2 — mAP at the three stages
- **Spec**: §11 · **Depends on**: T8.1, T4.5 · **Size**: S
- **Deliverables**: `results/map_stades.md`
- **Acceptance**: VOC2007 test mAP in float, in Python integer and on FPGA; the last two
  must be **identical** (bit-exact)
- **Done**: `tools/make_inputs.py` (int8 inputs of VOC2007 test, PIL `stretch`, read by the
  board instead of the JPEGs: stb ≠ PIL), `tools/eval_quant.py --save-dets` (integer
  detections per image), `tools/map_stades.py` → `results/map_stades.md` (image-by-image
  equality + mAP). Float 56.30, integer 55.66 (= T3.4, T4.5). FPGA stage in C-sim
  (`make bench-sim`, ARM driver + C-sim kernel, 22 processes, ≈ 40 min): **mAP 55.66,
  4,952 / 4,952 images identical** to the integer model.
- **Remaining**: the same `yolo_bench --inputs` on the KV260 (`board/dets_*.jsonl`).
- **PC profiles**: simulated chain by tiers [T12.9](M12-profils-pc.md#-t129--chaîne-carte-simulée), rehearsal of the protocol [T12.10](M12-profils-pc.md#-t1210--répétition-générale-du-protocole-carte)

### [ ] T8.3 — Comparative report
- **Spec**: §10.4 · **Depends on**: T8.2 · **Size**: M
- **Deliverables**: `results/rapport.md`
- **Acceptance**: comparison with 2023-zhai, 2024-zhang, 2025-kim and 2026-fata, flagging
  what is not comparable (dataset, pruning, frequency, power scope); optimisation
  directions costed by the roofline
- **Done**: `results/rapport.md`: comparison with the projection (until the board is
  available), "not comparable" column, directions costed by `tools/perf_model.py` (64/128-bit
  ports, valid channels, parallel requantization) and set against the roofline.
- **Remaining**: replace the projection with the board measurements.
