# M7 — SoC integration: deferred items

Parent: [M7-integration-soc.md](../M7-integration-soc.md). On the PC, the whole chain runs
on the `sim` backend (`make sw-sim`): 87 layer outputs, 0 differences. What follows needs
Vivado or the KV260. Board procedure: `hw/boards/kv260/README.md`. Index:
[README.md](README.md).

### [ ] T7.1 — Bitstream and timing
- **Condition**: Vivado (+ `dtc`) · **Needs first**: T6.5 (IP export)
- **Done on the PC**: `hw/boards/kv260/build.tcl`, `pl.dtsi`, `shell.json`,
  `firmware.sh`, written and never run.
- **To do**: everything that needs Vivado: bitstream generation, timing met at 200 MHz. The
  script has never been run (Vivado missing); `pl.dtsi` not compiled (dtc missing).
  Commands: `make hls-export BOARD=kv260` → `make vivado-build` → `make fpga-firmware`.
- **Acceptance**: bitstream generated from the tcl alone; WNS, TNS, WHS ≥ 0.
- **Unlocks**: T7.2, T8.1 (LUT/DSP/BRAM, chip power), T9.1.3, T10.9.

### [ ] T7.2 — ARM driver on the board
- **Condition**: Board, Vitis (header) · **Needs first**: T7.1
- **Done on the PC**: `run_compare --layer N` == dump on the sim backend; `make sw-board`
  builds.
- **To do**: `make check-regmap` against the generated `xyolo_conv_hw.h`; run on the KV260
  (`make sw-board` natively, `run_compare --uio /dev/uioN --layer 0`).
- **Acceptance**: an isolated layer run on the board == golden dump.

### [ ] T7.3 — Post-processing time on the A53
- **Condition**: Board · **Needs first**: T7.2
- **Done on the PC**: detections == `detections.json`; < 0.02 ms on x86.
- **To do**: post-processing time measured on the A53.

### [ ] T7.4 — 0 differences on the board, demo
- **Condition**: Board · **Needs first**: T7.3
- **Done on the PC**: sim backend: 87 layer outputs, 0 differences, identical detections
  (v2 VOC, v3 COCO, 3 images); `yolo_app --image 000004.jpg`: 7 cars.
- **To do**: 0 differences on the KV260 (`hw/boards/kv260/README.md`, step 3); demo on the
  board.
- **Acceptance**: 0 differences on all the layers for the 3 reference images.
- **Unlocks**: M8, T17.19, T20.9, T28.5.
