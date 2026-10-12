# ADR 0003 — Choice of the FPGA board

- Status: **accepted** (2026-10-05) — Vivado/Vitis installation still to be checked

## Context

The "single engine, layer by layer" architecture (§10.1) suits a SoC with an ARM
(Zynq-7000, Zynq UltraScale+/Kria). The choice fixes the resources (DSP, BRAM, DDR
bandwidth) and therefore the tiles Tm, Tn, Tr, Tc.

## Criteria

| Criterion | Measure |
|---|---|
| Feasibility | tiles found by `tools/roofline.py` (T5.6) that fit in BRAM |
| Estimated performance | ms/image at the selected roofline point |
| Processor | hard ARM available for the driver and the post-processing |
| Tools | supported Vivado/Vitis version, free licence |
| Cost and availability | price, delivery |
| Comparability | boards used by the base (Zynq 7035, Zybo Z7-20, KV260) |

## Candidates

Kria KV260, Zynq-7000 (Zybo Z7-20, PYNQ-Z2), Ultra96-V2.

| Criterion | KV260 | Ultra96-V2 | Zybo Z7-20 / PYNQ-Z2 |
|---|---|---|---|
| Feasibility (roofline, ≤ 80 %) | Tm, Tn, Tr, Tc = 32, 24, 13, 13; 896 DSP, 112 BRAM18 | 8, 32, 13, 13; 288 DSP, 80 BRAM18 | 8, 16, 13, 13; 160 DSP, 48 BRAM18 |
| Tiny-YOLOv2 / v3 (ms, roofline) | **31.9 / 27.0** | 86.2 / 71.7 | 198.4 / 159.4 |
| Processor | 4 × Cortex-A53 | 4 × Cortex-A53 | 2 × Cortex-A9 |
| Tools | Vitis/Vivado, free licence (xck26 in the Standard edition) | same | same |
| Cost | ~€250 | ~€250-300 | ~€300 / ~€150 |
| Comparability | 2026-fata (Vitis AI DPU, not hand-made) | — | 2025-kim (Zybo Z7-20, INT16) |

Source of the figures: `results/roofline.md` (`make roofline`), resources from
`hw/boards/*.yaml`.

## Decision

**Kria KV260** (`xck26-sfvc784-2LV-c`, 200 MHz), tiles **Tm = 32, Tn = 24, Tr = 13,
Tc = 13**: `hls/configs/kv260.tcl`.

- It is the only board whose roofline point leaves a real margin: 6× faster than the
  Zynq-7020 boards and 2.7× faster than the Ultra96-V2, for a similar price.
- BRAM is not the limiting factor (112 BRAM18 out of 288, URAM unused): Tr and Tc can grow,
  or the design can move to a line buffer, without changing boards.
- The A53 ARM is enough for the driver and the post-processing of M7 (decoding + NMS on the
  host, §10.3).
- Comparability: 2026-fata measures YOLOv3-tiny on the same board with a preconfigured DPU,
  which gives a direct "hand-made accelerator vs DPU" comparison point.

## Consequences

- The tiles are compile-time parameters (`-DACC_TM=… -DACC_TN=…`): changing boards means
  writing `hls/configs/<board>.tcl`. The testbench is also run with the Zynq-7020 config
  (8, 16, 13, 13) to check the partial tile borders.
- Vitis HLS is not yet installed on the development machine: the C-sim runs with g++ and
  the open-source `ap_int` headers (`hls/CMakeLists.txt`). Synthesis, co-simulation and
  export (T6.5) wait for the installation of Vitis 2023.2 or later (version validated on
  KV260 by AMD).
- Software stack of the board (M7): Kria Ubuntu 22.04, `xmutil` for the bitstream and the
  overlay, UIO for the registers and the interrupt, u-dma-buf for contiguous memory.
  Preferred to XRT (heavier Vitis platform flow) and to PYNQ (driver in Python): the driver
  stays in pure C++ and the same code runs on the PC with a simulated backend.
