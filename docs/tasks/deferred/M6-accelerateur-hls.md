# M6 — Vitis HLS accelerator: deferred items

Parent: [M6-accelerateur-hls.md](../M6-accelerateur-hls.md). Everything checkable without
Xilinx tools is done there (C-sim with g++, `make csim-gcc`, cycle estimates). What follows
needs Vitis HLS. Index and conditions: [README.md](README.md).

### [ ] T6.0 — Install Vitis / Vivado
- **Condition**: Vitis, Vivado · **Needs first**: —
- **Done on the PC**: ADR 0003 accepted (KV260, tiles 32, 24, 13, 13 at 200 MHz);
  `hls/configs/kv260.tcl`, `zybo-z7-20.tcl`.
- **To do**: Vitis/Vivado missing from the development machine — installation to do and to
  check (`which vitis_hls`).
- **Acceptance**: `which vitis_hls` and `which vivado` answer; `make csim` (Vitis C-sim)
  passes like `make csim-gcc`.
- **Unlocks**: every other item of this folder.

### [ ] T6.1 — II = 1 of the convolution PE
- **Condition**: Vitis · **Needs first**: T6.0
- **Done on the PC**: `conv_pe.cpp`, `tb_conv` == golden == dump on both networks.
- **To do**: II = 1 reached in the synthesis report (`make hls-synth`, Vitis required).
- **Acceptance**: inner loop `PIPELINE II=1` reached in the report.

### [ ] T6.2 — Co-sim and real cycles
- **Condition**: Vitis · **Needs first**: T6.1
- **Done on the PC**: ping-pong buffers, C-sim == golden, `make hls-cycles` (≈ 210 ms).
- **To do**: co-sim, traces and real cycles (`make hls-cosim`, Vitis required).
- **Acceptance**: co-sim == golden; transfer/compute overlap visible in the traces;
  cycles per layer measured.

### [ ] T6.3 — DSP cost of the output stage
- **Condition**: Vitis · **Needs first**: T6.1
- **Done on the PC**: layers 0 to 11 (and all the others) == golden in C-sim; 4 DSP48
  expected.
- **To do**: DSP cost measured in the synthesis report (`store_tile`, `make hls-report`).

### [ ] T6.5 — Synthesis, co-sim of one image, IP export
- **Condition**: Vitis · **Needs first**: T6.3, T6.4
- **Done on the PC**: `hls/scripts/{common,csim,synth,cosim,export}.tcl`,
  `tools/hls_report.py`; C-sim cycles section of `results/hls_report.md` filled.
- **To do**: everything that needs Vitis: synthesis, co-sim of one image (long: ~4·10⁷
  estimated cycles), timing at 200 MHz, IP export (`make hls-synth | hls-cosim |
  hls-export BOARD=kv260`, then `make hls-report`).
- **Acceptance**: RTL co-sim == golden on a full image; timing met at 200 MHz; IP exported
  for Vivado; synthesis and co-sim sections of `results/hls_report.md` filled.
- **Unlocks**: T7.1, `make check-regmap` (generated `xyolo_conv_hw.h`), T13.24.
