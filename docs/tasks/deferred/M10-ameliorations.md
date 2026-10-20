# M10 — Improvements: deferred items

Parent: [M10-ameliorations.md](../M10-ameliorations.md). On the PC, every direction is
done in C-sim with its cycle model (`perf_model --check`: 0 differences). What follows
needs Vitis, Vivado or the board. Index: [README.md](README.md).

## A. Single-engine kernel (Vitis)

### [ ] T10.1 — Synthesis of the 64-bit ports
- **Condition**: Vitis · **Needs first**: T6.0
- **Done on the PC**: 64-bit ports, 61.3 ms in C-sim, == golden.
- **To do**:
  - synthesis and co-sim (Vitis);
  - byte mask (WSTRB) of the partial writes; possible fallback: 8-bit output port widened
    by `m_axi_max_widen_bitwidth`;
  - II of the loader in upsample (2·WORD columns per word, L21 of v3).

### [ ] T10.2 — Synthesis of the parallel requantization
- **Condition**: Vitis · **Needs first**: T10.1
- **To do**: DSPs actually added and frequency met (synthesis).

### [ ] T10.3 — Synthesis check of the trim
- **Condition**: Vitis · **Needs first**: T6.0
- **Done on the PC**: `ACC_TRIM`, C-sim == golden.
- **To do**: confirm in the synthesis report that the per-lane mux of `compute` costs no
  frequency.

### [ ] T10.4 — BRAM of the 14 × 14 buffers
- **Condition**: Vitis · **Needs first**: T10.2
- **Done on the PC**: folding of L00 and per-layer tile, 37.2 ms in C-sim.
- **To do**: 14 × 14 buffers: BRAM to be confirmed in synthesis.

### [ ] T10.10 — Synthesis of the packed loader
- **Condition**: Vitis · **Needs first**: T6.0
- **Done on the PC**: 4-bit packing, `tb_net_mixed` == golden == dumps.
- **To do**: synthesis of the packed loader.

## B. ARM software (board)

### [ ] T10.5 — Throughput of the inter-image pipeline
- **Condition**: Board · **Needs first**: T8.1
- **Done on the PC**: `--pipeline`, ctests `yolo_bench_pipeline`, `yolo_app_pipeline`.
- **To do**: the throughput gain, measurable only on the KV260. In C-sim, the accelerator
  takes 8 s per image and hides everything.

### [ ] T10.6 — Cached DMA buffer on the board
- **Condition**: Board · **Needs first**: T8.1
- **Done on the PC**: `--cached`, ranged `sync_for_*`; `make sw-board` builds.
- **To do**: output == golden on the board, copy and post-processing times compared with
  the uncached mode.

### [ ] T10.7 — Conv → post chaining on the board
- **Condition**: Vitis (header), Board · **Needs first**: T9.1.3
- **Done on the PC**: sequencer, ctest `run_compare_chain`.
- **To do**:
  - real register offsets (Vitis header, `make check-regmap`);
  - ARM time per image measured on the board: in C-sim, the kernel runs in the calling
    thread.

## C. Streaming (Vitis, Vivado, board)

### [ ] T10.8 — Synthesis and co-sim of `yolo_stream`
- **Condition**: Vitis · **Needs first**: T6.0
- **Done on the PC**: ROM, outer PE, FIFO depths; `tb_stream_rom` == dumps.
- **To do**:
  - synthesis (DSP, LUT, BRAM/URAM, frequency) and co-sim of the DATAFLOW (Vitis missing):
    `make hls-synth-stream | hls-cosim-stream`;
  - check that Vitis accepts the 1.2 MB ROM of L10 in `ARRAY_RESHAPE` (otherwise, split by
    og group).
- **Acceptance**: compared with the plan of T9.4.1 (968 DSP, 32.1 img/s); co-sim without
  deadlock.

### [ ] T10.9 — Streaming design on the board
- **Condition**: Vitis, Vivado, Board · **Needs first**: T10.8, T7.1
- **Done on the PC**: AXIS ports, `StreamAccelerator`, emulated DMA; ctests
  `yolo_bench_stream_*`; `build.tcl ... stream` and `pl_stream.dtsi` written, not run.
- **To do**:
  - export of the IP (`make hls-export-stream`), block design and bitstream
    (`make vivado-build ENGINE=stream`, `ENGINE=stream make fpga-firmware`);
  - real register offsets and names of the AXIS ports (`in_r`, `out_r`);
  - output == golden, throughput and latency on the board.
