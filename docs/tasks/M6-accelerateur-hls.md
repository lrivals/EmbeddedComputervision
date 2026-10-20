# M6 — Vitis HLS accelerator

Goal: an HLS kernel "single engine, layer by layer" (§10.1), validated in C-sim then in RTL
co-simulation against the golden model, with usable synthesis reports.

### [ ] T6.0 — Choice of the board
- **Spec**: §10.1, §10.4 · **Depends on**: T5.6 · **Size**: S
- **Deliverables**: [ADR 0003](../adr/0003-choix-carte.md) completed;
  `hls/configs/<board>.tcl` (part, clock, selected tiles)
- **Acceptance**: decision argued along the criteria of the ADR; installation of the
  corresponding Vivado/Vitis tools checked
- **Done**: ADR 0003 accepted: **KV260**, tiles 32, 24, 13, 13 at 200 MHz (roofline point
  of T5.6, 6× faster than the Zynq-7020 boards at the same price); `hls/configs/kv260.tcl`,
  plus `zybo-z7-20.tcl` (8, 16, 13, 13) to test smaller tiles.
- **Deferred** (Vitis, Vivado): [deferred/M6](deferred/M6-accelerateur-hls.md) — T6.0, installation.

### [ ] T6.1 — Convolution PE
- **Spec**: §10.2 · **Depends on**: T6.0, T5.2 · **Size**: L
- **Deliverables**: `hls/kernels/conv_pe.cpp`, `hls/tb/tb_conv.cpp`,
  `hls/scripts/csim.tcl`
- **Acceptance**: C-sim == golden for all the conv layers; inner loop `PIPELINE II=1`,
  `UNROLL` of Tm × Tn, II reached in the synthesis report
- **Notes**: `ap_int<8>` / `ap_int<32>`; adder tree on Tn; buffer partitioning aligned on
  the unrolling
- **Done**: `hls/kernels/conv_pe.cpp` (+ `accel.hpp`, `layer_desc.hpp`): same tile order as
  `golden::conv_layer`; loop (i, j, trr, tcc) flattened in `PIPELINE II=1`, too and tii
  `UNROLL`, sum over Tn in `acc_t`; `in_buf` partitioned on dim 1, `w_buf` on dims 1-2,
  `out_buf` on dim 1, `DEPENDENCE inter false`. Buffer sizes checked against B_in, B_w,
  B_out. `hls/tb/tb_conv.cpp`: each conv of both networks, 3 images, output == golden ==
  dump (pooled map and pre-pooling map), KV260 and Zybo tiles. C-sim with g++ and the
  open-source `ap_int` headers (`hls/CMakeLists.txt`, `make csim-gcc`);
  `hls/scripts/csim.tcl` (`make csim`) ready for Vitis.
- **Deferred** (Vitis): [deferred/M6](deferred/M6-accelerateur-hls.md) — T6.1, II = 1 in synthesis.

### [ ] T6.2 — Memory interfaces and control
- **Spec**: §10.2 (double buffering) · **Depends on**: T6.1 · **Size**: L
- **Deliverables**: `m_axi` for weights, activations and requant; `s_axilite` for the
  layer parameters (shape, offsets, scale, options); load/compute/store ping-pong
- **Acceptance**: co-sim == golden; transfer/compute overlap visible in the traces;
  throughput measured in cycles per layer
- **Done**: top `yolo_conv`: `m_axi` `gmem_in` / `gmem_out` (same activation arena, two
  bundles to overlap load and store), `gmem_w` (weights), `gmem_p` (biases then M0);
  `s_axilite`: flat `LayerDesc` (shape, 2 input segments, offsets, shift, options).
  `in_buf`/`w_buf` ping-pong (load of ti + 1 during the compute of ti) and `out_buf` (store
  of tile k − 1 during tile k). C-sim still == golden. Cycle estimate per layer in C-sim
  (`accel::sim_cycles`, `make hls-cycles`): ≈ 210 ms/image for Tiny-YOLOv2 at 200 MHz
  against 41 ms of pure compute — the kernel is limited by the byte-by-byte loads
  (`results/hls_report.md`).
- **Deferred** (Vitis): [deferred/M6](deferred/M6-accelerateur-hls.md) — T6.2, co-sim and real cycles.

### [ ] T6.3 — Fused output stage
- **Spec**: §9.3, §10.3 · **Depends on**: T6.1 · **Size**: M
- **Deliverables**: bias + M0/shift requantization + leaky 13/128 + 2×2 maxpool with
  stride 2 **and 1**
- **Acceptance**: layers 0 to 11 == golden in C-sim; DSP cost of the requantization
  reported
- **Done**: `hls/kernels/output_stage.hpp`: `golden::requantize` / `leaky_int` / `clip8`
  reused as they are, 2×2 s2 and s1 maxpool (border replication), pre-pooling map in DDR
  (L08 → R). Layers 0 to 11 (and all the others) == golden in C-sim, v2 and v3. A single
  acc·M0 multiplier (64 bits) shared by the Tm channels: 4 DSP48 expected.
- **Deferred** (Vitis): [deferred/M6](deferred/M6-accelerateur-hls.md) — T6.3, DSP cost of `store_tile`.

### [x] T6.4 — Upsample and route by addressing
- **Spec**: §10.3 · **Depends on**: T6.2 · **Size**: M
- **Deliverables**: addressing modes in the input loader (read at address / 2); contiguous
  allocation by the driver
- **Acceptance**: full Tiny-YOLOv3 network == golden in C-sim, without a kernel dedicated
  to the upsample or the route
- **Done**: loader with 2 segments `{offset, c, h, w, up}`, read at
  `(c·h + (r >> up))·w + (col >> up)`; `sw/driver/program.{hpp,cpp}`: contiguous DDR arena
  (buffers of the manifest end to end, aligned to 64), per-layer views (same logic as
  `Engine::place_views`), one `LayerDesc` per conv. `hls/tb/tb_net.cpp`: full Tiny-YOLOv2
  (9 calls) and Tiny-YOLOv3 (13 calls), each layer checked right after its call == golden
  == dump, 3 images, route 20 in two segments (L18 seen ×2, prepool of L08).

### [ ] T6.5 — Co-simulation and synthesis
- **Spec**: §10.4 · **Depends on**: T6.3, T6.4 · **Size**: M
- **Deliverables**: `hls/scripts/{synth,cosim,export}.tcl`; report
  `results/hls_report.md` (LUT, FF, DSP, BRAM, frequency, latency in cycles per layer)
- **Acceptance**: RTL co-sim == golden on a full image; timing met at the target
  frequency; IP exported for Vivado
- **Done**: `hls/scripts/{common,csim,synth,cosim,export}.tcl` (`make csim | hls-synth |
  hls-cosim | hls-export BOARD=kv260`); `tools/hls_report.py` (`make hls-report`) →
  `results/hls_report.md`: cycles-per-layer section (C-sim estimate) filled; synthesis
  (LUT, FF, DSP, BRAM, period, II of the loops, DSP of `store_tile`) and co-sim sections
  read from the Vitis reports as soon as they exist.
- **Deferred** (Vitis): [deferred/M6](deferred/M6-accelerateur-hls.md) — T6.5, synthesis, co-sim, timing, IP export.
