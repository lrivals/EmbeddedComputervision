# M36 — Decision explanation: deferred items

Parent: [M36-explanation.md](../M36-explanation.md). Index: [README.md](README.md).

### [ ] T36.10 — Exact cost of the on-chip CAM
- **Condition**: Vitis · **Needs first**: T36.10 (PC part), T29.4, T6.0
- **To do**: LUT, DSP and BRAM of the integer CAM (in the kernel or the side kernel that
  reads the L13 buffer) from synthesis, in place of the `perf_model` cost.

### [ ] T36.10 — CAM read on the board
- **Condition**: Board · **Needs first**: T36.10 (Vitis part), T7.4
- **To do**: the 13 × 13 int32 map read by the ARM == golden on the reference images;
  latency added to the frame, measured.
