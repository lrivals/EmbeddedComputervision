# M39 — Learned failure prediction: deferred items

Parent: [M39-failure-prediction.md](../M39-failure-prediction.md). T39.0 to T39.5 and the
C-sim part of T39.6 run on the PC. Index: [README.md](README.md).

### [ ] T39.6 — Exact cost of the failure head
- **Condition**: Vitis · **Needs first**: T39.6 (PC part), T29.4 (deferred), T6.0
- **To do**: LUT, DSP, BRAM and cycles of the failure head (in the kernel or the side kernel)
  from synthesis and co-sim, in place of the `perf_model` cost; `make check-regmap` with the
  new status field.
- **Unlocks**: the hardware cost row of [T39.3](../M39-failure-prediction.md) (head against
  KGFP).
