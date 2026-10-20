# M30 — Latency bound and determinism (optional)

Goal: a latency bound that can be **demonstrated**, not only a mean. Sixth gap of the thesis
state of the art (ARCYS, 2026-10):

- FPGA papers give mean latencies; the latency can vary with the video (Fata 2026, Fig. 14)
  and the post-processing on the ARM is often the limit; no worst case, p99 or end-to-end
  jitter is published;
- CoDANN II argues that an FPGA is more predictable than a GPU (no hidden microcode, no
  DVFS, no shared scheduler), without a measurement;
- on GPU, traceability stops at the NVIDIA drivers (Belcaid 2024).

The engine of the repo is a fixed schedule of `LayerDesc` descriptors: its cycle count should
not depend on the image, except for the decode and the NMS, whose worst case is already
bounded ([T9.1](M9.1-postproc-materiel.md), [T12.7](M12-profils-pc.md)). This milestone
proves the bound for the accelerator, then measures the whole frame on the board.

**State**: not started (2026-10-20). T30.0 and T30.1 run on the PC; T30.2 and T30.3 need the
board.

### [ ] T30.0 — Static bound of the accelerator
- **Spec**: §10.3 · **Depends on**: T6.3, T9.1.2 · **Size**: M
- **Deliverables**: `tools/perf_model.py --wcet`; `results/latency_bound.md`
- **Acceptance**: per network, cycles per layer from the `LayerDesc` only (no image input),
  with the argument that no loop bound depends on data; NMS and decode bound for the worst
  case (all cells pass, slot capacity full); total in ms at the clock of the board, and the
  watchdog value used by [T29.5](M29-runtime-monitors.md)
- **Notes**: DDR transfers are the data-independent part that is not cycle-exact in C-sim:
  bound them by the transfer size and the measured bandwidth (T30.2).

### [ ] T30.1 — Cycle distribution in simulation
- **Spec**: §10.3 · **Depends on**: T30.0 · **Size**: M
- **Deliverables**: cycle counters of the C-sim (and of the co-sim when Vitis is available)
  over ≥ 10,000 images; adversarial inputs (uniform, noise, an image that makes every cell
  pass); `results/latency_bound.md` section
- **Acceptance**: the accelerator cycle count is identical for every image (or its spread is
  explained); the post-processing maximum is ≤ the static bound of T30.0 and reached by the
  adversarial input

### [ ] T30.2 — End-to-end latency on the board
- **Spec**: §10.4 · **Depends on**: T8.1, T30.0 · **Size**: M
- **Deliverables**: `sw/app` timing mode extended; `results/latency_bound.md` section
- **Acceptance**: over 10⁵ frames, per stage (capture or load, preprocessing, accelerator,
  post-processing, decision): mean, p99, p99.9, maximum and jitter; with an idle ARM and
  with a stress load (memory and CPU); the measured accelerator time against the static
  bound
- **Notes**: extends [T8.1](M8-mesures.md) (mean and p99 over 1,000 images).

### [ ] T30.3 — Sources of non-determinism on the PS
- **Spec**: §10.4 · **Depends on**: T30.2 · **Size**: M
- **Deliverables**: `results/latency_bound.md` section
- **Acceptance**: effect of Linux scheduling, DDR contention between the PS and the PL, and
  caches on the tail of T30.2, measured one by one; a variant (isolated core, `SCHED_FIFO`,
  locked memory, or bare metal) and its maximum; recommendation for the safety path

## Dependencies

```mermaid
graph LR
  T63[T6.3] --> T300[T30.0] --> T301[T30.1]
  T912[T9.1.2] --> T300
  T81[T8.1] --> T302[T30.2] --> T303[T30.3]
  T300 --> T302
```
