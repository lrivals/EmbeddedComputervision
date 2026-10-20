# M30 — Latency bound and determinism: deferred items

Parent: [M30-latency-bound.md](../M30-latency-bound.md). T30.0 and the C-sim part of T30.1
run on the PC. Index: [README.md](README.md).

### [ ] T30.1 — Co-sim cycles
- **Condition**: Vitis · **Needs first**: T30.1 (C-sim part), T6.5
- **To do**: cycle counters of the co-sim on the same images as the C-sim (a subset: one
  image of co-sim is ~4·10⁷ cycles); same acceptance as the C-sim part.

### [ ] T30.2 — End-to-end latency on the board
- **Condition**: Board · **Needs first**: T8.1, T30.0
- **Deliverables**: `sw/app` timing mode extended; `results/latency_bound.md` section.
- **Acceptance**: over 10⁵ frames, per stage (capture or load, preprocessing, accelerator,
  post-processing, decision): mean, p99, p99.9, maximum and jitter; with an idle ARM and
  with a stress load (memory and CPU); the measured accelerator time against the static
  bound.
- **Notes**: extends [T8.1](M8-mesures.md) (mean and p99 over 1,000 images). The measured
  DDR bandwidth goes back into the bound of T30.0. The timing mode can be written on the PC
  with the `sim` backend.

### [ ] T30.3 — Sources of non-determinism on the PS
- **Condition**: Board · **Needs first**: T30.2
- **Deliverables**: `results/latency_bound.md` section.
- **Acceptance**: effect of Linux scheduling, DDR contention between the PS and the PL, and
  caches on the tail of T30.2, measured one by one; a variant (isolated core, `SCHED_FIFO`,
  locked memory, or bare metal) and its maximum; recommendation for the safety path.
