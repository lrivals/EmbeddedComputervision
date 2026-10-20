# M29 — Runtime monitors: deferred items

Parent: [M29-runtime-monitors.md](../M29-runtime-monitors.md). Index: [README.md](README.md).

### [ ] T29.4 — Exact cost of the hardware monitor
- **Condition**: Vitis · **Needs first**: T29.4 (PC part), T6.0
- **To do**: LUT, DSP and BRAM of the monitor (in the kernel or the side kernel) from
  synthesis, in place of the `perf_model` cost.

### [ ] T29.5 — Fallback logic on the board
- **Condition**: Board · **Needs first**: T29.5 (PC part, `sw-sim`), T7.4, T30.2
- **To do**: final check on the board: every input combination, a missing or late frame
  (watchdog of T30.0, set from the T30.2 measurements) gives `UNSAFE`; `UNSAFE` at
  start-up.
