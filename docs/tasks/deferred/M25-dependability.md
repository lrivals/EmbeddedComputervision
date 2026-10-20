# M25 — Dependability: deferred items

Parent: [M25-dependability.md](../M25-dependability.md). The functions and their byte
checks run on the PC (C-sim gcc, `perf_model` estimates); only the exact costs and the
configuration-memory faults need the tools or the board. Index: [README.md](README.md).

### [ ] T25.3 — Exact cost of the ABFT lane
- **Condition**: Vitis · **Needs first**: T25.3 (PC part), T6.0
- **To do**: DSP and cycles of the checksum lane (Tm + 1) after synthesis, in place of the
  `perf_model` cost; check the combination with the packing of M23 (odd lane count).

### [ ] T25.4 — Exact cost of the protections
- **Condition**: Vitis · **Needs first**: T25.4 (PC part), T6.0
- **To do**: exact BRAM and DSP of ABFT, ECC on the weights in BRAM and duplication of the
  sensitive layers, in the section of `results/faults.md`, in place of the `perf_model`
  estimates.

### [ ] T25.7 — Configuration-memory upsets
- **Condition**: Vivado, Board · **Needs first**: T7.4
- **To do**: configuration-memory upsets are out of reach in C-sim; they need the board and
  a tool such as the Xilinx SEM IP (SEM IP in the block design, injection campaign,
  comparison with the C-sim buffer faults of T25.7).
