# M25 — Dependability: deferred items

Parent: [M25-dependability.md](../M25-dependability.md). The functions and their byte
checks run on the PC (C-sim gcc, `perf_model` estimates); only the exact costs and the
configuration-memory faults need the tools or the board. Index: [README.md](README.md).

### [ ] T25.3 — Exact cost of the ABFT lane
- **Condition**: Vitis · **Needs first**: T25.3 (PC part, done 2026-10-20), T6.0
- **To do**:
  - **Cost**: DSP and cycles of the checksum lane after synthesis, in place of the
    `perf_model --abft` projection (+Tn = 24 DSP, no cycle: lane in the same II = 1 loop,
    weights loaded on gmem_p in parallel).
  - **II**: check that the requantization pass keeps II = 1 with the RQ-lane sum and the
    compare.
  - **Registers**: `make check-regmap` for the new output `abft_err`. The offsets
    `ABFT_ERR = 0xd4`, `ABFT_ERR_CTRL = 0xd8` of `sw/driver/regmap.hpp` are assumed (a
    reserved word after `n_calls`).
  - **Packing (M23)**: the lane is computed apart from the Tm filters, so it does not make
    the Tm count odd; check the synthesis with `ACC_PACK = 2`.

### [ ] T25.4 — Exact cost of the protections
- **Condition**: Vitis · **Needs first**: T25.4 (PC part), T6.0
- **To do**: exact BRAM, DSP and LUT of the schemes of the T25.4 section of
  `results/faults.md` (2026-10-28), in place of the `perf_model` estimates:
  - the ABFT lane (`ACC_ABFT = 1`, projected +24 DSP);
  - SECDED on the `w_a` / `w_b` banks (768 × 9 int8, projected +62.5 % bits plus
    encoders and decoders), and whether the decoder keeps II = 1;
  - the duplication of L00-L04 (in time: +28.1 % cycles in the co-sim).

### [ ] T25.7 — Configuration-memory upsets
- **Condition**: Vivado, Board · **Needs first**: T7.4
- **To do**: configuration-memory upsets are out of reach in C-sim; they need the board and
  a tool such as the Xilinx SEM IP (SEM IP in the block design, injection campaign,
  comparison with the C-sim buffer faults of T25.7).
