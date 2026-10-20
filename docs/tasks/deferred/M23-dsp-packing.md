# M23 — Two MACs per DSP: deferred items

Parent: [M23-dsp-packing.md](../M23-dsp-packing.md). On the PC: packed PE C-sim == golden
to the byte; projections 24.7 / 23.8 ms (Tm 64), 30.0 ms (Tn 32). Index:
[README.md](README.md).

### [ ] T23.2 — Synthesis of the packed PE, choice of the tiles
- **Condition**: Vitis · **Needs first**: T6.0
- **To do**:
  - synthesis of the packed PE (`make hls-synth-pack`): DSP, LUT and timing at 200 MHz;
  - Tm 64 needs `w_buf` outside the flip-flops (LUTRAM); otherwise Tn 32. To choose after
    synthesis; `ACC_PACK` stays at 1 by default until then.
- **Unlocks**: T13.24 (packed PE rows).

### [ ] T23.3 — Packed PE measured on the board
- **Condition**: Vivado, Board · **Needs first**: T23.2, T8.1
- **To do**: measurement on the KV260 with the protocol of
  [protocole.md](../../../results/protocole.md), against the projection of T23.3.
