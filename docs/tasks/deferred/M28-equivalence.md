# M28 — Model ↔ circuit equivalence: deferred items

Parent: [M28-equivalence.md](../M28-equivalence.md). Index: [README.md](README.md).

### [ ] T28.0 — Co-sim and board rows of the matrix
- **Condition**: Vitis, Board · **Needs first**: T28.0 (PC rows), T6.5, T7.4
- **To do**: the C-sim → co-sim and co-sim → board rows of `results/equivalence.md`, which
  stay "pending" until Vitis and the board (M6, M7); `tools/equivalence.py --check`.

### [ ] T28.5 — Equivalence on the board
- **Condition**: Board · **Needs first**: T7.4, T28.0
- **Deliverables**: per-layer output hash in `sw/app` (debug mode) and its comparison with
  the golden; row "board" of `results/equivalence.md`. The hash mode can be written and
  tested on the PC with the `sim` backend first.
- **Acceptance**: every layer output byte-identical to the golden on ≥ 1,000 images, for
  each network on the board; the bitstream and model hashes recorded.
