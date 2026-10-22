# M17 — Article: deferred items

Parent: [M17-article.md](../M17-article.md). Index: [README.md](README.md).

### [ ] T17.19 — Switch "projection → measurement"
- **Condition**: Board · **Needs first**: T7.4, T8.1-T8.3 on the board
- **To do**: statuses of the performance and FPGA mAP keys moved to `mesure`; narrative of
  §0 (abstract), §8 and §13 (Limitations) rewritten; projection / measurement comparison
  added in §8; in §11, the latency bound (`wcet.v2.*`) compared with the end-to-end
  latency measured on the board (T30.2) and the co-sim / board rows of the equivalence
  matrix filled (T28.5) (`make article`).
- **Acceptance**: no `projection` key cited in the abstract any more; `python -m
  tools.article --check` passes.
