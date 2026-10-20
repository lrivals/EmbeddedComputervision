# M9 — Extensions (optional)

Directions opened once M8 is done. T9.1 to T9.4 have become milestones of their own; T9.5
remains a direction. The engineering optimisations (wide ports, ARM pipeline, CI…) are
gathered in [M10](M10-ameliorations.md).

| Direction | Milestone | Spec | Depends on |
|---|---|---|---|
| T9.1 — Hardware post-processing | [M9.1](M9.1-postproc-materiel.md) | §10.3 (2024-zhang) | T8.1 |
| T9.2 — REQ-YOLO quantization (ADMM, powers of 2) | [M9.2](M9.2-req-yolo.md) | §9.2 (2019-ding) | T4.6 → T9.3.3 |
| T9.3 — 4-bit quantization | [M9.3](M9.3-quant-4bits.md) | §9.2, §9.3 (2024-yan) | T4.6 |
| T9.4 — Streaming architecture | [M9.4](M9.4-streaming.md) | §10.1 (SATAY, 2018-venieris) | T8.3 |

M9.3 comes before M9.2: it brings the fake-quant, the QAT mode of the trainer and the
parameterised bit width that the ADMM uses.

### [ ] T9.5 — Online learning on FPGA
- **Spec**: §4 (EF-Train), §12 (gaps) · **Depends on**: T8.3
- Open research direction: fine-tune the YOLO head on the board (BN in full precision,
  §9.1). To be crossed with the continual learning axis.
- **Deferred** (board): [deferred/M9](deferred/M9-extensions.md) — T9.5.
