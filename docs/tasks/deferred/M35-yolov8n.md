# M35 — YOLOv8n without PyTorch: deferred items

Parent: [M35-yolov8n.md](../M35-yolov8n.md). Index: [README.md](README.md).

### [ ] T35.7 — YOLOv8n convs in synthesis
- **Condition**: Vitis · **Needs first**: T35.7 (PC part: export, golden == C-sim with
  gcc), T26.3, T6.0
- **To do**: synthesis and co-sim of the backbone, neck and head convs of the exported
  YOLOv8n (`make hls-synth`, `hls-cosim`); resources and cycles in place of the
  `perf_model` estimate.

### [ ] T35.7 — YOLOv8n post-processing on the board
- **Condition**: Board · **Needs first**: T35.7 (Vitis part), T7.4
- **To do**: integer DFL and class sigmoid of `sw/postproc` on the A53, equal to the
  Python integer DFL on the head outputs of 10 images; time per frame on the A53.
