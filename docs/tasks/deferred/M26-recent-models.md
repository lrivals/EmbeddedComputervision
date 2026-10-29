# M26 — Recent models: deferred items

Parent: [M26-recent-models.md](../M26-recent-models.md). T26.0 runs on the PC up to the
C-sim (float, integer, export, golden, C-sim on COCO val2017). Index: [README.md](README.md).

### [ ] T26.0 — YOLOv7-tiny in synthesis and on the board
- **Condition**: Vitis, Board · **Needs first**: T26.0 (PC part), T6.5, T7.4
- **To do**:
  - synthesis with the stride-1 5 × 5 pool of the output stage (`PK_MAX = 5`,
    `LayerDesc::pool_lo`): LUT, DSP, BRAM and timing against the v4-tiny build;
  - co-sim on a few images (same detections as the C-sim);
  - on the board: the 5,000 COCO images through `yolo_bench --uio`, equal to the C-sim
    image by image, and the time per frame against the `perf_model` projection
    (16.6 M cycles, 83 ms at 200 MHz).
