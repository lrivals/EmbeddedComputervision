# Per-layer mixed precision (T10.10)

Tiny-YOLOv2 VOC. Weights per layer in INT8, `uniform6` (integers ±31), `mixed6` (powers of
2 of REQ-YOLO, T9.2) or `uniform4` (±7, packed two per byte). INT8 activations calibrated
(`build/quant/tiny-yolov2-voc/calib.json`). The scale factor is chosen per channel, on the
grid of `pow2.project`. `stretch` preprocessing.

Production: `tools/mixed_precision.py sens | front | full`, with the results in
`build/m10/mixed/`.

## Sensitivity (500 images of VOC2007 test)

A single layer in low bit, the others in projected INT8. Reference: 57.78 on these 500
images. The table gives the mAP loss in points; a negative value is a gain, within the
noise of the subset.

| conv | L00 | L02 | L04 | L06 | L08 | L10 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| uniform4 | +4.23 | +2.88 | +2.56 | +0.14 | +0.06 | -0.38 | +0.03 | +0.29 | +0.43 |
| uniform6 | +1.27 | -0.18 | +0.67 | +0.17 | -0.24 | -0.05 | +0.26 | +0.02 | +0.39 |
| mixed6 | +3.35 | +0.49 | -0.20 | -0.71 | -0.26 | +0.15 | -0.23 | +0.28 | +0.20 |

Reading:

- The first three convs (L00, L02, L04) are the most sensitive, from 2.6 to 4.2 points in
  4 bits. L00 also loses 1.3 points in 6 bits.
- The tail, L06 to L13, loses less than 0.3 point in 4 bits. Yet it holds 98 % of the
  weights.
- `mixed6` does not stand out from `uniform6`, except on L00 (−3.4 points).

## Greedy front (500 images)

The moves (layer, uniform level) are sorted by isolated loss / bits gained, then applied
one by one; each prefix is evaluated.

- *Engine bytes*: storage of weights.bin for the single engine (6 bits on one byte).
- *Tight bytes*: storage of the streaming design (6 bits = 0.75 byte).
- DSP and img/s: KV260 `stream_model` plan with per-layer weights, two MACs per DSP in 4
  bits.

| step | change | mAP | loss | engine bytes | tight bytes | streaming DSP | streaming img/s |
|---|---|---|---|---|---|---|---|
| 0 | INT8 (reference) | 57.78 | +0.00 | 15.86 MB | 15.86 MB | 968 | 32.1 |
| 1 | L02 → uniform6 | 57.96 | -0.18 | 15.86 MB | 15.86 MB | 968 | 32.1 |
| 2 | L08 → uniform6 | 57.95 | -0.17 | 15.86 MB | 15.78 MB | 968 | 32.1 |
| 3 | L10 → uniform4 | 57.99 | -0.21 | 15.27 MB | 15.19 MB | 992 | 32.1 |
| 4 | L13 → uniform6 | 58.04 | -0.26 | 15.27 MB | 12.83 MB | 992 | 32.1 |
| 5 | L12 → uniform4 | 57.91 | -0.13 | 12.91 MB | 10.47 MB | 992 | 32.1 |
| 6 | L13 → uniform4 | 56.81 | +0.97 | 8.19 MB | 8.11 MB | 992 | 64.2 |
| 7 | L08 → uniform4 | 55.46 | +2.32 | 8.04 MB | 8.04 MB | 992 | 64.2 |
| 8 | L06 → uniform4 | 55.35 | +2.43 | 8.00 MB | 8.00 MB | 996 | 64.2 |
| 9 | L14 → uniform4 | 53.48 | +4.29 | 7.94 MB | 7.94 MB | 996 | 64.2 |
| 10 | L04 → uniform6 | 53.49 | +4.29 | 7.94 MB | 7.94 MB | 996 | 64.2 |
| 11 | L04 → uniform4 | 50.51 | +7.27 | 7.93 MB | 7.93 MB | 988 | 64.2 |
| 12 | L02 → uniform4 | 48.15 | +9.62 | 7.93 MB | 7.93 MB | 996 | 64.2 |
| - | all uniform4 | 43.19 | +14.59 | 7.93 MB | 7.93 MB | 996 | 64.2 |

## Selected configuration

Criterion: the smallest footprint of the single engine with a loss of at most 1 point on
the subset. It is step 6:

- L10, L12 and L13 in packed 4 bits;
- L02 and L08 in 6 bits;
- the rest in INT8.

| | mAP VOC2007 test (4,952 images) | weights | streaming DSP | streaming img/s |
|---|---|---|---|---|
| projected INT8 (same scale grid) | 55.69 | 15.86 MB | 968 | 32.1 |
| **selected mixed** | **54.68** (−1.01) | **8.19 MB** (−48 %) | 992 | **64.2** |
| all uniform4, PTQ (subset) | 43.19 (−14.6 on 500 images) | 7.93 MB | 996 | 64.2 |
| w4a4 QAT 600 iterations (`quant_4bits.md`) | 36.38 | 7.9 MB | 996 | 64.2 |

Reading:

- **Streaming.** L13 sets the II of the 8-bit plan. In 4 bits, it does two MACs per DSP and
  the II drops to that of the other stages. The mixed configuration thus gets the
  throughput of all-4-bit (64.2 img/s) for 1 point of mAP instead of 14.6, without
  retraining.
- **Single engine.** The weights in DDR are halved, but the cycles do not move (7,443,739,
  `perf_model --manifest`): the kernel is compute-bound and the weight loads are
  overlapped. The gain is in memory and bandwidth, not in latency.
- **Integer contract.** The packed export (`wbits` = 4 in the manifest) gives 0
  differences, on 3 images, between the C++ golden model, the C-sim (`tb_net_mixed`) and the
  Python dumps.
- **Limit.** The loss reaches 1.01 points on the full test set, just above the tolerance
  set on the subset (0.97). A per-layer QAT fine-tuning would reduce it, but the current
  QAT only has a global scheme (`lowbit.qat_from_steps`).
