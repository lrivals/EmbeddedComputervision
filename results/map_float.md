# Float reference mAP (T3.4)

Tiny-YOLOv2 VOC, Darknet weights `yolov2-tiny-voc.weights` (pjreddie.com), NumPy float32
model of `python/yolo/`, VOC2007 test (4,952 images), 416×416 input, confidence threshold
0.005, per-class NMS 0.45, 11-point AP (VOC2007), `difficult` objects ignored.

```
python tools/eval_voc.py --net tiny-yolov2-voc --weights weights/yolov2-tiny-voc.weights \
    --resize stretch            # reference
python tools/eval_voc.py ... --resize letterbox [--interp darknet]
```

## Result

**Reference for M4 (integer) and M8 (hardware): mAP = 56.30** (direct resize, Pillow
interpolation). Value published by Darknet for Tiny-YOLOv2 VOC: **57.1**. Gap:
**0.8 point**.

| Preprocessing | Interpolation | mAP |
|---|---|---|
| direct resize (`stretch`) | Pillow bilinear | **56.30** |
| direct resize (`stretch`) | Darknet (`resize_image`) | 56.32 |
| letterbox | Pillow bilinear | 54.18 |
| letterbox | Darknet (`resize_image`) | 53.98 |

## Explaining the gap

- **Preprocessing (≈ 2.1 points)**: these weights were trained and evaluated by the
  Darknet of the YOLOv2 era, which resizes the image to a square without keeping the
  aspect ratio. With the letterbox (the one of our training pipeline), the objects are
  smaller and distorted differently from training: −2.1 points. The reference therefore
  uses `stretch`, which is also the simplest to reproduce in hardware.
- **Interpolation (≈ 0)**: reproducing Darknet's `resize_image` (bilinear with aligned
  corners, without anti-aliasing; tested against a line-by-line port) instead of Pillow's
  bilinear only changes the mAP by ±0.2 point. It is not the cause of the gap.
- **Remainder (0.8 point, under the 1-point tolerance)**: plausible causes, not measured
  here — JPEG decoding (stb_image in Darknet, libjpeg in Pillow), details of Darknet's
  evaluation loop (`validate_detector`, version of the code that produced the 57.1) and the
  evaluation script used for the published figure. Our evaluation is identical to the
  official devkit to within 1e-12 (`python/tests/test_metrics.py`).

## AP per class

| Class | stretch | stretch (Darknet) | letterbox | letterbox (Darknet) |
|---|---|---|---|---|
| aeroplane | 61.4 | 61.9 | 57.9 | 58.0 |
| bicycle | 71.6 | 71.8 | 67.8 | 67.4 |
| bird | 48.5 | 48.4 | 47.2 | 46.1 |
| boat | 41.5 | 42.1 | 36.5 | 34.8 |
| bottle | 21.6 | 21.7 | 21.2 | 19.9 |
| bus | 67.8 | 67.9 | 67.7 | 66.9 |
| car | 67.0 | 67.2 | 63.0 | 62.9 |
| cat | 70.4 | 70.7 | 66.4 | 66.8 |
| chair | 35.3 | 35.0 | 31.3 | 31.8 |
| cow | 54.6 | 53.8 | 53.5 | 53.8 |
| diningtable | 57.5 | 58.2 | 55.4 | 57.0 |
| dog | 62.3 | 62.4 | 62.0 | 62.4 |
| horse | 70.9 | 71.0 | 70.1 | 70.0 |
| motorbike | 70.5 | 68.8 | 69.8 | 68.8 |
| person | 60.6 | 59.9 | 58.3 | 58.1 |
| pottedplant | 27.5 | 27.8 | 26.8 | 27.1 |
| sheep | 56.5 | 55.6 | 50.7 | 52.2 |
| sofa | 52.1 | 52.0 | 51.8 | 50.7 |
| train | 69.7 | 70.7 | 68.1 | 67.7 |
| tvmonitor | 58.6 | 59.2 | 58.1 | 57.2 |
| **mAP** | **56.30** | **56.32** | **54.18** | **53.98** |

Time: ≈ 170 ms/image (NumPy float32 forward pass, 22 cores), ≈ 15 min per pass.
