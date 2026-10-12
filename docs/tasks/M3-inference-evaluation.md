# M3 — Inference and evaluation

Goal: from the raw network output to the final boxes, and a VOC mAP that serves as the
reference for all the following stages.

### [x] T3.1 — Decoding the heads
- **Spec**: §8.1 · **Depends on**: T1.8, T2.1 · **Size**: S
- **Deliverables**: `python/yolo/infer/decode.py` + tests
- **Acceptance**: decode(encode(box)) == box to within 1e-9 (consistency with T2.3); v3
  score $\sigma(t_o)\sigma(t_c)$ and v2 score $\sigma(t_o)\,\text{softmax}_c$; boxes brought
  back to the coordinates of the original image (inverse letterbox)
- **Done** (brought forward for T2.8): `decode_head`, `decode(outputs, net)` (v2 for
  `region`), `to_original`; box order (anchor, row, column), heads concatenated.

### [x] T3.2 — Threshold and NMS
- **Spec**: §8.2 · **Depends on**: T3.1 · **Size**: S
- **Deliverables**: `python/yolo/infer/nms.py` + tests
- **Acceptance**: per-class NMS; results identical to a naive O(n²) implementation on
  1,000 random sets
- **Notes**: default thresholds `conf = 0.25`, `iou = 0.45` (outside the base), as
  parameters
- **Done**: `nms` (stable sort, suppression if IoU > threshold), `filter_and_nms` (strict
  threshold then per-class NMS), `postprocess` (decoding + NMS per image); equal to the
  pseudo-code of §8.2 on 1,000 sets, NMS alone and per class.

### [x] T3.3 — PASCAL VOC mAP
- **Spec**: §8.3 · **Depends on**: T3.2, T0.6 · **Size**: M
- **Deliverables**: `python/yolo/infer/metrics.py`, `tools/eval_voc.py`
- **Acceptance**: 11-point AP; `difficult` objects ignored; a ground truth matched only
  once; results identical to the official VOC script on a provided set of test detections
- **Done**: `eval_class`/`evaluate` in VOC pixels (+1 convention), inclusive IoU threshold
  (`VOCevaldet.m`), 11-point AP (VOC2010+ area option); raw `xyxy` added to the
  annotations. Reference: the `voc_eval.py` port of py-faster-rcnn
  (`python/tests/voc_eval_ref.py`, aligned on the MATLAB devkit: `>=`, stable sort);
  equality to 1e-12 on a set of 50 synthetic images (difficult objects, duplicates, false
  alarms). `tools/eval_voc.py` evaluates a network or `comp4_det_test_<class>.txt` files.

### [x] T3.4 — Demo and float reference mAP
- **Spec**: §10.5 step 1 · **Depends on**: T3.3, T1.9 · **Size**: S
- **Deliverables**: `tools/detect.py` (image → annotated image), `results/map_float.md`
- **Acceptance**: VOC2007 test mAP of Tiny-YOLOv2 VOC (Darknet weights) close to the value
  published by Darknet; gap explained if it exceeds 1 point
- **Notes**: **validation milestone**: this mAP is the reference of the integer (M4) and
  hardware (M8) stages
- **Done**: `yolo/infer/pipeline.py` (`preprocess` letterbox | stretch, Pillow | Darknet
  interpolation; `detect`), `tools/detect.py`, `tools/eval_voc.py --weights`,
  `make detect` / `make eval-float`. **Reference mAP = 56.30** (stretch, Pillow) against
  57.1 published (gap 0.8 point); letterbox: 54.18 (−2.1 points, these weights were trained
  without letterbox); Darknet interpolation: ±0.2 point. Details:
  `results/map_float.md`.
