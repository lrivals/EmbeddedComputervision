# M2 — Targets, loss, training

Goal: a YOLOv3 loss whose gradients are all checked, a training loop able to overfit one
image, then fine-tuning on VOC.

### [x] T2.1 — IoU
- **Spec**: §5.3, §5.2 (`iou_wh`) · **Depends on**: T0.2 · **Size**: S
- **Deliverables**: `python/yolo/infer/boxes.py` (centre/corner conversions, vectorised
  N×M IoU, `iou_wh`) + tests
- **Acceptance**: edge cases (disjoint, nested, identical, IoU = 0 on contact)
- **Done**: zero union → IoU 0; `iou` (cxcywh), `iou_xyxy`, `iou_wh` as (N, M) matrices.

### [x] T2.2 — Anchors by k-means
- **Spec**: §5.2 · **Depends on**: T2.1, T0.6 · **Size**: S
- **Deliverables**: `python/yolo/data/anchors.py`, `tools/kmeans_anchors.py`
- **Acceptance**: k = 5 (v2) and k = 6 (v3) on VOC; mean IoU to the nearest centroid
  reported and compared with the Darknet anchors; fixed seed
- **Notes**: distance $1 - \text{IoU}$, never Euclidean
- **Done** (`make anchors` → `results/anchors.md`, 40,058 VOC07+12 trainval boxes in pixels
  of the 416 letterbox): k = 5 → mean IoU 0.611 (YOLOv2 reports 61.0 for 5 clusters),
  `yolov2-tiny-voc.cfg` anchors 0.559; k = 6 → 0.631, `yolov3-tiny.cfg` (COCO) anchors
  0.603. **k-means++** initialisation with d = 1 − IoU (outside the base): the uniform draw
  of §5.2 easily falls into a local minimum (two centroids in the same cluster).

### [x] T2.3 — Target encoding and masks
- **Spec**: §5.1 · **Depends on**: T2.1 · **Size**: M
- **Deliverables**: `python/yolo/data/targets.py` + tests
- **Acceptance**: for a hand-built set of boxes: right cell, right anchor (among the 6, all
  scales), exact targets $x^*, y^*, t_w^*, t_h^*$, correct obj / ignore / noobj masks
- **Notes**: the *ignore* mask depends on the **predicted** boxes: it is computed in the
  loss, at every iteration, not when the data is loaded
- **Done**: anchors converted to a fraction by /416 whatever the input size
  (conventions.md); cell bounded to S − 1 for g = 1; collision on the same cell/same anchor:
  the last object wins (Darknet, outside the base). `heads(net)` describes the
  `yolo`/`region` heads.

### [x] T2.4 — YOLOv3 loss and gradients
- **Spec**: §6.2 · **Depends on**: T2.3, T1.6 · **Size**: M
- **Deliverables**: `python/yolo/train/loss.py` + tests
- **Acceptance**: gradcheck of the full loss ≤ 1e-7 on the mini-network of §11 (2 images,
  3 objects); analytical gradients == table of §6.2
- **Notes**: coordinates on $\sigma(t_x)$ (variant outside the base); weight
  $\omega = 2 - g_w g_h$; *ignore* anchors excluded from all the terms
- **Done**: `yolo_loss` returns the loss (sum over the batch), the δ of the heads, the
  coord/obj/noobj/cls parts and the elementary terms (gradcheck term by term). Gradcheck of
  the full loss ≤ 1e-7 on the mini-network of §11 (16×16 input, 2 images, 3 objects,
  obj/ignore/noobj cells all present, no IoU within 1e-3 of the threshold); gradients ==
  table of §6.2 to 1e-12. `ignore_thresh` = 0.5 by default (spec; the `.cfg` files have
  0.7), `lambda_coord` = 1.

### [x] T2.5 — YOLOv2 loss (and variants)
- **Spec**: §2.2, §6.1, §6.3 · **Depends on**: T2.4 · **Size**: M
- **Deliverables**: class softmax for Tiny-YOLOv2 in `loss.py`; GIoU/CIoU as an option
- **Acceptance**: gradcheck ≤ 1e-7 for each implemented variant
- **Notes**: GIoU/DIoU/CIoU are outside the base (§6.3); to do only if useful
- **Done**: `class_mode="softmax"` (cross-entropy, stabilised softmax), chosen
  automatically for a `region` layer; gradcheck ≤ 1e-7. GIoU/DIoU/CIoU not done.

### [x] T2.6 — Data pipeline
- **Spec**: §1, §7.1 (augmentation) · **Depends on**: T0.6 · **Size**: M
- **Deliverables**: `python/yolo/data/{letterbox,augment,loader}.py`
- **Acceptance**: reversible letterbox (boxes reprojected to within 1e-6); geometric
  augmentation ±20 % and HSV ×1.5 applied identically to the boxes; visual check
- **Notes**: parallel loading (multiprocessing) so as not to starve the loop
- **Done**: geometry = a single affine map (letterbox, then scale and translation ±20 %
  around the centre, optional horizontal flip, outside the base) applied to the image (PIL)
  and to the box corners; boxes clipped, removed under 2 px. HSV in NumPy. Test: synthetic
  white rectangle == transformed box to within 1.5 px. `difficult` objects removed in
  training (Darknet). Loader: `multiprocessing.Pool`, bounded prefetch, draws by
  `default_rng([seed, epoch, image])` → identical in sequential and parallel runs. Visual
  check: `python tools/show_augment.py` → `build/augment_samples/`.

### [x] T2.7 — Training loop
- **Spec**: §7.1 · **Depends on**: T2.4, T2.6, T1.8 · **Size**: M
- **Deliverables**: `python/yolo/train/{optim,schedule,trainer}.py`, `tools/train.py`
- **Acceptance**: SGD momentum 0.9 + weight decay 5e-4 checked on a quadratic problem;
  warm-up; steps; multi-scale every 10 iterations (320 to 608); checkpoint save and resume
- **Notes**: no weight decay on γ, β and biases (usual convention, outside the base)
- **Done**: loss and gradients divided by N in the trainer; ramp lr·((it+1)/burn_in)⁴
  (Darknet, outside the base); multi-scale size = pure function of (seed, it // 10), so
  checkpoint resume is bit for bit (tested). COCO → VOC transfer: `copy_matching`
  (everything except heads 15 and 22). Measurement: `tools/train.py --init coco`, batch 8
  at 416, float32, 4 workers: ≈ 3.6 s/iteration (≈ 0.45 s/image).

### [x] T2.8 — Overfitting one image
- **Spec**: §11 · **Depends on**: T2.7 · **Size**: S
- **Deliverables**: `python/tests/test_overfit.py` (marked `slow`)
- **Acceptance**: the loss tends to 0; after decoding (T3.1-T3.2), the predicted boxes
  coincide with the truth (IoU > 0.9)
- **Done** (`make test-slow`, ≈ 1 min): Tiny-YOLOv3-VOC, He init, 256×256 synthetic image
  with 3 objects, 200 SGD iterations (lr 1e-3, ramp 50): loss 893 → 0.11; best box of each
  class: score > 0.99, IoU > 0.997.

### [ ] T2.9 — Fine-tuning on VOC
- **Spec**: §7.1 · **Depends on**: T2.8, T1.9, T3.3 · **Size**: L
- **Deliverables**: fine-tuned weights in `weights/`, loss curve, report in `results/`
- **Acceptance**: VOC2007 test mAP of a Tiny-YOLOv3 fine-tuned from the COCO weights,
  reported and compared with the Darknet reference
- **Notes**: **main risk of the project**: the slowness of NumPy. Measure the time per
  iteration from T1.2 on; if needed, a subset of VOC or fewer classes. M3 and M4 do not
  depend on this task (T1.9 shortcut)
- **Ready**: `tools/train.py --net tiny-yolov3-voc --init coco` (checkpoints, `loss.csv`,
  `final.weights`). At the rate measured in T2.7 (≈ 0.45 s/image at 416), one VOC07+12
  epoch (16,551 images) takes ≈ 2 h. To watch: the initial noobj loss is very high with
  He-initialised heads (≈ 2,600 per image); it collapses within a few iterations.
