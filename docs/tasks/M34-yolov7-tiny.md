# M34 — YOLOv7-tiny from scratch, without PyTorch (optional)

Goal: YOLOv7-tiny (leaky variant, Darknet cfg) trained and run in NumPy end to end: Darknet
weights, ELAN and SPP-CSP blocks in forward and backward passes, `new_coords` decoding, the
YOLOv5-style targets and loss (with the OTA assignment of the reference training as an
option), training, then the integer chain. [T26.0](M26-recent-models.md) puts the network
on the inference and hardware chain; this milestone adds the backward passes, the loss and
the training, with no torch at any step.

YOLOv7-tiny is the closest of the three recent networks to the existing code: leaky
activation, Darknet `.cfg` / `.weights` (read by `python/yolo/io/darknet_weights.py`, no
pickle), and only conv / maxpool / route / upsample layers. What it adds: routes of 4
sources, stride-1 maxpools 5 / 9 / 13, logistic heads with `scale_x_y = 2` and
`new_coords = 1`, and a training loss that is not the Darknet v3 loss of M2.

Constraints kept: NumPy only ([ADR 0001](../adr/0001-numpy-pur.md)); M1 layer interface and
gradcheck; the shared pieces of [M33](M33-yolov5n.md) (CIoU, targets, loss, training recipe)
are reused, not copied.

Related backlog: [T26.0](M26-recent-models.md) (cfg, weights, float and integer inference,
export, C-sim), [M24](M24-yolov4-tiny.md) (stride-2 lowering, `scale_x_y`, COCO
evaluation), [M1](M1-briques-numpy.md) (route, maxpool, leaky), [M2](M2-cibles-perte-entrainement.md)
(trainer), [M37](M37-notebooks-recent-models.md) (COCO and VOC notebooks).

**State** (2026-10-22):
- **Done**: T34.0 (layer table, weights) and T34.1 (backward of the routes and stride-1
  pools, ELAN / SPP-CSP gradchecks).
- **T34.2**: the whole network is gradchecked, but COCO is 34.3 AP at 416 against the
  published 35.2, outside the 0.5-point acceptance.
- **T34.3**: OTA loss on the Darknet heads, gradchecked; the torch reference value is
  open.
- **T34.4**: training code and notebook done; the VOC run (≈ 10 days on CPU) is in
  [long-runs.md](long-runs.md).
- **T34.5**: waits for the T34.4 weights.
- The M33 pieces it reuses (T33.4-T33.7) are written.

## Architecture reference

From `yolov7-tiny.cfg` (input 416 in the project, 640 in the reference): every conv is
conv + BN + leaky 0.1, except the three head convs (1×1, bias, `activation=logistic`).

- **Stem**: conv 3×3 / 2 (32), conv 3×3 / 2 (64).
- **ELAN-tiny block** (×4 in the backbone, ×4 in the neck), input $x$, width $c$:
  $a = \mathrm{conv}_{1×1}(x)$, $b = \mathrm{conv}_{1×1}(x)$ (route `-2`),
  $d_1 = \mathrm{conv}_{3×3}(b)$, $d_2 = \mathrm{conv}_{3×3}(d_1)$,
  $y = \mathrm{conv}_{1×1}([d_2, d_1, b, a])$ (route `-5,-3,-2,-1`, then a 1×1 conv).
- **Downsampling**: maxpool 2×2 / 2 between backbone ELAN blocks (strides 8, 16, 32), 3×3 / 2
  convs in the PAN path.
- **SPP-CSP** at stride 32: $u = \mathrm{conv}_{1×1}(x)$, $v = \mathrm{conv}_{1×1}(x)$;
  $[\mathrm{mp}_5(v), \mathrm{mp}_9(v), \mathrm{mp}_{13}(v), v]$ (stride 1, same size) →
  1×1 conv → concat with $u$ (route `-10,-1`) → 1×1 conv.
- **Neck**: FPN (1×1 conv, upsample ×2, route to the backbone, ELAN) then PAN (3×3 / 2 conv,
  route, ELAN); three heads at strides 8, 16, 32 with the YOLOv5 anchors
  (10, 13) … (373, 326), 3 per level.
- **Size**: 2.89 GMAC at 416 (`make count-macs`), 6.2 M parameters.

### [x] T34.0 — Layer table and parameter check
- **Spec**: §3 · **Depends on**: T26.0 (cfg, weights) · **Size**: S
- **Deliverables**: `tiny-yolov7-coco` in `specs.NETWORKS` with per-layer shapes and MACs;
  test that the weight file is read to the last byte (as `test_darknet_weights.py` for v4)
- **Acceptance**: per-layer BFLOPs equal to the Darknet output for this cfg; parameter count
  equal to the published 6.2 M (with C = 80)
- **Notes**: published COCO numbers of the leaky model (YOLOv7 README) recorded with their
  input size, to compare with T34.2.
- **Done**:
  - **Table**: `specs.NETWORKS["tiny-yolov7-coco"]` equals the parsed cfg (99 layers);
    `make count-macs` gives 6,227,101 parameters (6.23 M with C = 80) and 2.89 GMAC at 416,
    i.e. 5.78 GFLOPs. Published (YOLOv7 paper, arXiv 2207.02696, Table 1): 6.2 M and
    5.8 G at 416 (test `test_v7_tiny.py::test_layer_table_matches_published`). Darknet
    itself was not run, so the per-layer BFLOPs are checked through the total only.
  - **Weights**: `weights/yolov7-tiny.weights` (24,967,560 bytes) is read to the last byte:
    20-byte header + 4 bytes per parameter and BN statistic (test).
  - **Published accuracy** of the leaky model (same Table 1, COCO val2017): AP 35.2,
    AP50 52.8, AP75 37.3 at 416; AP 30.8 at 320. Neither README (YOLOv7, AlexeyAB
    Darknet) lists the tiny model. The 38.7 AP at 640 of the paper is the SiLU variant
    (YOLOv7-tiny-SiLU), not these weights: there is no published leaky value at 640.

### [x] T34.1 — Backward of the multi-source route and of the stride-1 pools
- **Spec**: §4 · **Depends on**: T34.0 · **Size**: M
- **Deliverables**: `route_backward` for N sources (split of `dy` by the source widths) in
  `layers/route.py`; maxpool backward for k = 5, 9, 13, s = 1, padding $k/2$ in
  `layers/pool.py`; gradient accumulation in `models/graph.py` for a layer read by several
  later layers (the ELAN input read by the two 1×1 convs and by the route; $v$ read by three
  pools and the route); tests
- **Acceptance**: gradcheck ≤ 1e-7 of one ELAN block and of the SPP-CSP block (small widths,
  8×8 maps); the gradient of a tensor read 4 times equals the sum of the 4 contributions;
  the padded border of a pool never receives a gradient
- **Notes**:
  - Padding of the stride-1 pools is $-\infty$ (Darknet and PyTorch): the border cells take
    the max of the valid window only. Check the forward against a naive window max first.
  - Ties: route the gradient to the first maximum in raster order, consistent with the
    forward argmax (gradient mass conserved, test with constant inputs).
  - The SPPF identity of T26.0 ($\mathrm{mp}_9 = \mathrm{mp}_5 ∘ \mathrm{mp}_5$,
    $\mathrm{mp}_{13} = \mathrm{mp}_5^{∘3}$) holds in the forward; in the backward the two
    forms give the same gradient only without ties: keep the direct form for training, the
    chained one for the hardware.
- **Done**: no code change. `route_backward` already splits `dy` by the N source widths,
  `maxpool_backward` handles k = 5 / 9 / 13 at stride 1 (edge padding, equal to −∞ for a
  max, its gradient folded back on the border cell), and `graph.Network.backward` sums the
  gradient of a map read by several layers. Tests in `test_v7_tiny.py`:
  - the stride-1 forward equals a naive window max over the valid cells, and the backward
    equals a brute-force "first valid maximum in raster order" on distinct and on
    constant inputs (mass conserved, nothing kept on the padded border);
  - gradcheck ≤ 1e-7 of an ELAN block (width 3) and of the SPP-CSP block (width 4) on
    8 × 8 maps, every parameter and the input, train and eval BN;
  - a map read by the 5 / 9 / 13 pools and the route gets the sum of the four
    contributions (graph backward against the hand sum seeded at the map).

### [ ] T34.2 — Float network and COCO mAP
- **Spec**: §8 · **Depends on**: T34.1 · **Size**: M
- **Deliverables**: forward and backward of the whole graph in train mode (batch BN) and
  eval mode; COCO val2017 mAP at 416 and 640 (letterbox of T33.4); row in
  `results/map_coco.md`; `tiny-yolov7-coco_infer` notebook (M14, [M37](M37-notebooks-recent-models.md))
- **Acceptance**: mAP at 640 within 0.5 point of the published value; gradcheck of the full
  network on a 2-image 64×64 batch (sampled indices)
- **Done so far**:
  - **Network**: the Darknet graph trains and evaluates in both BN modes. The whole
    YOLOv7-tiny is gradchecked on 2 images 64 × 64 in train mode, ≤ 1e-7 on sampled
    entries of x and of 6 parameters (stem, ELAN γ, SPP-CSP, PAN, head bias), marked
    `slow` (≈ 1 min). In eval mode an image's output does not depend on its batch.
  - **COCO val2017**, `--preset v5` ([map_coco.md](../../results/map_coco.md)):
    **34.3 AP / 51.6 AP50 at 416** (published 35.2 / 52.8, Table 1 of the paper) and
    34.8 / 52.9 at 640.
  - **Acceptance not met**: the leaky model has no published 640 value (T34.0), and at
    416 the gap is −0.9 AP. The same pipeline gives YOLOv5n within 0.1 point, so the
    gap is not in the decoding. Not measured yet: the Darknet weights against the
    PyTorch `yolov7-tiny.pt` of the paper, and rectangular batches (pad 0.5) as in its
    `test.py`.
  - **Notebook**: `notebooks/coco/tiny-yolov7-coco_infer.ipynb` (`RESIZE = "v5"` by
    default for v7).
- **Notes**:
  - Decoding (logistic already applied by the head conv, `scale_x_y = 2`, `new_coords = 1`):
    $b_{xy} = (2p_{xy} - 0.5 + c_{xy})·s$, $b_{wh} = (2p_{wh})^2·A$, score $= p_o·p_c$: the
    YOLOv5 formulas of T33.4, already in `decode_head` (`wh_new_coords`, `scale_xy`).
  - Since the head applies the logistic, the loss of T34.3 takes the post-sigmoid values:
    write the BCE on probabilities with the logit recovered as $\log(p/(1-p))$ only for the
    backward through the logistic layer, or (simpler) mark the head conv `linear` during
    training and apply the logistic in the loss, as in T33.6.

### [ ] T34.3 — Targets and loss
- **Spec**: §5, §6 · **Depends on**: T34.2, T33.5, T33.6 · **Size**: M
- **Deliverables**: the v5 targets and loss of M33 run on the Darknet graph outputs
  (`yolo_v5_loss` with the v7-tiny hyperparameters); optional OTA assignment
  `build_targets_ota` in `data/targets.py`; tests
- **Acceptance**: loss and gradient on a fixture batch equal to a stored reference to within
  1e-5; gradcheck through the heads; OTA: on hand-built cases, the dynamic-k selection and
  the conflict resolution match the rule below
- **Notes**:
  - Baseline: candidates exactly as T33.5 (`anchor_t = 4`, 3 cells per match), CIoU box
    loss, IoU-target objectness, BCE classes (T33.6).
  - Reference training (`yolov7` repo, `ComputeLossOTA`): the T33.5 candidates are refined
    per image by a SimOTA-style assignment. Cost $C_{ij} = L^{cls}_{ij} + 3·(-\log
    \mathrm{IoU}_{ij})$ between ground truth $i$ and candidate $j$; dynamic $k_i =
    \max(1, \lfloor \sum \mathrm{top}_{10}(\mathrm{IoU}_{i·}) \rfloor)$; each ground truth
    keeps its $k_i$ cheapest candidates; a candidate claimed by several ground truths goes to
    the cheapest one. Check which loss the published tiny weights were trained with
    (`loss_ota` in the hyp file) before choosing the default.
  - Hyperparameters (`hyp.scratch.tiny.yaml`): box 0.05, cls 0.5, obj 1.0, `anchor_t = 4`,
    balance [4.0, 1.0, 0.4]; read them from the file and record them in the notebook.
- **Done**:
  - **Default**: `hyp.scratch.tiny.yaml` (yolov7 repo, read 2026-10-22) has `loss_ota: 1`,
    so the published tiny weights were trained with OTA; `HYP_V7_TINY` in
    `train/loss_v5.py` is `HYP_V5` (box 0.05, cls 0.5, obj 1.0, `anchor_t` 4, balance
    4 / 1 / 0.4, no label smoothing) with `ota: True`. The file also sets mixup 0.05 and
    paste-in 0.05, not implemented (`augment_v5` doc).
  - **Loss**: `yolo_v5_loss` of T33.6 runs unchanged on the Darknet head outputs through
    `v5_levels` (the head convs are linear after `cfg.py`, so the outputs are logits and
    the sigmoid is in the loss: the "mark the head `linear`" option of T34.2).
  - **OTA**: `build_targets_ota` (in `train/loss_v5.py`, it needs the predictions) and
    `ota_match`. The T33.5 candidates of each image (all levels) are decoded in pixels.
    Cost = BCE(logit √(σ_cls σ_obj), one-hot) + 3·(−ln IoU), dynamic k = max(1,
    ⌊Σ top-10 IoU⌋), with conflicts going to the cheapest ground truth. The reference
    `ComputeLossOTA` was not readable verbatim here; this follows its published algorithm.
  - **Tests**:
    - the dynamic k and the conflict rule on hand-built matrices; the OTA matches are a
      subset of the candidates with their image's labels;
    - gradcheck ≤ 1e-7 of the OTA loss through the three heads of the real network,
      every matched cell plus others, with the assignment, IoU targets and CIoU α held
      fixed (`targets_from`).
  - **Not done**: the comparison of the OTA loss with a torch reference value
    (`ComputeLossOTA` of the yolov7 repo). The non-OTA path is checked: `yolo_v5_loss`
    equals the YOLOv5 `ComputeLoss` to 6·10⁻¹⁷ on the `weights/m33/yolov5n_loss.npz` fixture
    (T33.6, 2026-10-22).

### [ ] T34.4 — Training
- **Spec**: §7 · **Depends on**: T34.3, T33.7 · **Size**: M
- **Deliverables**: the M33 recipe (SGD Nesterov, warmup, three parameter groups, EMA,
  mosaic / affine / HSV / flip) on the Darknet graph; `tiny-yolov7-voc_train` notebook (M14,
  [M37](M37-notebooks-recent-models.md));
  run in [long-runs.md](long-runs.md)
- **Acceptance**: one image overfitted (T2.8 criterion); VOC fine-tuning from the COCO
  weights improves on the out-of-domain mAP; float VOC mAP recorded for M15
- **Notes**: the Darknet trainer of M2 (burn-in, steps) stays for the v2 / v3 / v4 networks;
  the v7-tiny path uses the M33 schedule because its weights come from the PyTorch recipe.
- **Done so far** (code; the VOC run is open, [long-runs.md](long-runs.md)):
  - **Trainer**: `TrainerV5` of T33.7 trains the Darknet graph unchanged (`v5_levels`,
    OTA loss by default). `Trainer` (M2) now refuses `new_coords` heads with a pointer to
    it.
  - **CLI**: `tools/train_v5.py --net tiny-yolov7-voc --init coco` writes the VOC cfg
    (`<out>/net.cfg`, `make_cfg`). It copies the Darknet COCO weights except the three
    head convs (layers 89, 93, 97), which get the bias prior, and saves the EMA weights as
    `final.weights`. A smoke run (16 images, 3 batches at 320) trains, saves and evaluates.
  - **Notebook**: `voc/tiny-yolov7-voc_train`.
  - **Overfit**: the one-image overfit of the recipe is the T33.7 test, on a small
    network; YOLOv7-tiny itself is not overfitted.

### [ ] T34.5 — Integer model and hardware chain
- **Spec**: §9, §10 · **Depends on**: T34.2, T26.0 · **Size**: M
- **Deliverables**: integer COCO and VOC mAP of the trained networks; the T26.0 export,
  golden and C-sim passes rerun on the VOC fine-tuned weights
- **Acceptance**: integer gap documented; golden == C-sim on the fine-tuned network
- **Notes**: the 4-source routes are the T26.0 study (`layout: "contiguous"` or
  `MAX_SEGMENTS` raised in `hls/kernels/layer_desc.hpp`); the logistic heads stay on the
  integer sigmoid LUT of T4.4.

## Dependencies

```mermaid
graph LR
  T260[T26.0] --> T340[T34.0] --> T341[T34.1] --> T342[T34.2] --> T343[T34.3] --> T344[T34.4]
  T335[T33.5] & T336[T33.6] --> T343
  T337[T33.7] --> T344
  T342 --> T345[T34.5]
  T260 --> T345
```
