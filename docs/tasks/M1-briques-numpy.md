# M1 — NumPy building blocks (forward and backward passes)

Goal: all the layers of §4, checked by finite differences, then assembled into Tiny-YOLOv2
and Tiny-YOLOv3. Common interface: `forward(x) -> (y, cache)`,
`backward(dy, cache) -> (dx, grads)`.

### [x] T1.1 — Gradient check utility
- **Spec**: §11 · **Depends on**: T0.2 · **Size**: S
- **Deliverables**: `python/yolo/testing/gradcheck.py`, `python/tests/test_gradcheck.py`
- **Acceptance**: centred differences h = 1e-6 in float64; relative error
  $|a-n|/\max(|a|,|n|,\epsilon)$; validated on $f(x)=\sum x^3$
- **Notes**: check by sampling indices for large tensors
- **Done**: $|\cdot|$ = max norm over the tensor (an accidentally small gradient would make
  the element-wise error dominated by the rounding noise of h = 1e-6); the loss can be
  returned term by term so that the unaffected terms cancel exactly. `check_layer` runs the
  gradcheck of a layer with $L=\sum y\cdot R$.

### [x] T1.2 — 2D convolution (naive + im2col)
- **Spec**: §4.1 · **Depends on**: T1.1 · **Size**: M
- **Deliverables**: `python/yolo/layers/conv.py`, `python/tests/test_conv.py`
- **Acceptance**: gradcheck ≤ 1e-7 on $W$, $b$, $X$ (k = 1 and 3, s = 1 and 2);
  im2col == naive to 1e-12; time of a 13×13×1024→1024 conv measured
- **Notes**: the naive version stays as the test reference; im2col is the one used in
  training
- **Measurement** (`python tools/bench_conv.py`, Intel Core Ultra 7 155H, OpenBLAS): batch
  1, 1.59 GMAC — float32: forward 13 ms (≈ 120 GMAC/s), backward 37 ms; float64: forward
  30 ms, backward 58 ms. Batch 8 in float32: forward 107 ms, backward 226 ms.

### [x] T1.3 — Batch normalization
- **Spec**: §4.2 · **Depends on**: T1.1 · **Size**: M
- **Deliverables**: `python/yolo/layers/batchnorm.py` + tests
- **Acceptance**: gradcheck ≤ 1e-7 on $\gamma$, $\beta$, $x$; inference mode with the
  running averages; test that the gradient of a conv bias before BN is ≈ 1e-15
- **Notes**: momentum of the running averages and $\varepsilon$ as parameters

### [x] T1.4 — Leaky ReLU
- **Spec**: §4.3 · **Depends on**: T1.1 · **Size**: S
- **Deliverables**: `python/yolo/layers/activations.py` + tests
- **Acceptance**: gradcheck ≤ 1e-7 (avoiding the points $x = 0$)

### [x] T1.5 — Max pooling stride 2 and stride 1
- **Spec**: §4.4 · **Depends on**: T1.1 · **Size**: M
- **Deliverables**: `python/yolo/layers/pool.py` + tests
- **Acceptance**: gradcheck ≤ 1e-7; stride 1 keeps 13×13; gradient routed to the argmax
  only
- **Notes**: stride 1 = padding on the right and at the bottom by replicating the border
  (layer 11, outside the base); test with distinct values to avoid argmax ties

### [x] T1.6 — Upsample, route, sigmoid
- **Spec**: §4.5 · **Depends on**: T1.1 · **Size**: S
- **Deliverables**: `python/yolo/layers/{upsample,route}.py`, stable `sigmoid` in
  `activations.py` + tests
- **Acceptance**: gradcheck ≤ 1e-7; sigmoid without overflow for $t = \pm 1000$
- **Notes**: the route splits $\delta y$; the accumulation of the gradients of a reused map
  is handled by the graph (T1.7)

### [x] T1.7 — Network graph and `.cfg` parser
- **Spec**: §3, §4.5 · **Depends on**: T1.2 to T1.6 · **Size**: M
- **Deliverables**: `python/yolo/models/graph.py` (forward execution, caches, backward in
  reverse order, **gradient accumulation** for outputs consumed twice),
  `python/yolo/models/cfg.py` (Darknet parser)
- **Acceptance**: gradcheck of the mini-network of §11
  (conv-BN-leaky-pool-conv-BN-leaky-pool(s1)-route-conv1×1) ≤ 1e-7
- **Notes**: in Tiny-YOLOv3, layer 13 feeds layer 14 and route 17

### [x] T1.8 — Tiny-YOLOv2 and Tiny-YOLOv3
- **Spec**: §3.1, §3.2 · **Depends on**: T1.7, T0.5 · **Size**: S
- **Deliverables**: `python/yolo/models/tiny_yolo.py`, `.cfg` files in
  `python/yolo/models/cfg/`
- **Acceptance**: output shapes and parameter count of each layer == tables of §3; 416×416
  forward pass without error; He initialisation (§7.1)
- **Notes**: the Darknet `.cfg` files are outside the base: cite them as such; check the
  built network against `yolo.models.specs` (T0.5), already tested against the tables of
  §3

### [~] T1.9 — Darknet weight loader
- **Spec**: §7.1 (pre-training), §10.5 step 1 · **Depends on**: T1.8 · **Size**: M
- **Deliverables**: `python/yolo/io/darknet_weights.py`
- **Acceptance**: loading of `yolov3-tiny.weights` (COCO) and `yolov2-tiny-voc.weights`;
  visually correct detection on a test image (with T3.1-T3.2)
- **Notes**: Darknet order per layer: [bias or β], [γ, μ, σ²] if BN, then weights; header
  of 4 or 5 words depending on the version
- **Done**: `tools/get_weights.sh`; both files load while consuming exactly all the bytes
  (`yolov2-tiny-voc.weights`: 4-word header; `yolov3-tiny.weights`: 5 words, with the COCO
  `cfg/yolov3-tiny.cfg`). Quick check without NMS on the first 12 images of the VOC 2007
  test set: Tiny-YOLOv2 finds at least one ground-truth class on 9.
  **Remaining**: the visual check of the boxes, after T3.1-T3.2.
