# M4 — Quantization and bit-exact integer model

Goal: a Python model that computes in integers only, exactly as the hardware will, with a
measured mAP loss, then the export that serves as the contract for the C++ and HLS stages.

### [x] T4.1 — BN folding
- **Spec**: §9.1 · **Depends on**: T1.8 · **Size**: S
- **Deliverables**: `python/yolo/quant/fuse_bn.py` + tests
- **Acceptance**: folded network == network with BN (inference mode) to 1e-12 in float64;
  mAP unchanged
- **Done**: `fuse_bn` (§9.1) and `fuse_network` (same graph, convs without BN with a
  bias); equality to 1e-12 (relative to the max) on v2 and v3 with random BN, same
  detections with the Darknet weights. `Network.forward(all_outputs=True)` returns all the
  maps (calibration).

### [x] T4.2 — Weight quantization and calibration
- **Spec**: §9.2 · **Depends on**: T4.1, T3.3 · **Size**: M
- **Deliverables**: `python/yolo/quant/{quantize,calibrate}.py`
- **Acceptance**: symmetric int8 weights **per output channel**; per-layer activation
  scales calibrated on ~500 images; comparison of the 90th, 95th and 99th percentiles and of
  the max, choice documented
- **Notes**: as 2026-fata does, choose the clipping that minimises the FP32/INT8 error
- **Done**: `quantize.py` (rounding ⌊v+½⌋, per-channel weights s_w = max|W_f|/127, int32
  bias, M0 < 2³¹ with n ≤ 31 maximal per layer, input at 1/127); `calibrate.py` +
  `tools/calibrate.py` (`make calibrate`): 500 randomly drawn VOC2007 **trainval** images,
  4,096 values sampled per image and per layer, candidates
  p90/p95/p99/p99.9/p99.99/max, FP32/fake-INT8 MSE criterion: p99.99 selected everywhere
  (p90-p99 clip too much, the max wastes range) — `results/calibration_<net>.md`. Route: a
  common scale imposed on the sources (v3: {L08, L18}, the largest). **Heads**: fixed scale
  1/8 rather than 1/16 (see T4.4).

### [x] T4.3 — Integer layers
- **Spec**: §9.3 · **Depends on**: T4.2 · **Size**: L
- **Deliverables**: `python/yolo/quant/int_layers.py`, `python/yolo/quant/int_model.py` +
  tests
- **Acceptance**: int8 conv with an int32 accumulator (overflow tested: 30 bits at worst
  for layer 13); requantization $(acc \cdot M_0 + 2^{n-1}) \gg n$; leaky
  $(y \cdot 13) \gg 7$; s2/s1 maxpool and upsample in integers; gap to the float layer
  ≤ 2 % of the dynamic range
- **Notes**:
  - **route**: the two inputs of the concatenation (layers 19 and 8) must share **the same
    scale**, otherwise the "free" concatenation of §10.3 is impossible. Impose the common
    scale during calibration
  - rounding and arithmetic shift on signed integers: define the behaviour for negatives
    precisely, C++ and HLS will have to reproduce it
  - everything in `np.int64`, with no float at all in the forward pass
- **Done**: `int_layers.py` (`rshift_round`, `conv_int`, `leaky_int`…) and `int_model.py`
  (`QuantModel`, `IntNetwork`, `fake_quant_forward`); contract written in
  `docs/conventions.md` ("Integer arithmetic"). Gap to the float layer ≤ 1.5 % of the
  dynamic range (max) with the Darknet weights; layer 13: bound
  Σ|q_w|·127 + |q_b| = 2^25.7 (generic worst case 9,216 × 127² < 2³⁰); −128 never
  produced. `f64` engine (integer products through BLAS, exact because |acc| < 2⁵³) bit for
  bit equal to the `int64` engine, 4× faster.
  **Deviation from the spec**: rounded leaky `(13y + 64) >> 7` instead of `(13y) >> 7`;
  the floor biases each negative output by −½ step and the bias accumulates (mean gap at
  the heads ×1.7 on one image).

### [x] T4.4 — Sigmoid and exponential by LUT
- **Spec**: §9.4 · **Depends on**: T4.3 · **Size**: S
- **Deliverables**: `python/yolo/quant/lut.py`, integer decoding in `decode.py`
- **Acceptance**: 256-entry LUT, max error on σ ≤ 0.0075 (step 1/16); thresholding on the
  logit $t_o > \ln\frac{\theta}{1-\theta}$ identical to thresholding after the sigmoid
- **Done**: `lut.py` (σ and e^{t} indexed by `q + 128`, e^{d·s} for the three-stage v2
  softmax, Q16 values; e^{t} in Q`exp_frac` to fit in 32 bits), `decode_head_int`/
  `decode_int` (threshold on the integer t_o then tables, boxes in float on the host side)
  and `postprocess_int`. Threshold on the logit identical to σ > θ and to LUT > θ for every
  q. Error of the table at the grid points ≤ 2⁻¹⁷.
  **Deviation from the spec**: end to end, the max error is σ'(0)·s/2 = 1/128 ≈ 0.0078 at
  step 1/16 (not 0.0075). Above all, step 1/16 saturates the logits of Tiny-YOLOv2
  (p99.99 of |t| ≈ 19): 2.5 % of the outputs clipped, flattened softmax, mAP 49.7 against
  57.2 in float (500 images, letterbox). The heads are therefore at step **1/8** (σ error
  ≤ 1/64 ≈ 0.016, clipping 0.03 %), mAP intact; a calibrated scale (p99.99, 0.149) gives
  56.6.

### [x] T4.5 — mAP of the integer model
- **Spec**: §10.5 step 2 · **Depends on**: T4.4, T3.4 · **Size**: S
- **Deliverables**: `results/map_int8.md`
- **Acceptance**: integer mAP and gap to the float model reported, most sensitive layer
  identified (quantizing one layer at a time)
- **Notes**: indicative target: less than one point of loss (2025-kim: -0.2 % in INT16)
- **Done**: `tools/eval_quant.py` (`make eval-int`; variants `float`, `int`, `fq:all`,
  `fq:<id>`, in parallel) → `results/map_int8.md`. Full VOC2007 test, reference
  preprocessing (stretch): float **56.30**, integer **55.66**, i.e. **−0.64 point**;
  letterbox: 54.18 → 53.68 (−0.50). Most sensitive layer alone: L10 (−0.61 on 1,000
  images, at noise level; none dominates).

### [x] T4.6 — QAT (if needed) — not applicable
- **Spec**: §9.2 (PTQ or QAT) · **Depends on**: T4.5, T2.7 · **Size**: L
- **Deliverables**: `python/yolo/quant/fake_quant.py` (STE), QAT mode of the trainer
- **Acceptance**: gradcheck of the fake-quant with STE; mAP gap reduced below the target
- **Notes**: to trigger only if T4.5 exceeds the target; lr = 1e-4 (2026-fata)
- **Decision**: not triggered, the PTQ loss (−0.64 point in stretch, −0.50 in letterbox) is
  below the 1-point target. To revisit if a fine-tuned network (T2.9) or a lower bit width
  exceeds it.

### [x] T4.7 — Export of the integer model
- **Spec**: §10.5 steps 2-3 · **Depends on**: T4.3, T0.4 · **Size**: M
- **Deliverables**: `python/yolo/io/export.py`, `tools/export_model.py`, output in
  `model/`
- **Acceptance**: valid manifest (T0.4 schema); blobs read back and fed into the Python
  integer model == original output; dumps `L00.npy`… for 3 test images
- **Done**: `export.py` (generic `layout`: A/B ping-pong, `S` for a map read twice, `R` for
  the routes, `H<grid>` for the heads; it reproduces `model/example_manifest.json` exactly,
  whose generator has become a thin wrapper; `export_model`, `load_model`, `load_luts`),
  `tools/export_model.py` (`make export`) → `model/tiny-yolov2-voc/` and
  `model/tiny-yolov3-coco/` (COCO weights: no Tiny-YOLOv3 VOC weights before T2.9). Schema
  extended in a compatible way: `region` layer, non-integer anchors (34.56 px),
  `luts.bin`, `lut_offset`/`exp_frac` per head. Dumps `dumps/<id>/{input,L00…}.npy` +
  `detections.json` for VOC2007 test 000001-000003.
