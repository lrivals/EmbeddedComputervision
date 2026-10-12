# mAP of the integer model — Tiny-YOLOv2 VOC (T4.5)

Darknet weights `yolov2-tiny-voc.weights`, VOC2007 test (4,952 images), 416×416 (PIL),
threshold 0.005, NMS 0.45, 11-point AP (`tools/eval_voc.py`, T3.3). Integer model: folded
BN (T4.1), per-channel int8 weights, int8 activations calibrated on 500 VOC2007 trainval
images (p99.99, MSE criterion, `results/calibration_tiny-yolov2-voc.md`), heads at step
1/8, M0/shift requantization, rounded leaky 13/128, LUT decoding (T4.3, T4.4), bit-exact
`IntNetwork` engine (`tools/eval_quant.py`, `make eval-int`).

## Result

| Preprocessing | Float (fused BN) | **Integer INT8** | Gap |
|---|---|---|---|
| stretch (reference T3.4, `make eval-float`) | 56.30 | **55.66** | **−0.64 point** |
| letterbox (the training one) | 54.18 | **53.68** | **−0.50 point** |

The float mAPs are exactly those of `results/map_float.md` (BN folding changes nothing).
The calibration is done in letterbox; it also holds in stretch. The gap is under the
1-point target (conventions.md) in both cases: **QAT (T4.6) is not needed**.
`make eval-int` measures the stretch case.

## Per class (letterbox)

| Class | Float | Integer | Gap |
|---|---|---|---|
| aeroplane | 57.9 | 57.9 | -0.0 |
| bicycle | 67.8 | 66.7 | -1.1 |
| bird | 47.2 | 46.3 | -0.9 |
| boat | 36.5 | 37.7 | +1.2 |
| bottle | 21.2 | 21.7 | +0.5 |
| bus | 67.7 | 66.6 | -1.1 |
| car | 63.0 | 62.6 | -0.4 |
| cat | 66.4 | 65.9 | -0.5 |
| chair | 31.3 | 31.5 | +0.2 |
| cow | 53.5 | 51.3 | -2.3 |
| diningtable | 55.4 | 54.5 | -0.9 |
| dog | 62.0 | 61.0 | -1.0 |
| horse | 70.1 | 69.7 | -0.4 |
| motorbike | 69.8 | 68.8 | -1.0 |
| person | 58.3 | 57.7 | -0.6 |
| pottedplant | 26.8 | 28.1 | +1.4 |
| sheep | 50.7 | 50.1 | -0.6 |
| sofa | 51.8 | 51.6 | -0.1 |
| train | 68.1 | 67.5 | -0.6 |
| tvmonitor | 58.1 | 56.3 | -1.8 |

## Per-layer sensitivity

A single convolution quantized at a time (weights fake-quantized per channel, leaky 13/128,
output fake-quantized at s_y), the others in float; first 1,000 images of VOC2007 test
(float simulation `fake_quant_forward`, faster than the integer model).

| Quantized layer | mAP | Gap to float |
|---|---|---|
| none (float) | 56.19 | — |
| L00 | 56.12 | -0.07 |
| L02 | 56.21 | +0.02 |
| L04 | 55.94 | -0.25 |
| L06 | 56.21 | +0.02 |
| L08 | 55.84 | -0.34 |
| **L10** | 55.58 | -0.61 |
| L12 | 56.13 | -0.06 |
| L13 | 56.06 | -0.13 |
| L14 | 56.38 | +0.19 |
| all (simulation) | 55.64 | -0.55 |

Most sensitive layer: **L10** (−0.61 point alone), then L08 and L04. The per-layer gaps
(≤ 0.6 point) are of the same order as the noise of a 1,000-image subset (L14 alone
*gains* 0.19): no layer dominates the loss, which is spread over the network.

## Head scale

First 500 images of VOC2007 test, integer model (`IntNetwork`); only the scale of the head
(L14) changes.

| Head scale | Range of t | Clipped outputs | Integer mAP |
|---|---|---|---|
| 1/16 (LUT step of §9.4) | ±7.94 | 2.5 % | 49.66 |
| **1/8 (selected)** | ±15.9 | 0.03 % | **57.26** |
| calibrated p99.99 (0.149) | ±18.9 | 0.01 % | 56.59 |
| float | — | — | 57.24 |

At step 1/16, the class logits of Tiny-YOLOv2 (|t| up to ~19) saturate; several classes
reach the same maximum and the softmax flattens: the scores cap around 0.45 and the mAP
loses ~7.5 points. At step 1/8 the 256-entry table covers ±15.9; the error of σ due to the
input step goes from 1/128 to 1/64, with no measurable effect on the mAP. The exponential
table is then in Q12 (`exp_frac`) to fit in 32 bits.

## Arithmetic choices that matter

- **Rounded leaky.** `(13·y + 64) >> 7` rather than `(13·y) >> 7` (§9.3): the floor biases
  each negative value by −½ step and the bias accumulates from layer to layer (mean gap at
  the heads ×1.7 on a test image).
- **Heads at step 1/8** (above).
- Activations at p99.99 of |x|: p90-p99 clip too much (MSE ×10 to ×1000), the max wastes
  range (MSE ×2 to ×5).
