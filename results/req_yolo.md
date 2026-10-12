# REQ-YOLO: weights on powers of 2 — Tiny-YOLOv2 VOC (M9.2)

Reproduction of the quantization of 2019-ding [2019-ding#008.1, #009.1, #009.4]. The
activations stay in **calibrated INT8**, like the reference model: only the effect of the
weights is measured.

**Levels** (`python/yolo/quant/pow2.py`):

- `mixed6`: 6-bit weights, |q| ∈ {0, 2^a, 2^a + 2^{a−k}}, with a ≤ 6 (3 primary bits) and
  k ∈ {1, 2, 3} (2 secondary bits). That makes 45 signed levels, |q| ≤ 96. The product q·x
  is ±((x ≪ a) + (x ≪ (a − k))): **two shifts and an addition**.
- `uniform6`: equidistant levels ±31.

In both cases, α is chosen per output channel: minimum squared error over 32 scales.

## Results (VOC2007 test, 4,952 images, stretch)

| Weights | Method | mAP | Gap to INT8 |
|---|---|---|---|
| INT8 per channel (reference) | PTQ | 55.66 | — |
| uniform6 (±31) | PTQ | 54.35 | −1.31 |
| **mixed6 (powers of 2)** | PTQ, direct projection | 52.46 | −3.20 |
| mixed6 | ADMM, 600 iterations, then projection | 43.07 | −12.59 |

**ADMM** (`python/yolo/train/admm.py`, `tools/train.py --admm`):

- folded weights, mixed6 on all the convs;
- ρ = 1e-3 at the start, × 1.3 at each Z/U step (every 50 iterations);
- lr 1e-4, batches of 8, 600 iterations.

**The ADMM did not converge.** The per-layer residual ‖W − Z‖/‖W‖
(`build/train/admm-mixed6/admm.csv`) should tend to 0. Instead, it goes from 0.05-0.06
(iteration 50) to 0.07-0.09 (iteration 600). Final ρ: 0.023, too weak against the gradient
of the detection loss, and the weights are not attracted to the levels.

**The result is worse than the direct projection**: 43.07 against 52.46. The detection
loss itself rose during the fine-tuning, from 7.3 to 8.1 on average over 50 iterations:
the 600 iterations degraded the network instead of bringing it closer to the levels.

Probable causes, **not verified** (the float mAP of the checkpoint was not measured):

- fine-tuning of the network **with folded BN**, without BN, with the augmentations of the
  loader, from Darknet weights that had already converged;
- batches of 8, too small for this network without BN;
- ρ too weak to compensate.

The algorithm itself is validated on a quadratic problem (`test_admm.py`: residual
< 1e-3). As things stand, the best powers-of-2 option remains the PTQ at 52.46.

Planned fixes:

- first measure the float mAP of a fine-tuning without ADMM, to isolate the effect of the
  fine-tuning;

- a starting ρ 10 to 100 times larger (0.05 → 1);
- more iterations;
- choice of the scheme per layer from the sensitivity of each layer (study prepared,
  cancelled for lack of time).

## Hardware (T9.2.4)

**Bit-exactness.** The mixed6 weights fit in the int8 of the engine. The existing golden
model and kernel stay exact without changes: `tb_net`, 0 differences on 3 images.

**Variant `ACC_WMODE_POW2`** (`hls/kernels/accel.hpp`, `mul_w`): the PE replaces each
multiplication by two shifts and an addition.

- C-sim (`tb_net_pow2`, ctest): **0 differences** with the golden model on the 3 images of
  the mixed6 model.
- DSP savings **estimated** on the single KV260 engine (Tm·Tn = 768):

| | INT8 | mixed6 (shifts) |
|---|---|---|
| MAC DSPs | 768 | **0** |
| requantization DSPs | 128 (4·Tm) | 128 |
| LUTs per MAC (order of magnitude, outside the base) | ≈ 0 (DSP) | ≈ 40 (2 7:1 shifters on 8 bits + 15-bit adder) |
| MAC LUTs | — | ≈ 31,000, i.e. 26 % of the 117,120 LUTs of the XCK26 |

The 768 freed DSPs can serve to double Tm·Tn, or be left free on a smaller chip.
2019-ding (Virtex-7, FFT + powers of 2) reports 314 images/s [2019-ding, Tables 1-2], but
on another architecture.

## Remaining

- A converged ADMM (larger ρ, more iterations, per-layer plan), without degrading the
  float network.
- Synthesis of both PE variants: DSP, LUT and timing measured (`make hls-synth` and
  `make hls-synth-pow2`; the figure `results/figures/resultats/ressources.png`, T13.24,
  then moves from estimate to measurement).
- 6-bit codes stored in memory instead of int8 (25 % gain on the weights).

Figures: `results/figures/resultats/map_formats.png` (T13.14) and `entrainement_admm-mixed6.png` (T13.26, ADMM residuals), `python -m tools.figures map_formats entrainement`.
