# 4-bit quantization — Tiny-YOLOv2 VOC (M9.3)

"w4a4" scheme (`python/yolo/quant/lowbit.py`):

- **Weights**: all the convs on 4 bits, symmetric per channel, integers in [−7, 7].
- **Activations**: conv outputs on 4 bits (`qmax` = 7 in the manifest), with a **power-of-2
  step** [2024-yan#006.0].
- **Exceptions**:
  - the head stays on 8 bits at the 1/8 step of the LUTs (§9.4);
  - the input stays on 8 bits (pixels).
- **First layer**: it can be computed by two convolutions with 4-bit inputs,
  x = 16·(x ≫ 4) + (x & 15) [2024-yan#007.3]. The equality is checked exhaustively
  (`golden::mul_split4`, `int_layers.conv_acc_split4`).

VOC2007 test, 4,952 images, stretch preprocessing, threshold 0.005, 11-point VOC07 mAP.

## Results

| Model | mAP | Gap to INT8 |
|---|---|---|
| INT8 PTQ (reference, `results/map_int8.md`) | 55.66 | — |
| w4a4 PTQ (calibrated 2^j steps, minimum squared error, 500 trainval images) | 17.06 | −38.6 |
| **w4a4 QAT, 600 iterations** | **36.38** | −19.3 |

**QAT conditions** (`tools/train.py --qat w4a4`, `python/yolo/quant/fake_quant.py`):

- network with folded BN, initialised from the Darknet weights, with the steps of the PTQ;
- SGD with lr 1e-4 [2026-fata#015.3], ramp over 50 iterations, batches of 8 at 416;
- VOC07+12 trainval, **only 600 iterations** (4,800 images, ≈ 0.3 epoch, ≈ 1 h on CPU):
  the time budget was cut on purpose.
- The mean loss goes from 16.0 (first 50 iterations) to 10.1 (last 50). It is still
  decreasing: a longer QAT should improve the mAP.

The QAT recovers 19 points over the PTQ (17.1 → 36.4), but stays far from INT8: with 0.3
epoch, it is only the start of a fine-tuning. 2024-yan reaches 4 bits on YOLOv5s with a
full training [2024-yan#006.0].

**Faithfulness to the hardware.** The fake-quant follows the order of the integer model:
rounding to the step, then integer leaky (13r + 64) ≫ 7, then clipping. The gradients go
through the straight-through estimator, and through LSQ for the step.

- Gradcheck of the surrogate function: `test_fake_quant.py`.
- On a random network, more than 95 % of the outputs are equal to the integer of the
  exported model.
- The exponent of the step is learned without the 1/√(n·qmax) factor of LSQ: with it, it
  did not move (|δk| ~ 1e-8 per iteration).

**Integer chain checked.** The exported 4-bit model
(`tools/quant_lowbit.py --scheme w4a4 [--checkpoint …]`) has a `qmax` per layer in the
manifest. The C++ golden model and the HLS C-sim (`tb_net`) reproduce it **to the byte**
(0 differences on 3 images).

## Estimated hardware gains (outside the base, synthesis to do)

| | INT8 | 4 bits |
|---|---|---|
| weight memory | 15.9 MB | 7.9 MB (packing of 2 weights per byte, not done in the engine yet) |
| MACs per DSP | 1 | 2 (two 4-bit products per DSP48, like the packing of 2024-zhang) |
| single engine Tm·Tn = 768 | 768 DSP | 384 DSP, or Tm·Tn doubled at constant DSP |
| KV260 streaming (`results/streaming.md`) | 32.1 img/s | 64.2 img/s |

## Remaining

- A longer QAT: several epochs, or a progressive ramp 8 → 6 → 4 bits.
- 4-bit packing in the engine: int8 container today, hence no bandwidth gain.
- Packing of 2 MACs per DSP in the HLS: done in C-sim (M23, [dsp_packing.md](dsp_packing.md));
  synthesis remains.

Figures: `results/figures/resultats/map_formats.png` (T13.14) and `entrainement_qat-w4a4.png` (T13.26), `python -m tools.figures map_formats entrainement`.
