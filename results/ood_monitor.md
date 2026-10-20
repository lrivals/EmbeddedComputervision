# Out-of-distribution monitor — Tiny-YOLOv2 VOC (T25.5)

The monitor flags inputs that are far from the training domain (VOC), using only the backbone
features of the detector: the int8 output of **L13** (last backbone conv, 1024 × 13 × 13),
dequantized by its scale. The accelerator already writes this map, so the monitor runs on the
ARM after the backbone (`python/yolo/infer/ood.py`, `tools/ood_monitor.py`).

Conditions:

- **Model**: integer model (`IntNetwork`), `stretch` preprocessing, threshold 0.005, NMS 0.45.
- **Fit**: 1,500 images of VOC2007 trainval. The **reference** set is 500 more trainval
  images, held out from the fit. All 2,000 are drawn with seed 0.
- **In-domain test**: the first 500 images of VOC2007 test.
- **Out of domain** (domain shifts of T11.2):
  - 500 images of the ExDark test split (low light);
  - 500 images of FLIR ADAS v2 thermal val.

  Both sets are drawn with seed 0.
- **mAP**: VOC07 (AP50), on the classes each set shares with VOC (`datasets.eval_view`), for
  ExDark and FLIR too.

```
tools/m25.sh ood        # = the three commands below (≈ 10 min, 16 jobs)
python tools/ood_monitor.py extract --jobs 16
python tools/ood_monitor.py fit
python tools/ood_monitor.py eval
```

## Scores

Three raw scores per image, computed from the pooled features (global average and max, 2,048
values) or from the map:

| Score | What it measures | Fit |
|---|---|---|
| norm | L2 norm of the pooled features (energy of the map) | none |
| Mahalanobis (baseline) | distance to the mean of the pooled fit features | Ledoit-Wolf shrinkage 0.027 (2,048 dims, 1,500 images) |
| autoencoder | mean squared reconstruction error of the standardized map | per-cell 1×1 conv autoencoder 1024 → 128 → 32 → 128 → 1024, leaky, AdamW lr 1e-3, 2,000 iterations of 16 images (loss 1.19 → 0.57) |

The autoencoder uses the NumPy layers of `python/yolo/layers` (`Network`, `conv_forward` /
`conv_backward`), with a gradient check in `python/tests/test_ood.py`.

**Finding: the domain shift lowers the activations.** Medians of the raw scores:

| Set | norm | Mahalanobis | autoencoder |
|---|---|---|---|
| reference (VOC trainval, held out) | 85.9 | 61.1 | 0.575 |
| VOC test | 85.2 | 60.2 | 0.567 |
| ExDark | 76.5 | 56.5 | 0.493 |
| FLIR | 74.3 | 55.3 | 0.481 |

- Dark and thermal images give weaker features. In 2,048 dimensions, the in-domain features
  lie on a shell around their mean. The OOD features fall *inside* that shell: closer to the
  mean, and easier to reconstruct.
- A raw score is therefore **lower** out of domain (AUROC below 0.5).
- The monitor uses the **typicality** of each score, |s − s_ref|, with s_ref the median on
  the held-out reference images. A score far from the typical value in either direction is
  flagged (Nalisnick et al. 2019, typicality test).

## Detection of the domain shift

| Score | AUROC ExDark (raw) | AUROC FLIR (raw) | AUROC ExDark (typicality) | AUROC FLIR (typicality) | Spearman score / mAP (15 groups) | MACs (share of the detector) |
|---|---|---|---|---|---|---|
| **norm** | 0.195 | 0.086 | **0.744** | **0.884** | **−0.66** | 0.35 M (0.01 %) |
| Mahalanobis | 0.321 | 0.213 | 0.669 | 0.726 | −0.41 | 4.54 M (0.13 %) |
| autoencoder | 0.279 | 0.183 | 0.686 | 0.718 | −0.14 | 45.69 M (1.31 %) |

How to read the table:

- **AUROC**: VOC test (negative) against the OOD set (positive); 0.5 is chance.
- **Spearman correlation**: per set, the images are split into five quintiles of the score.
  The correlation is taken between the mean score and the mAP of the 15 groups; negative
  means a higher score goes with a lower mAP.
- **MACs**: share of the 3,486 M MACs of the detector.

What it shows:

- **The autoencoder adds nothing over the simple scores.** It detects no better than
  Mahalanobis and worse than the norm. It costs 10× the Mahalanobis distance and 130× the
  norm, and its score barely follows the mAP (−0.14). The task note asked for this check against the baseline
  first: the answer is negative.
- **The norm of the features is the best monitor here.** AUROC 0.88 on thermal and 0.74 on
  low light, for 0.01 % of the detector. It also follows the mAP best (−0.66).
- **Low light is the harder shift.** Many ExDark images are only dim, and the backbone sees
  them as ordinary scenes.

## Score against the mAP per group

mAP (%) of the five quintiles of the typicality score in each set. Q1 has the most typical
images, Q5 the least typical.

| Score | Set | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|
| norm | VOC (mAP 57.7) | 72.4 | 50.2 | 63.7 | 61.3 | 55.9 |
| norm | ExDark (mAP 28.8) | 35.7 | 27.6 | 32.8 | 28.5 | 26.9 |
| norm | FLIR (mAP 4.9) | 5.1 | 7.7 | 5.4 | 3.9 | 2.6 |
| Mahalanobis | VOC | 67.1 | 71.7 | 62.4 | 61.9 | 52.6 |
| Mahalanobis | ExDark | 32.7 | 33.0 | 23.9 | 29.1 | 36.0 |
| Mahalanobis | FLIR | 5.0 | 4.7 | 6.2 | 4.5 | 3.0 |
| autoencoder | VOC | 48.9 | 57.8 | 61.7 | 65.9 | 63.3 |
| autoencoder | ExDark | 29.1 | 36.7 | 31.4 | 27.2 | 29.5 |
| autoencoder | FLIR | 4.9 | 4.2 | 5.5 | 5.4 | 5.2 |

Reading:

- Most of the correlation comes from the gap **between** sets (VOC 57.7, ExDark 28.8,
  FLIR 4.9).
- **Within** a set, 100 images per group give a noisy mAP. Only the norm and Mahalanobis
  show the expected trend at the ends (Q1 > Q5 for VOC and FLIR), and not monotonically.
- A per-image score is therefore a domain alarm, not a predictor of the accuracy on one
  image.

## T29.0: six monitors compared (2026-10-20)

`python tools/ood_monitor.py compare` uses the same features, sets, fit split (1,500
trainval images) and held-out reference (500) as T25.5. It adds three monitors of the
literature, computed on the **integer** features: per-channel sum (int32) and max (int8) of
the int8 L13 map, 2,048 values (`ood.pool_int`).

| Monitor | Computes | Fit |
|---|---|---|
| max_score | 1 − top detection score of the image (MSP of Hendrycks 2017 for a detector) | none |
| boxes | fewest neurons outside a [min, max] box, over 4 k-means clusters (Henzinger 2020) | 4 boxes × 2,048 neurons |
| patterns | Hamming distance to the nearest binary pattern of the fit; the 64 pooled neurons of largest variance, on when above their median (Cheng 2018) | 1,500 patterns (all distinct) |

AUROC of VOC test against each shifted set: raw score, then typicality |s − median of the
reference|. FPR@95: share of VOC test images flagged when the threshold flags 95 % of the
shifted set, for the better of the two forms. Memory: stored state. Ops: operations per image
after the pooling; compares, XOR and popcount count one each.

| Monitor | AUROC ExDark raw / typ. | AUROC FLIR raw / typ. | FPR@95 ExDark | FPR@95 FLIR | Memory | Ops per image |
|---|---|---|---|---|---|---|
| norm (T25.5) | 0.195 / 0.744 | 0.086 / 0.884 | 85.2 % (typ) | 38.4 % (typ) | 0 B | 2,048 |
| Mahalanobis (T25.5) | 0.321 / 0.669 | 0.213 / 0.726 | 85.8 % (typ) | 81.0 % (typ) | 16.8 MB | 4.2 M |
| autoencoder (T25.5) | 0.279 / 0.686 | 0.183 / 0.718 | 86.8 % (typ) | 84.0 % (typ) | 1.1 MB | 45.7 M |
| **max_score** | **0.809** / 0.794 | 0.801 / 0.775 | **70.0 %** (raw) | 68.0 % (raw) | 0 B | 1 |
| boxes | 0.642 / 0.541 | 0.624 / 0.449 | 91.0 % (raw) | 79.6 % (raw) | 64 KB | 16,384 |
| **patterns** | 0.777 / 0.639 | **0.933** / 0.863 | 75.4 % (raw) | **29.4 %** (raw) | 12.5 KB | 6,064 |

The rows of T25.5 reproduce `eval.json` to the third decimal.

Reading:

- **No monitor wins on both shifts.**
  - On low light (ExDark), the detector's own confidence is the best: AUROC 0.81 at the
    cost of one compare.
  - On thermal images (FLIR), the activation patterns are the best: AUROC 0.93, 29 % false
    alarms at 95 % detection, against 38 % for the norm.
  - A pair (max_score, patterns) costs ≈ 6,000 operations and 12.5 KB.
- **Mahalanobis and the autoencoder** are dominated by cheaper monitors on both shifts. They
  cost 4.2 M and 45.7 M operations.
- **Abstraction boxes** are compare-only (the most hardware-friendly logic), but weak: with
  2,048 neurons, a VOC test image also falls outside every box on some neurons.
- **The new monitors need no typicality**: their raw form is the better one (a low score =
  in domain). The norm, Mahalanobis and the autoencoder only work as typicality (§ above).
- **False alarms stay high**: none of them is an alarm with an acceptable false-alarm rate
  on its own. FPR@95 ≥ 29 %.
- **Next (T29.1)**: the question that matters for safety is whether the monitor flags the
  images where a person is missed. That needs the person ground truth of M27.

## Remaining

- Monitor on L12 instead of L13 (`extract --layer 12`), and on the HIT-UAV and drone sets of
  M18.
- Raising an alarm needs a threshold. For example, the 95th percentile of the typicality on
  the reference set; its false-alarm rate is then measured on VOC test.
