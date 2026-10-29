# Decision explanation (M36)

Why the detector outputs a box, and why it misses one, at the float, integer and golden
stages. Tasks: [docs/tasks/M36-explanation.md](../docs/tasks/M36-explanation.md); tools and
viewer: [docs/explain.md](../docs/explain.md). The tables between the
`<!-- explain:… -->` markers are written by `python tools/explain.py report` from
`build/explain/*.json`; the text around them is written by hand.

**Conditions.**
- Images: VOC2007 test in file order (the first `--n`), `stretch` preprocessing at 416.
- Networks: Tiny-YOLOv2 VOC, and Tiny-YOLOv3 and YOLOv4-tiny COCO, whose VOC classes are
  mapped to COCO.
- Explained box: the float detection of best score at the operating point (conf 0.25,
  NMS 0.45) that matches a ground truth of its class at IoU ≥ 0.5.
- Float maps: computed on the BN-folded network in float32.
- Integer stage: the `QuantModel` of `build/quant/<net>/calib.json`.
- Golden stage: `golden_run` on `model/<net>/`.
- These are **reduced runs**: 20 images per network (T36.1 to T36.3), 50 (T36.4) and 500
  person images (T36.5). The full sizes are listed in
  [long-runs.md](../docs/tasks/long-runs.md).

**State** (2026-10-23): only T36.3 is run (below). The other sections are empty until
their command has run (`tools/explain.py <command>`, then `report`); the commands and their
durations are in [long-runs.md](../docs/tasks/long-runs.md).

## T36.1: gradient maps, sanity checks

**Completeness.** Integrated Gradients, 64 midpoint steps; the table gives the relative
error |Σ attributions − (s(x) − s(x′))| / |s(x) − s(x′)|.

**Sanity checks** (Adebayo 2018). The weights of the top convs are redrawn (He init), from
the top conv down to every conv. ρ is the Spearman correlation between the absolute map of
the original network and that of the randomized one. A method that keeps its map once the
weights are random (ρ ≥ 0.3) fails.

<!-- explain:sanity:begin -->
| net | method | images | IG completeness error | ρ top conv | ρ top 3 | ρ all convs | sanity (ρ < 0.3) | passes | s / map |
|---|---|---|---|---|---|---|---|---|---|
| v2 | saliency | 17 | – | 0.97 | 0.96 | 0.95 | **fail** | 2 | 2.2 |
| v2 | smoothgrad | 17 | – | 0.97 | 0.97 | 0.96 | **fail** | 50 | 60.2 |
| v2 | ig_black | 17 | 0.026 | 0.94 | 0.94 | 0.92 | **fail** | 130 | 284.7 |
| v2 | ig_blur | 17 | 0.007 | 0.95 | 0.95 | 0.94 | **fail** | 130 | 249.1 |
| v2 | gradcam | 17 | – | 0.06 | 0.24 | 0.20 | pass | 2 | 2.1 |
| v3 | saliency | 14 | – | 1.00 | 1.00 | 0.95 | **fail** | 2 | 1.5 |
| v3 | smoothgrad | 14 | – | 1.00 | 1.00 | 0.96 | **fail** | 50 | 41.3 |
| v3 | ig_black | 14 | 0.026 | 1.00 | 1.00 | 0.93 | **fail** | 130 | 187.5 |
| v3 | ig_blur | 14 | 0.016 | 1.00 | 1.00 | 0.95 | **fail** | 130 | 180.6 |
| v3 | gradcam | 14 | – | 1.00 | 1.00 | -0.03 | pass | 2 | 1.6 |
| v4 | saliency | 20 | – | 1.00 | 1.00 | 0.96 | **fail** | 2 | 2.3 |
| v4 | smoothgrad | 20 | – | 1.00 | 1.00 | 0.97 | **fail** | 50 | 61.7 |
| v4 | ig_black | 20 | 0.065 | 1.00 | 1.00 | 0.94 | **fail** | 130 | 277.1 |
| v4 | ig_blur | 20 | 0.020 | 1.00 | 1.00 | 0.96 | **fail** | 130 | 281.5 |
| v4 | gradcam | 20 | – | 0.96 | 0.95 | 0.11 | pass | 2 | 2.5 |
<!-- explain:sanity:end -->

**Results** (2026-10-29, `tools/explain.py sanity --jobs 4`, 8 h 37 on a shared CPU; the
first 20 VOC test images, 17 / 14 / 20 with a matched detection):
- **Only Grad-CAM passes the sanity check.** Once every conv is randomized its map no
  longer matches (ρ 0.20, −0.03, 0.11). Saliency, SmoothGrad and both IG variants keep
  ρ ≥ 0.92 with random weights: on these detectors they show the input edges, not what the
  network uses (Adebayo 2018 found the same on classifiers).
- **For v3 and v4, randomizing only the top conv or the top three leaves Grad-CAM
  unchanged** (ρ 1.00, 0.95-0.96). The last convs in file order are the head of the other
  scale, which the explained box does not go through; only the full cascade reaches the
  layers it uses.
- **IG completeness at 64 steps: 0.7 to 6.5 %** (blurred baseline 0.7-2.0 %, black
  baseline 2.6-6.5 %). The 1 % of the acceptance holds only for the blurred baseline on
  v2; more steps are needed on the real networks (the tiny net needed 128).
- **Cost**: Grad-CAM and saliency 1.5-2.5 s per map (2 passes), SmoothGrad 41-62 s
  (50), IG 180-285 s (130 passes).

## T36.2: perturbation maps at the three stages

**Methods.**
- D-RISE: 300 masks, grid 16, p = 0.5.
- Occlusion: grey 64 px patch, stride 32.
- Weight of a perturbed image: the best IoU · class score over the candidates before the
  NMS.

The integer and golden stages share the int8 heads (M28), so their maps must be equal
**bit for bit**; this is checked on the first images with 100 masks.

<!-- explain:stages:begin -->
| net | method | images | Spearman float / int | passes / map | s / map float | s / map int |
|---|---|---|---|---|---|---|
| v2 | drise | 17 | 0.99 | 300 | 198 | 429 |
| v2 | occlusion | 17 | 0.73 | 145 | 100 | 199 |
| v3 | drise | 14 | 0.99 | 300 | 199 | 398 |
| v3 | occlusion | 14 | 0.63 | 145 | 97 | 179 |
| v4 | drise | 20 | 0.98 | 300 | 193 | 441 |
| v4 | occlusion | 20 | 0.74 | 145 | 95 | 208 |

Integer against golden, the same masks:

| net | images | masks | weights equal | maps equal (bit for bit) | s / map golden (1 process) |
|---|---|---|---|---|---|
| v2 | 3 | 100 | 3 / 3 | 3 / 3 | 601 |
| v3 | 2 | 100 | 2 / 2 | 2 / 2 | 485 |
| v4 | 5 | 100 | 5 / 5 | 5 / 5 | 730 |
<!-- explain:stages:end -->

**Results** (2026-10-29, `tools/explain.py stages --jobs 5`, 3 h 11 on a shared CPU):
- **D-RISE is stable across the float to integer step** (Spearman 0.98-0.99): the int8
  model is looking at the same regions as the float one.
- **Occlusion much less** (0.63-0.74). A 64 px patch moves the score by small steps that
  the integer requantization rounds differently, and the map has only 145 cells.
- **Integer == golden bit for bit** on every map checked (3, 2 and 5 images, 100 masks):
  an explanation computed in Python stands for the circuit as is.
- **Cost**: the integer stage takes 2× the float one per map; the golden 2.4-3.8 min per
  100-mask map on one process. A D-RISE map costs 300 inferences, which is fine on the PC
  and too much for the board.
- The deletion and insertion AUC per method come from the T36.4 run.

## T36.3: exact integer attribution

**Exact split.** On 16 random neurons per conv and image, the int64 sum of the products
qW·qx is compared with the accumulator minus the bias.

**Relevance.** LRP over the integer weights: ε rule with ε = 0.25 · std(z), and z⁺.
Requantization and leaky pass the relevance through unchanged, so the relevance kept per
layer measures what that approximation and the bias lose.

<!-- explain:intattr:begin -->
| net | images | Σ contributions ≠ acc − qb | relevance at input (ε) | (z⁺) | ρ int LRP / float LRP | ρ int LRP / Grad-CAM | ρ z⁺ / Grad-CAM | s / map |
|---|---|---|---|---|---|---|---|---|
| v2 | 17 | 0 / 2448 | 0.03 | 1.00 | 0.62 | 0.05 | 0.07 | 18.7 |
| v3 | 14 | 0 / 2912 | 0.02 | 1.00 | 0.57 | 0.04 | 0.14 | 10.3 |
| v4 | 20 | 0 / 6720 | 0.00 | 1.00 | 0.50 | 0.02 | 0.24 | 6.7 |

Relevance kept per conv (Σ R_in / Σ R_out, mean over the images):

| net | rule | per layer |
|---|---|---|
| v2 | eps | L0:0.21 L2:0.53 L4:0.41 L6:1.61 L8:0.53 L10:0.71 L12:0.88 L13:0.94 L14:0.80 |
| v2 | zplus | L0:1.00 L2:1.00 L4:1.00 L6:1.00 L8:1.00 L10:1.00 L12:1.00 L13:1.00 L14:1.00 |
| v3 | eps | L0:0.06 L2:0.68 L4:0.71 L6:0.47 L8:0.63 L10:1.04 L12:0.67 L13:0.76 L14:0.63 L15:0.88 |
| v3 | zplus | L0:1.00 L2:1.00 L4:1.00 L6:1.00 L8:1.00 L10:1.00 L12:1.00 L13:1.00 L14:1.00 L15:1.00 |
| v4 | eps | L0:1.83 L1:-0.13 L2:0.64 L4:8.23 L5:-0.43 L7:4.36 L10:6.82 L12:3.15 L13:-0.13 L15:0.58 L18:0.89 L20:0.40 L21:0.94 L23:0.85 L26:0.89 L27:1.07 L28:0.64 L29:1.12 L32:-0.08 L35:1.55 L36:1.06 |
| v4 | zplus | L0:1.00 L1:1.00 L2:1.00 L4:1.00 L5:1.00 L7:1.00 L10:1.00 L12:1.00 L13:1.00 L15:1.00 L18:1.00 L20:1.00 L21:1.00 L23:1.00 L26:1.00 L27:1.00 L28:1.00 L29:1.00 L32:1.00 L35:1.00 L36:1.00 |
<!-- explain:intattr:end -->

**Reading.**
- **Exact split.** The split of the accumulators is exact: no mismatch over 12,080
  neurons.
- **z⁺ rule.** It keeps all the relevance in every conv.
- **ε rule.** It loses almost all of it on the way to the input: 0 to 3 % arrives.
  - The bias takes its share at every layer.
  - The ratio leaves [0, 1] on YOLOv4-tiny, where the route layers split the channels.
  - For a map of the circuit, z⁺ is therefore the rule to use. The ε split stays exact
    in its linear part only.
- **Integer against float LRP.** Spearman 0.50 to 0.62: the two networks place the
  relevance alike.
- **LRP against Grad-CAM.** The input LRP maps agree little with Grad-CAM (ρ ≤ 0.24). They
  are pixel-level maps, Grad-CAM is a 13 × 13 map upsampled.

## T36.4: evaluation of the maps

Measures, for each method and stage:
- the pointing game (tolerance 15 px);
- the share of the positive map inside the box;
- deletion AUC (grey fill) and insertion AUC (into a blurred image), 16 steps, scored by
  the D-RISE weight of the explained box at the same stage.

Baselines: a random map and a centre Gaussian.

<!-- explain:evaluate:begin -->
<!-- explain:evaluate:end -->

## T36.5: causes of the missed persons

Persons of VOC2007 test (not difficult), at the operating point conf 0.25. Each miss gets
one cause, by the fixed order of tests of `yolo.explain.misses`.

One Grad-CAM map per cause, 5 examples, Tiny-YOLOv2 float:
`results/explain/misses_<cause>.png`.

**ExDark** (2026-10-28, rows `@exdark`): the first 500 test images with a `People` box
(1,324 persons, ground truth through `MAPPINGS`). In low light the miss rate doubles or
more (v2 39 → 81 %, v3 39 → 74 %, v4 18 → 54 %). The miss still comes mostly from low
objectness (77-86 %), but **no anchor** rises from 1-3 % to 9-12 %: dark scenes hold
persons whose box shape fits no anchor of the network (far or partial). Quantization adds
little (1-4 %).

**CrowdHuman val** (2026-10-29, rows `@crowdhuman`): the first 500 val images, 11,131
persons (full boxes, not `difficult`), 34 min with 6 jobs. The VOC and COCO networks miss
**61 % (v4) to 77 % (v2)** of the persons. The causes differ from VOC:
- **no anchor** goes up to **26-30 %** (1-3 % on VOC, 9-12 % on ExDark): crowds hold many
  small, partly hidden persons whose box shape is close to no anchor;
- low objectness stays first (53-68 %);
- NMS removes only 2-3 %: in crowds the persons are lost before the NMS, not by it;
- quantization adds 2 %, as on ExDark.
The cause points at the anchors (k-means on CrowdHuman, T15.14) rather than at the NMS.

The degraded sets of T27.3 wait for T27.3.

<!-- explain:misses:begin -->
| net | stage | persons | missed | quantization | no_anchor | low_objectness | class_confusion | below_threshold | nms_suppressed | mislocalized |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 | float | 1085 | 424 (39%) | 0% | 3% | 72% | 4% | 3% | 7% | 12% |
| v2 | int | 1085 | 427 (39%) | 7% | 3% | 69% | 3% | 1% | 6% | 11% |
| v3 | float | 1085 | 419 (39%) | 0% | 1% | 89% | 2% | 4% | 3% | 1% |
| v3 | int | 1085 | 416 (38%) | 5% | 1% | 86% | 2% | 3% | 2% | 1% |
| v4 | float | 1085 | 191 (18%) | 0% | 2% | 79% | 4% | 4% | 8% | 3% |
| v4 | int | 1085 | 190 (18%) | 11% | 2% | 72% | 4% | 4% | 7% | 2% |
| v2@exdark | float | 1324 | 1074 (81%) | 0% | 10% | 83% | 2% | 1% | 1% | 3% |
| v2@exdark | int | 1324 | 1068 (81%) | 1% | 10% | 83% | 2% | 1% | 1% | 2% |
| v3@exdark | float | 1324 | 977 (74%) | 0% | 9% | 86% | 2% | 1% | 1% | 1% |
| v3@exdark | int | 1324 | 976 (74%) | 2% | 9% | 85% | 1% | 1% | 1% | 1% |
| v4@exdark | float | 1324 | 714 (54%) | 0% | 12% | 80% | 2% | 3% | 1% | 2% |
| v4@exdark | int | 1324 | 708 (53%) | 4% | 12% | 77% | 2% | 1% | 2% | 2% |
| v2@crowdhuman | float | 11131 | 8554 (77%) | 0% | 27% | 57% | 0% | 1% | 3% | 11% |
| v2@crowdhuman | int | 11131 | 8453 (76%) | 2% | 28% | 53% | 1% | 1% | 3% | 13% |
| v3@crowdhuman | float | 11131 | 8016 (72%) | 0% | 26% | 68% | 0% | 1% | 2% | 3% |
| v3@crowdhuman | int | 11131 | 7972 (72%) | 2% | 26% | 67% | 0% | 0% | 2% | 3% |
| v4@crowdhuman | float | 11131 | 6832 (61%) | 0% | 30% | 62% | 1% | 1% | 3% | 4% |
| v4@crowdhuman | int | 11131 | 6766 (61%) | 2% | 30% | 60% | 1% | 1% | 2% | 4% |
<!-- explain:misses:end -->

## T36.6: concepts and channels

Network Dissection-lite and TCAV, on the layer read by the head.

<!-- explain:concepts:begin -->
tiny-yolov2-voc, L13, 200 VOC test images. Share of the channels matching no class (best IoU < 0.04): **39%**.

| class | channel 1 (IoU) | channel 2 | channel 3 |
|---|---|---|---|
| aeroplane | 310 (0.167) | 806 (0.118) | 584 (0.117) |
| bicycle | 559 (0.180) | 732 (0.170) | 371 (0.165) |
| bird | 425 (0.114) | 137 (0.096) | 373 (0.091) |
| boat | 624 (0.119) | 714 (0.112) | 105 (0.112) |
| bottle | 217 (0.116) | 579 (0.111) | 91 (0.108) |
| bus | 801 (0.116) | 947 (0.107) | 115 (0.105) |
| car | 63 (0.085) | 163 (0.084) | 735 (0.078) |
| cat | 706 (0.075) | 350 (0.073) | 701 (0.069) |
| chair | 922 (0.067) | 807 (0.067) | 556 (0.058) |
| cow | 118 (0.168) | 616 (0.145) | 209 (0.122) |
| diningtable | 715 (0.145) | 390 (0.098) | 240 (0.095) |
| dog | 834 (0.083) | 363 (0.082) | 649 (0.074) |
| horse | 437 (0.121) | 479 (0.114) | 329 (0.098) |
| motorbike | 389 (0.087) | 553 (0.084) | 416 (0.083) |
| person | 696 (0.029) | 788 (0.028) | 27 (0.026) |
| pottedplant | 630 (0.219) | 355 (0.153) | 419 (0.138) |
| sheep | 657 (0.137) | 222 (0.124) | 728 (0.113) |
| sofa | 723 (0.067) | 930 (0.061) | 917 (0.057) |
| train | 183 (0.150) | 228 (0.135) | 821 (0.124) |
| tvmonitor | 589 (0.155) | 675 (0.139) | 102 (0.127) |

TCAV for the class person (20 random sets of 30 images):

| concept | images | TCAV score | random-random mean | t | p | significant (p < 0.05) |
|---|---|---|---|---|---|---|
| dark | 30 | 0.95 | 0.46 | 6.09 | 3.41e-06 | yes |
| fog | 30 | 0.96 | 0.46 | 6.49 | 2.72e-06 | yes |
| small | 30 | 0.67 | 0.46 | 2.00 | 0.0525 | no |
| occluded | 30 | 0.94 | 0.46 | 5.95 | 4.72e-06 | yes |
<!-- explain:concepts:end -->

## T36.7: training recorder

**Run.** Tiny-YOLOv2 VOC fine-tuned from its Darknet weights on the first 200 images of
VOC trainval:
- batch 8, 250 iterations (10 epochs), lr 1e-4;
- per-layer records every 10 iterations, weights every 50;
- the same run with 5 % of the images relabelled at random (`--corrupt 0.05`).

**Learned.** An image is learned in an epoch when its loss is under the median loss of the
last epoch.

**Results** (2026-10-28, 2 h 45 for both runs side by side on 4 cores):
- the recorder costs **3 %** of the training time (≤ 10 %: met);
- 71 of the 200 images are never learned and 90 are forgotten at least once in the clean
  run: with 10 epochs on 200 images, the median-loss rule still sees many images as hard;
- the 10 relabelled images are found by their **mean loss** (AUROC 0.78), much less by
  "never learned" (AUROC 0.58, precision 7 %). On a short fine-tuning the mean loss is the
  signal to use; "never learned" is too coarse a flag.

<!-- explain:record:begin -->
| run | images | epochs | recorder overhead | never learned | forgotten | corrupted | AUROC never learned | AUROC mean loss |
|---|---|---|---|---|---|---|---|---|
| clean | 200 | 10 | 3.0% | 71 | 90 | – | – | – |
| corrupt5 | 200 | 10 | 2.8% | 68 | 93 | 10 | 0.58 | 0.78 |
<!-- explain:record:end -->

## T36.8: representation dynamics (CKA)

Linear CKA on the outputs of every conv, pooled over 2 × 2 cells, over 64 VOC test images:
- float against integer (dequantized), per layer;
- each checkpoint of the T36.7 clean run against its last one.

**Results** (2026-10-29, `tools/explain.py cka --run build/explain/record/clean`, 2 jobs;
4 jobs overflowed a 12 GB cap):
- **Float against integer**: CKA ≥ 0.99 on every conv of Tiny-YOLOv2. Tiny-YOLOv3 reaches
  0.9900 at L15 and YOLOv4-tiny drops to 0.9887 at L29, the head convs of their
  second scale: the deepest maps lose the most to int8 (not yet cross-checked with the
  per-layer LSB of T28.0).
- **Across the checkpoints**: every conv is at CKA ≥ 0.98 to the last checkpoint from
  iteration 50. A 250-iteration fine-tuning from the Darknet weights barely moves the
  representations; the last conv (L14, the head) and the middle convs 1-3 move most, at
  iteration 100. A run from scratch would be needed to see which layers settle first.
- Figures (T13.60): `build/figures/resultats/explain_cka_float_int.png`,
  `explain_cka_checkpoints.png`, `explain_gradcam_strip.png`.

<!-- explain:cka:begin -->
Probe: 64 VOC test images, maps pooled over 2 × 2 cells.

| net | convs | min CKA float / int (layer) | layers < 0.99 |
|---|---|---|---|
| v2 | 9 | 0.9977 (L14) | none |
| v3 | 13 | 0.9900 (L15) | L15 |
| v4 | 21 | 0.9887 (L29) | L29 |

CKA of each conv to the last checkpoint, run `build/explain/record/clean`:

| iteration | conv 0 | conv 1 | conv 2 | conv 3 | conv 4 | conv 5 | conv 6 | conv 7 | conv 8 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 |
| 100 | 0.99 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 |
| 150 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| 200 | 1.00 | 1.00 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| 250 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
<!-- explain:cka:end -->
