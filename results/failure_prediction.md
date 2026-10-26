# Learned failure prediction (M39)

A monitor trained to flag the images where the detector misses a person
([docs/tasks/M39-failure-prediction.md](../docs/tasks/M39-failure-prediction.md)). Tools:
`tools/failure_labels.py` (labels and features), `tools/failure_prediction.py` (head, stages,
calibration), `tools/ood_monitor.py compare --labels` (against the monitors of T29.0),
`tools/kgfp_baseline.py`.

## Summary (reduced runs, 2026-10-26)

**Status of these numbers**: reduced runs on the data present on the PC, `tiny-yolov3-coco`
(person class) on 500 COCO val2017 person images and the 592 PREWID test frames. The
threshold is the **provisional** 0.25 (T27.2 not done), and the test splits are 100 and 119
images. CrowdHuman, `tiny-yolov3-crowdhuman` and the joint training (T39.2) are open
([long-runs.md](../docs/tasks/long-runs.md)).

- **Labels (T39.0)**:
  - At 0.25 the detector misses 61 % of the COCO persons (70 % of the images have a miss)
    and 56 % of the PREWID persons (30 % of the frames; 289 frames are empty).
  - The imbalance runs the other way on COCO (0.4 failure-free images per failing one) and
    is 2.3 on PREWID.
  - Low objectness is the main cause of a miss (60 % on COCO, 94 % on PREWID).
  - On COCO, 34 % of the misses are `no_anchor`: small persons whose best anchor is anchor 0,
    which no Tiny-YOLOv3 head uses. This case made `misses.responsible_index` crash; it now
    counts as `no_anchor`.
  - Quantization explains 3-4 % of the integer misses.
- **Learned head against post-hoc monitors (T39.1)**:
  - On COCO, the head reaches AUROC 0.82 (float) / 0.81 (integer), against 0.77 for the max
    score. The other T29.0 monitors stay at 0.38-0.54: the OOD scores are blind to in-domain
    misses.
  - On PREWID, the head's 0.94 comes from the empty frames. Among the frames with persons, the
    max score (0.94) beats the head (0.87).
  - At 5 % alarm rate, 9-12 % of the missed persons are flagged. When 30-70 % of the images
    fail, a 5 % alarm rate cannot flag much more.
- **KGFP (T39.3)**:
  - Not run: its code needs a GPU and YOLOv8 features.
  - With its metric on COCO test, the person recall among the accepted images is 54 % for the
    max score and 50 % for the heads, against 43 % for the detector alone. The paper reports
    84.5 % for YOLOv8, a different detector, so the two figures do not compare.
  - DINO ViT-B/16 costs 85.8 M parameters and 17.6 GMACs: 1,743× the parameters of the head
    and 357,000× its MACs.
- **Calibration (T39.4)**:
  - Temperature scaling (T ≈ 1.1-1.2) barely moves the ECE: it is slightly better on COCO and
    worse on PREWID.
  - It is monotone, so the failure AUROC and AURC cannot change. Whether label smoothing
    degrades failure prediction (Zhu 2024) waits for its fine-tune.
- **Three stages (T39.5)**:
  - The integer head (zero point and standardization folded, int8 weights) takes 51 KB,
    against 197 KB in float32.
  - AUROC float → integer: 0.817 → 0.821 on COCO, 0.938 → 0.901 on PREWID. The flag changes
    on 1 and 3 test images.
  - Integer == C-sim (`hls/tb/tb_failure.cpp`) on all 1,092 images.

<!-- m39:labels -->
## Failure labels (T39.0)

Threshold 0.25 (NMS 0.45), **provisional**: the operating point of T27.2 is not chosen yet; a new threshold means new labels. A person is missed when no
detection of the person class kept by the NMS has IoU ≥ 0.5 with it (`misses.matched`, as T36.5); an image fails when one of its persons is missed.
Split fit / val / test 60 / 20 / 20, seed 0.

| Labels | Stage | Split | Images | Persons | Missed persons | Images with a miss | Negatives per positive |
|---|---|---|---|---|---|---|---|
| tiny-yolov3-coco_coco-val2017 | float | fit | 300 | 1219 | 63.8 % | 71.0 % | 0.4 |
| tiny-yolov3-coco_coco-val2017 | float | val | 100 | 403 | 57.1 % | 73.0 % | 0.4 |
| tiny-yolov3-coco_coco-val2017 | float | test | 100 | 345 | 56.8 % | 62.0 % | 0.6 |
| tiny-yolov3-coco_coco-val2017 | float | all | 500 | 1967 | 61.2 % | 69.6 % | 0.4 |
| tiny-yolov3-coco_coco-val2017 | int | fit | 300 | 1219 | 63.7 % | 70.0 % | 0.4 |
| tiny-yolov3-coco_coco-val2017 | int | val | 100 | 403 | 58.1 % | 76.0 % | 0.3 |
| tiny-yolov3-coco_coco-val2017 | int | test | 100 | 345 | 57.1 % | 61.0 % | 0.6 |
| tiny-yolov3-coco_coco-val2017 | int | all | 500 | 1967 | 61.4 % | 69.4 % | 0.4 |
| tiny-yolov3-coco_prewid-test | float | fit | 355 | 206 | 59.2 % | 32.4 % | 2.1 |
| tiny-yolov3-coco_prewid-test | float | val | 118 | 66 | 47.0 % | 23.7 % | 3.2 |
| tiny-yolov3-coco_prewid-test | float | test | 119 | 71 | 53.5 % | 31.1 % | 2.2 |
| tiny-yolov3-coco_prewid-test | float | all | 592 | 343 | 55.7 % | 30.4 % | 2.3 |
| tiny-yolov3-coco_prewid-test | int | fit | 355 | 206 | 61.2 % | 33.2 % | 2.0 |
| tiny-yolov3-coco_prewid-test | int | val | 118 | 66 | 48.5 % | 24.6 % | 3.1 |
| tiny-yolov3-coco_prewid-test | int | test | 119 | 71 | 52.1 % | 30.3 % | 2.3 |
| tiny-yolov3-coco_prewid-test | int | all | 592 | 343 | 56.9 % | 30.9 % | 2.2 |

The last column is the class imbalance T39.1 trains against (failure-free images per failing
image): the positive weight of `imbalance="weight"`.

Causes of the missed persons (T36.5, fixed order of tests):

| Labels | Stage | Missed | quantization | no_anchor | low_objectness | class_confusion | below_threshold | nms_suppressed | mislocalized |
|---|---|---|---|---|---|---|---|---|---|
| tiny-yolov3-coco_coco-val2017 | float | 1204 | 0.0 % | 34.2 % | 60.4 % | 0.8 % | 1.7 % | 1.3 % | 1.6 % |
| tiny-yolov3-coco_coco-val2017 | int | 1208 | 3.0 % | 34.1 % | 58.9 % | 0.7 % | 0.6 % | 1.2 % | 1.5 % |
| tiny-yolov3-coco_prewid-test | float | 191 | 0.0 % | 2.1 % | 93.7 % | 0.0 % | 1.0 % | 0.5 % | 2.6 % |
| tiny-yolov3-coco_prewid-test | int | 195 | 3.6 % | 2.1 % | 90.8 % | 1.0 % | 0.0 % | 0.0 % | 2.6 % |
<!-- /m39:labels -->

<!-- m39:head -->
## Post-hoc learned monitor (T39.1)

Heads: MLP d → 32 (ReLU) → 1 on the pooled features of the two head inputs (mean
and max per channel, d = 1,536 for Tiny-YOLOv3), trained on the fit split; imbalance
strategy chosen on val; numbers on test. `max_score`: 1 − the top person score, the best
post-hoc monitor of T29.0 on ExDark. Alarm rates are over all test images; "system recall":
persons detected or in an image that raises an alarm. The KGFP metric: threshold rejecting 5 %
of the failure-free images, person recall among the accepted images (against the detector
recall over all images), and the share of images accepted.

**tiny-yolov3-coco_coco-val2017** (test split, 100 images, threshold 0.25):

| Monitor | Stage | AUROC | AUROC, images with persons | Missed flagged @1 % | @5 % | System recall @1 % | @5 % | Detector recall | Person recall accepted @5 % FPR | Accepted | Memory | MACs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_score | float | 0.768 | 0.768 | 0.5 % | 3.6 % | 43.5 % | 45.2 % | 43.2 % | 53.8 % | 69.0 % | 0 B | 0 |
| head_miss | float | 0.817 | 0.817 | 0.5 % | 8.7 % | 43.5 % | 48.1 % | 43.2 % | 50.2 % | 77.0 % | 196,868 B | 49,184 |
| head_tcp | float | 0.804 | 0.804 | 0.0 % | 11.7 % | 43.2 % | 49.9 % | 43.2 % | 50.5 % | 79.0 % | 196,868 B | 49,184 |
| max_score | int | 0.770 | 0.770 | 0.5 % | 5.6 % | 43.2 % | 46.1 % | 42.9 % | 53.8 % | 72.0 % | 0 B | 0 |
| head_miss | int | 0.805 | 0.805 | 0.5 % | 11.7 % | 43.2 % | 49.6 % | 42.9 % | 54.5 % | 71.0 % | 196,868 B | 49,184 |
| head_tcp | int | 0.810 | 0.810 | 0.0 % | 11.7 % | 42.9 % | 49.6 % | 42.9 % | 49.0 % | 78.0 % | 196,868 B | 49,184 |

**tiny-yolov3-coco_prewid-test** (test split, 119 images, threshold 0.25):

| Monitor | Stage | AUROC | AUROC, images with persons | Missed flagged @1 % | @5 % | System recall @1 % | @5 % | Detector recall | Person recall accepted @5 % FPR | Accepted | Memory | MACs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max_score | float | 0.427 | 0.936 | 0.0 % | 0.0 % | 46.5 % | 46.5 % | 46.5 % | 46.5 % | 96.6 % | 0 B | 0 |
| head_miss | float | 0.938 | 0.869 | 0.0 % | 10.5 % | 46.5 % | 52.1 % | 46.5 % | 76.3 % | 72.3 % | 196,868 B | 49,184 |
| head_tcp | float | 0.937 | 0.848 | 2.6 % | 10.5 % | 47.9 % | 52.1 % | 46.5 % | 76.3 % | 73.1 % | 196,868 B | 49,184 |
| max_score | int | 0.442 | 0.930 | 0.0 % | 2.7 % | 47.9 % | 49.3 % | 47.9 % | 48.6 % | 95.8 % | 0 B | 0 |
| head_miss | int | 0.930 | 0.856 | 2.7 % | 10.8 % | 49.3 % | 53.5 % | 47.9 % | 66.7 % | 78.2 % | 196,868 B | 49,184 |
| head_tcp | int | 0.916 | 0.821 | 2.7 % | 10.8 % | 49.3 % | 53.5 % | 47.9 % | 53.6 % | 87.4 % | 196,868 B | 49,184 |
<!-- /m39:head -->

<!-- m39:monitors-tiny-yolov3-coco_coco-val2017-float -->
## T29.0 monitors and learned heads against missed persons: tiny-yolov3-coco_coco-val2017, float stage (T39.1)

`tools/ood_monitor.py compare --labels … --failure …`, test split. The T29.0
monitors are fitted on the fit split (pooled features of the labelled network, integer for
the boxes and patterns); raw score or typicality, whichever has the better val AUROC. The
autoencoder is left out (it needs the full maps; T29.0 found it dominated). Alarm rates are
over all test images.

| Monitor | Score | AUROC | AUROC, images with persons | Missed flagged @1 % | @5 % | System recall @5 % | Detector recall | Memory | Ops per image |
|---|---|---|---|---|---|---|---|---|---|
| norm | raw | 0.386 | 0.386 | 0.0 % | 3.1 % | 44.9 % | 43.2 % | 0 B | 1,536 |
| mahalanobis | raw | 0.383 | 0.383 | 0.0 % | 0.5 % | 43.5 % | 43.2 % | 9,443,328 B | 2,360,832 |
| max_score | typ | 0.659 | 0.659 | 0.5 % | 3.6 % | 45.2 % | 43.2 % | 0 B | 1 |
| boxes | typ | 0.506 | 0.506 | 0.0 % | 7.7 % | 47.5 % | 43.2 % | 49,152 B | 12,288 |
| patterns | typ | 0.472 | 0.472 | 0.0 % | 0.5 % | 43.5 % | 43.2 % | 2,912 B | 1,264 |
| failure (miss, float) | learned | 0.817 | 0.817 | 0.5 % | 8.7 % | 48.1 % | 43.2 % | 196,868 B | 49,184 |
| failure (tcp, float) | learned | 0.804 | 0.804 | 0.0 % | 11.7 % | 49.9 % | 43.2 % | 196,868 B | 49,184 |
<!-- /m39:monitors-tiny-yolov3-coco_coco-val2017-float -->

<!-- m39:monitors-tiny-yolov3-coco_coco-val2017-int -->
## T29.0 monitors and learned heads against missed persons: tiny-yolov3-coco_coco-val2017, int stage (T39.1)

`tools/ood_monitor.py compare --labels … --failure …`, test split. The T29.0
monitors are fitted on the fit split (pooled features of the labelled network, integer for
the boxes and patterns); raw score or typicality, whichever has the better val AUROC. The
autoencoder is left out (it needs the full maps; T29.0 found it dominated). Alarm rates are
over all test images.

| Monitor | Score | AUROC | AUROC, images with persons | Missed flagged @1 % | @5 % | System recall @5 % | Detector recall | Memory | Ops per image |
|---|---|---|---|---|---|---|---|---|---|
| norm | raw | 0.384 | 0.384 | 0.0 % | 7.1 % | 47.0 % | 42.9 % | 0 B | 1,536 |
| mahalanobis | raw | 0.423 | 0.423 | 0.0 % | 4.1 % | 45.2 % | 42.9 % | 9,443,328 B | 2,360,832 |
| max_score | typ | 0.667 | 0.667 | 0.5 % | 5.6 % | 46.1 % | 42.9 % | 0 B | 1 |
| boxes | typ | 0.509 | 0.509 | 0.0 % | 7.6 % | 47.2 % | 42.9 % | 49,152 B | 12,288 |
| patterns | raw | 0.539 | 0.539 | 0.0 % | 0.5 % | 43.2 % | 42.9 % | 2,912 B | 1,264 |
| failure (miss, int) | learned | 0.805 | 0.805 | 0.5 % | 11.7 % | 49.6 % | 42.9 % | 196,868 B | 49,184 |
| failure (tcp, int) | learned | 0.810 | 0.810 | 0.0 % | 11.7 % | 49.6 % | 42.9 % | 196,868 B | 49,184 |
<!-- /m39:monitors-tiny-yolov3-coco_coco-val2017-int -->

<!-- m39:monitors-tiny-yolov3-coco_prewid-test-float -->
## T29.0 monitors and learned heads against missed persons: tiny-yolov3-coco_prewid-test, float stage (T39.1)

`tools/ood_monitor.py compare --labels … --failure …`, test split. The T29.0
monitors are fitted on the fit split (pooled features of the labelled network, integer for
the boxes and patterns); raw score or typicality, whichever has the better val AUROC. The
autoencoder is left out (it needs the full maps; T29.0 found it dominated). Alarm rates are
over all test images.

| Monitor | Score | AUROC | AUROC, images with persons | Missed flagged @1 % | @5 % | System recall @5 % | Detector recall | Memory | Ops per image |
|---|---|---|---|---|---|---|---|---|---|
| norm | raw | 0.472 | 0.314 | 0.0 % | 0.0 % | 46.5 % | 46.5 % | 0 B | 1,536 |
| mahalanobis | raw | 0.619 | 0.408 | 2.6 % | 10.5 % | 52.1 % | 46.5 % | 9,443,328 B | 2,360,832 |
| max_score | raw | 0.427 | 0.936 | 0.0 % | 0.0 % | 46.5 % | 46.5 % | 0 B | 1 |
| boxes | raw | 0.551 | 0.438 | 0.0 % | 2.6 % | 47.9 % | 46.5 % | 49,152 B | 12,288 |
| patterns | raw | 0.563 | 0.432 | 0.0 % | 2.6 % | 47.9 % | 46.5 % | 3,144 B | 1,380 |
| failure (miss, float) | learned | 0.938 | 0.869 | 0.0 % | 10.5 % | 52.1 % | 46.5 % | 196,868 B | 49,184 |
| failure (tcp, float) | learned | 0.937 | 0.848 | 2.6 % | 10.5 % | 52.1 % | 46.5 % | 196,868 B | 49,184 |
<!-- /m39:monitors-tiny-yolov3-coco_prewid-test-float -->

<!-- m39:monitors-tiny-yolov3-coco_prewid-test-int -->
## T29.0 monitors and learned heads against missed persons: tiny-yolov3-coco_prewid-test, int stage (T39.1)

`tools/ood_monitor.py compare --labels … --failure …`, test split. The T29.0
monitors are fitted on the fit split (pooled features of the labelled network, integer for
the boxes and patterns); raw score or typicality, whichever has the better val AUROC. The
autoencoder is left out (it needs the full maps; T29.0 found it dominated). Alarm rates are
over all test images.

| Monitor | Score | AUROC | AUROC, images with persons | Missed flagged @1 % | @5 % | System recall @5 % | Detector recall | Memory | Ops per image |
|---|---|---|---|---|---|---|---|---|---|
| norm | raw | 0.453 | 0.275 | 0.0 % | 0.0 % | 47.9 % | 47.9 % | 0 B | 1,536 |
| mahalanobis | raw | 0.610 | 0.443 | 2.7 % | 10.8 % | 53.5 % | 47.9 % | 9,443,328 B | 2,360,832 |
| max_score | raw | 0.442 | 0.930 | 0.0 % | 2.7 % | 49.3 % | 47.9 % | 0 B | 1 |
| boxes | raw | 0.547 | 0.432 | 0.0 % | 2.7 % | 49.3 % | 47.9 % | 49,152 B | 12,288 |
| patterns | raw | 0.548 | 0.412 | 0.0 % | 2.7 % | 49.3 % | 47.9 % | 3,144 B | 1,380 |
| failure (miss, int) | learned | 0.930 | 0.856 | 2.7 % | 10.8 % | 53.5 % | 47.9 % | 196,868 B | 49,184 |
| failure (tcp, int) | learned | 0.916 | 0.821 | 2.7 % | 10.8 % | 53.5 % | 47.9 % | 196,868 B | 49,184 |
<!-- /m39:monitors-tiny-yolov3-coco_prewid-test-int -->

<!-- m39:calib-tiny-yolov3-coco_coco-val2017 -->
## Calibration against failure prediction (T39.4): tiny-yolov3-coco_coco-val2017

`tiny-yolov3-coco_coco-val2017`. Temperature fitted by NLL on the person detections (down to 0.005) of
the fit split, s' = σ(logit(s)/T); ECE (15 bins) on the test split; AUROC and AURC (lower is
better) of the failure prediction on the test images. Temperature scaling is monotone: the
max-score monitor keeps its ranking, so its AUROC and AURC cannot move, and the head does
not read the score. The question of Zhu 2024 is open until the label-smoothing fine-tune
(`tools/train.py --label-smoothing`) gives its labels (`--smoothed`).

| Detector | Stage | Calibration | T | ECE person score | AUROC max score | AURC max score | AUROC head | AURC head |
|---|---|---|---|---|---|---|---|---|
| base | float | none | 1.00 | 0.0450 | 0.768 | 0.466 | 0.817 | 0.385 |
| base | float | temperature | 1.23 | 0.0439 | 0.768 | 0.466 | 0.817 | 0.385 |
| base | int | none | 1.00 | 0.0415 | 0.770 | 0.454 | 0.805 | 0.380 |
| base | int | temperature | 1.20 | 0.0390 | 0.770 | 0.454 | 0.805 | 0.380 |
<!-- /m39:calib-tiny-yolov3-coco_coco-val2017 -->

<!-- m39:stages-tiny-yolov3-coco_coco-val2017 -->
## Failure monitor at the three stages (T39.5): tiny-yolov3-coco_coco-val2017

`tiny-yolov3-coco_coco-val2017`, miss head of the float stage. Integer head: zero point and standardization folded, int8 weights, int32 accumulators, hidden layer requantized as a conv (§9.3); 50,981 B against 196,868 B in float32, 49,184 MACs. Flag at 5 % alarm rate, threshold chosen on val per stage; AUROC and missed persons flagged on test (at the integer stage against the integer labels). Integer == C-sim on every image: **True**.

| Stage | AUROC | Missed flagged @5 % | Alarm rate on test |
|---|---|---|---|
| float | 0.817 | 8.7 % | 2.0 % |
| int | 0.821 | 6.1 % | 1.0 % |
| csim | 0.821 | 6.1 % | 1.0 % |

Test images whose flag changes between float and integer: 1.

| Image | Flag float → int | Fails float / int | Causes (int) |
|---|---|---|---|
| 423617 | 1 → 0 | 1 / 1 | low_objectness, no_anchor |
<!-- /m39:stages-tiny-yolov3-coco_coco-val2017 -->

<!-- m39:calib-tiny-yolov3-coco_prewid-test -->
## Calibration against failure prediction (T39.4): tiny-yolov3-coco_prewid-test

`tiny-yolov3-coco_prewid-test`. Temperature fitted by NLL on the person detections (down to 0.005) of
the fit split, s' = σ(logit(s)/T); ECE (15 bins) on the test split; AUROC and AURC (lower is
better) of the failure prediction on the test images. Temperature scaling is monotone: the
max-score monitor keeps its ranking, so its AUROC and AURC cannot move, and the head does
not read the score. The question of Zhu 2024 is open until the label-smoothing fine-tune
(`tools/train.py --label-smoothing`) gives its labels (`--smoothed`).

| Detector | Stage | Calibration | T | ECE person score | AUROC max score | AURC max score | AUROC head | AURC head |
|---|---|---|---|---|---|---|---|---|
| base | float | none | 1.00 | 0.0311 | 0.427 | 0.291 | 0.938 | 0.077 |
| base | float | temperature | 1.13 | 0.0369 | 0.427 | 0.291 | 0.938 | 0.077 |
| base | int | none | 1.00 | 0.0286 | 0.442 | 0.279 | 0.930 | 0.075 |
| base | int | temperature | 1.08 | 0.0308 | 0.442 | 0.279 | 0.930 | 0.075 |
<!-- /m39:calib-tiny-yolov3-coco_prewid-test -->

<!-- m39:stages-tiny-yolov3-coco_prewid-test -->
## Failure monitor at the three stages (T39.5): tiny-yolov3-coco_prewid-test

`tiny-yolov3-coco_prewid-test`, miss head of the float stage. Integer head: zero point and standardization folded, int8 weights, int32 accumulators, hidden layer requantized as a conv (§9.3); 50,981 B against 196,868 B in float32, 49,184 MACs. Flag at 5 % alarm rate, threshold chosen on val per stage; AUROC and missed persons flagged on test (at the integer stage against the integer labels). Integer == C-sim on every image: **True**.

| Stage | AUROC | Missed flagged @5 % | Alarm rate on test |
|---|---|---|---|
| float | 0.938 | 10.5 % | 5.0 % |
| int | 0.901 | 13.5 % | 2.5 % |
| csim | 0.901 | 13.5 % | 2.5 % |

Test images whose flag changes between float and integer: 3.

| Image | Flag float → int | Fails float / int | Causes (int) |
|---|---|---|---|
| 215201_CAM0 | 1 → 0 | 1 / 1 | low_objectness |
| 215351_CAM0 | 1 → 0 | 0 / 0 | – |
| 219891_CAM3 | 1 → 0 | 1 / 1 | low_objectness |
<!-- /m39:stages-tiny-yolov3-coco_prewid-test -->

<!-- m39:kgfp-cost -->
## KGFP baseline: cost (T39.3)

| Model | Parameters | MACs per image | Weights |
|---|---|---|---|
| DINO ViT-B/16 (KGFP's vision model) | 85.8 M | 17.56 G | 327 MiB (float32) |
| Failure head (T39.1, float32) | 49,217 | 49,184 | 192 KiB |
| Failure head (T39.5, integer) | same | same | ≈ 50 KiB (int8 weights) |

Ratio: 1,743× the parameters, 357,089× the MACs. KGFP also runs its own feature encoder on the detector features (not counted here); the failure head reuses maps the detector already computes.
<!-- /m39:kgfp-cost -->

<!-- m39:kgfp-tiny-yolov3-coco_coco-val2017 -->
## KGFP metric (T39.3): tiny-yolov3-coco_coco-val2017

`tiny-yolov3-coco_coco-val2017`, float stage, test split (100 images), detector person recall over all images 43.2 %. Threshold rejecting 5 % of the failure-free images.

| Monitor | Person recall among accepted images | Images accepted | Failing images rejected |
|---|---|---|---|
| max_score | 53.8 % | 69.0 % | 48.4 % |
| head_miss (T39.1) | 50.2 % | 77.0 % | 35.5 % |
| head_tcp (T39.1) | 50.5 % | 79.0 % | 32.3 % |

KGFP itself is not run here: its code (gitlab.cc-asp.fraunhofer.de/iosb_public/KGFP, arXiv 2603.25499) needs a GPU and YOLOv8 features. Paper figure, for YOLOv8 on COCO persons: 64.3 % → 84.5 %, a different detector and different failing images.
<!-- /m39:kgfp-tiny-yolov3-coco_coco-val2017 -->
