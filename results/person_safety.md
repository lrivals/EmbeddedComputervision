# Person detection as a safety function (M27)

The detector measured the way a safety function is judged: a missed person is the dreaded
event, a false alarm an unwanted stop
([docs/tasks/M27-safety-metrics.md](../docs/tasks/M27-safety-metrics.md)). Tools:
`tools/person_safety.py`, `tools/degrade.py`, `tools/dreaded_event.py`; metrics in
`python/yolo/infer/metrics.py`. Matching: VOC rule at IoU 0.5, `difficult` boxes and ignore
regions neither hit nor false alarm; detections down to 0.001.

Notation: **FPPI** false positives per image; **R@0.1** recall at the lowest threshold whose
FPPI ≤ 0.1; **MR** = 1 − recall; **LAMR** log-average miss rate over FPPI ∈ [10⁻², 10⁰]
(Caltech, Dollár 2012; lower is better).

<!-- m27:stages-prewid-test -->
## Three stages: prewid test (T27.1)

prewid `test`, 592 images. Recall and miss rate in %; per-bin recall at the R@0.1 threshold of each row.

| Net | Stage | Images | Persons | R@0.1 | R@0.01 | MR@0.1 | LAMR | thr@0.1 | R@0.1 (FPPI on empty) | empty frames |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny-yolov3-coco | float | 592 | 343 | 59.8 | 32.1 | 40.2 | 41.8 | 0.107 | 64.1 | 289 |
| tiny-yolov3-coco | integer | 592 | 343 | 56.6 | 31.2 | 43.4 | 45.1 | 0.121 | 60.9 | 289 |
| tiny-yolov4-coco | float | 592 | 343 | 90.7 | 82.2 | 9.3 | 9.6 | 0.0654 | 91.3 | 289 |
| tiny-yolov4-coco | integer | 592 | 343 | 90.1 | 75.5 | 9.9 | 10.5 | 0.0736 | 91.3 | 289 |
| tiny-yolov4-coco | C-sim (100 images) | 100 | 55 | 94.5 | 69.1 | 5.5 | 8.8 | 0.067 | – | – |
| tiny-yolov4-coco | integer (100 images) | 100 | 55 | 94.5 | 69.1 | 5.5 | 8.8 | 0.067 | – | – |

**Recall per box height** (pixels of the original image, Caltech bins), at 0.1 FPPI

| Net | Stage | < 30 px | 30-80 px | ≥ 80 px |
|---|---|---|---|---|
| tiny-yolov3-coco | float | 0.0 (0/3) | 34.4 (45/131) | 76.6 (160/209) |
| tiny-yolov3-coco | integer | 0.0 (0/3) | 29.0 (38/131) | 74.6 (156/209) |
| tiny-yolov4-coco | float | 0.0 (0/3) | 82.4 (108/131) | 97.1 (203/209) |
| tiny-yolov4-coco | integer | 0.0 (0/3) | 80.9 (106/131) | 97.1 (203/209) |

**Loss per stage** (difference of R@0.1 and LAMR, points)

| Net | Step | Images | ΔR@0.1 / ΔLAMR |
|---|---|---|---|
| tiny-yolov3-coco | quantization (integer − float) | all | -3.2 / +3.3 |
| tiny-yolov4-coco | quantization (integer − float) | all | -0.6 / +0.9 |
| tiny-yolov4-coco | engine (C-sim − integer) | 100 images | +0.0 / +0.0 |

**Per camera**

| Camera | Net | Stage | Images | Persons | R@0.1 | R@0.01 | MR@0.1 | LAMR | thr@0.1 | R@0.1 (FPPI on empty) | empty frames |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAM0 | tiny-yolov3-coco | float | 146 | 55 | 49.1 | 30.9 | 50.9 | 50.2 | 0.0772 | 67.3 | 94 |
| CAM0 | tiny-yolov3-coco | integer | 146 | 55 | 47.3 | 25.5 | 52.7 | 51.6 | 0.0741 | 65.5 | 94 |
| CAM0 | tiny-yolov4-coco | float | 146 | 55 | 87.3 | 78.2 | 12.7 | 7.2 | 0.048 | 98.2 | 94 |
| CAM0 | tiny-yolov4-coco | integer | 146 | 55 | 89.1 | 81.8 | 10.9 | 7.4 | 0.0565 | 98.2 | 94 |
| CAM1 | tiny-yolov3-coco | float | 147 | 105 | 62.9 | 51.4 | 37.1 | 33.5 | 0.153 | 74.3 | 45 |
| CAM1 | tiny-yolov3-coco | integer | 147 | 105 | 57.1 | 48.6 | 42.9 | 37.7 | 0.182 | 69.5 | 45 |
| CAM1 | tiny-yolov4-coco | float | 147 | 105 | 93.3 | 83.8 | 6.7 | 7.5 | 0.0633 | 93.3 | 45 |
| CAM1 | tiny-yolov4-coco | integer | 147 | 105 | 90.5 | 72.4 | 9.5 | 9.6 | 0.0776 | 92.4 | 45 |
| CAM2 | tiny-yolov3-coco | float | 150 | 109 | 60.6 | 21.1 | 39.4 | 46.7 | 0.0798 | 62.4 | 61 |
| CAM2 | tiny-yolov3-coco | integer | 150 | 109 | 58.7 | 22.9 | 41.3 | 49.2 | 0.102 | 60.6 | 61 |
| CAM2 | tiny-yolov4-coco | float | 150 | 109 | 89.0 | 80.7 | 11.0 | 11.7 | 0.0821 | 88.1 | 61 |
| CAM2 | tiny-yolov4-coco | integer | 150 | 109 | 89.9 | 78.9 | 10.1 | 12.1 | 0.0805 | 89.0 | 61 |
| CAM3 | tiny-yolov3-coco | float | 149 | 74 | 66.2 | 48.6 | 33.8 | 31.2 | 0.0591 | 83.8 | 89 |
| CAM3 | tiny-yolov3-coco | integer | 149 | 74 | 64.9 | 43.2 | 35.1 | 34.3 | 0.0437 | 78.4 | 89 |
| CAM3 | tiny-yolov4-coco | float | 149 | 74 | 93.2 | 85.1 | 6.8 | 6.7 | 0.0595 | 94.6 | 89 |
| CAM3 | tiny-yolov4-coco | integer | 149 | 74 | 93.2 | 79.7 | 6.8 | 8.4 | 0.0743 | 94.6 | 89 |
<!-- /m27:stages-prewid-test -->

<!-- m27:stages-prewid-lab -->
## Three stages: prewid lab (T27.1)

prewid `lab`, 3000 images. Recall and miss rate in %; per-bin recall at the R@0.1 threshold of each row.

| Net | Stage | Images | Persons | R@0.1 | R@0.01 | MR@0.1 | LAMR | thr@0.1 | R@0.1 (FPPI on empty) | empty frames |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny-yolov4-coco | float | 3000 | 1574 | 88.4 | 80.4 | 11.6 | 12.2 | 0.0946 | 89.6 | 1426 |
| tiny-yolov4-coco | integer | 3000 | 1574 | 90.6 | 78.8 | 9.4 | 10.6 | 0.0953 | 91.7 | 1426 |

**Recall per box height** (pixels of the original image, Caltech bins), at 0.1 FPPI

| Net | Stage | < 30 px | 30-80 px | ≥ 80 px |
|---|---|---|---|---|
| tiny-yolov4-coco | float | 0.0 (0/5) | 41.5 (27/65) | 90.8 (1365/1504) |
| tiny-yolov4-coco | integer | 0.0 (0/5) | 40.0 (26/65) | 93.1 (1400/1504) |

**Loss per stage** (difference of R@0.1 and LAMR, points)

| Net | Step | Images | ΔR@0.1 / ΔLAMR |
|---|---|---|---|
| tiny-yolov4-coco | quantization (integer − float) | all | +2.2 / -1.6 |
<!-- /m27:stages-prewid-lab -->

<!-- m27:oppoint-prewid-test-tiny-yolov4-coco -->
## Asymmetric operating point: tiny-yolov4-coco, prewid test (T27.2)

tiny-yolov4-coco on prewid `test`: threshold chosen on the calibration half (300 images), checked on the other half (292 images); halves made of alternating blocks of 25 consecutive frames per sequence, so neighbouring frames do not leak across. Recall in %.

| Stage | Target recall | Threshold | Calibration recall / FPPI | Check recall / FPPI | Target held |
|---|---|---|---|---|---|
| float | 95.0 | 0.04273 | 95.6 / 0.253 | 89.1 / 0.250 | **no** |
| float | 99.0 | never reached (max 98.7 at 0.001) | – | – | – |
| integer | 95.0 | 0.03287 | 95.6 / 0.553 | 90.8 / 0.565 | **no** |
| integer | 99.0 | never reached (max 96.9 at 0.001) | – | – | – |

**Repo thresholds on the curve** (all images)

| Stage | Threshold | Recall | FPPI |
|---|---|---|---|
| float | 0.005 | 95.9 | 3.684 |
| float | 0.5 | 64.7 | 0.000 |
| integer | 0.005 | 95.3 | 4.095 |
| integer | 0.5 | 65.9 | 0.000 |

Curves: `build/m27/prewid-test/tiny-yolov4-coco/oppoint.json`; figure `python -m tools.figures miss_fppi`.
<!-- /m27:oppoint-prewid-test-tiny-yolov4-coco -->

<!-- m27:oppoint-prewid-test-tiny-yolov3-coco -->
## Asymmetric operating point: tiny-yolov3-coco, prewid test (T27.2)

tiny-yolov3-coco on prewid `test`: threshold chosen on the calibration half (300 images), checked on the other half (292 images); halves made of alternating blocks of 25 consecutive frames per sequence, so neighbouring frames do not leak across. Recall in %.

| Stage | Target recall | Threshold | Calibration recall / FPPI | Check recall / FPPI | Target held |
|---|---|---|---|---|---|
| float | 95.0 | never reached (max 88.7 at 0.001) | – | – | – |
| float | 99.0 | never reached (max 88.7 at 0.001) | – | – | – |
| integer | 95.0 | never reached (max 84.9 at 0.001) | – | – | – |
| integer | 99.0 | never reached (max 84.9 at 0.001) | – | – | – |

**Repo thresholds on the curve** (all images)

| Stage | Threshold | Recall | FPPI |
|---|---|---|---|
| float | 0.005 | 82.2 | 6.333 |
| float | 0.5 | 26.8 | 0.005 |
| integer | 0.005 | 79.3 | 6.885 |
| integer | 0.5 | 26.2 | 0.005 |

Curves: `build/m27/prewid-test/tiny-yolov3-coco/oppoint.json`; figure `python -m tools.figures miss_fppi`.
<!-- /m27:oppoint-prewid-test-tiny-yolov3-coco -->

<!-- m27:oppoint-prewid-lab-tiny-yolov4-coco -->
## Asymmetric operating point: tiny-yolov4-coco, prewid lab (T27.2)

tiny-yolov4-coco on prewid `lab`: threshold chosen on the calibration half (1500 images), checked on the other half (1500 images); halves made of alternating blocks of 100 consecutive frames per sequence, so neighbouring frames do not leak across. Recall in %.

| Stage | Target recall | Threshold | Calibration recall / FPPI | Check recall / FPPI | Target held |
|---|---|---|---|---|---|
| float | 95.0 | 0.03815 | 95.1 / 0.745 | 85.7 / 0.855 | **no** |
| float | 99.0 | never reached (max 97.1 at 0.001) | – | – | – |
| integer | 95.0 | 0.08106 | 95.1 / 0.145 | 87.1 / 0.134 | **no** |
| integer | 99.0 | never reached (max 97.3 at 0.001) | – | – | – |

**Repo thresholds on the curve** (all images)

| Stage | Threshold | Recall | FPPI |
|---|---|---|---|
| float | 0.005 | 92.3 | 3.237 |
| float | 0.5 | 71.8 | 0.003 |
| integer | 0.005 | 94.0 | 2.838 |
| integer | 0.5 | 68.6 | 0.004 |

Curves: `build/m27/prewid-lab/tiny-yolov4-coco/oppoint.json`; figure `python -m tools.figures miss_fppi`.
<!-- /m27:oppoint-prewid-lab-tiny-yolov4-coco -->

<!-- m27:degrade-prewid-test-tiny-yolov4-coco -->
## Degraded conditions: tiny-yolov4-coco, prewid test (T27.3)

tiny-yolov4-coco on prewid `test` (592 images), seed 0. **R@0.1**: recall at 0.1 FPPI with the threshold re-chosen on the degraded images; **at clean thr**: recall and FPPI when the threshold stays at the clean 0.1-FPPI operating point (what a deployed system would see). Degradations of `tools/degrade.py` (severity 1-3).

| Condition | R@0.1 float | float at clean thr: R / FPPI | R@0.1 integer | integer at clean thr: R / FPPI | integer − float (R@0.1) |
|---|---|---|---|---|---|
| clean | 90.7 | 90.7 / 0.10 | 90.1 | 90.1 / 0.10 | -0.6 |
| low_light:1 | 66.2 | 73.8 / 0.52 | 61.5 | 70.0 / 0.53 | -4.7 |
| low_light:2 | 48.1 | 54.5 / 0.58 | 42.0 | 51.0 / 0.58 | -6.1 |
| low_light:3 | 10.8 | 12.2 / 0.27 | 7.9 | 8.7 / 0.25 | -2.9 |
| fog:1 | 87.8 | 85.7 / 0.05 | 87.5 | 84.3 / 0.05 | -0.3 |
| fog:2 | 80.5 | 72.6 / 0.03 | 77.8 | 73.2 / 0.03 | -2.7 |
| fog:3 | 60.6 | 49.9 / 0.01 | 64.7 | 51.9 / 0.01 | +4.1 |
| motion_blur:1 | 83.7 | 83.7 / 0.10 | 84.3 | 84.5 / 0.12 | +0.6 |
| motion_blur:2 | 43.7 | 43.1 / 0.09 | 43.7 | 44.3 / 0.11 | +0.0 |
| motion_blur:3 | 4.4 | 4.4 / 0.10 | 3.8 | 4.1 / 0.16 | -0.6 |
| sensor_noise:1 | 84.8 | 86.6 / 0.21 | 84.5 | 85.1 / 0.17 | -0.3 |
| sensor_noise:2 | 67.1 | 70.0 / 0.20 | 68.5 | 69.7 / 0.17 | +1.4 |
| sensor_noise:3 | 36.2 | 38.2 / 0.17 | 36.2 | 36.7 / 0.15 | +0.0 |
| jpeg:1 | 80.5 | 83.4 / 0.17 | 80.5 | 81.0 / 0.16 | +0.0 |
| jpeg:2 | 53.1 | 58.6 / 0.29 | 53.6 | 58.3 / 0.26 | +0.5 |
| jpeg:3 | 6.7 | 3.5 / 0.02 | 5.8 | 3.5 / 0.02 | -0.9 |
| occlusion:1 | 68.8 | 68.2 / 0.09 | 68.2 | 67.6 / 0.07 | -0.6 |
| occlusion:2 | 19.8 | 27.1 / 0.16 | 21.9 | 29.7 / 0.15 | +2.1 |
| occlusion:3 | 3.8 | 3.8 / 0.08 | 4.1 | 3.2 / 0.07 | +0.3 |

These six degradations are the ODD axes proposed to T31.0: lighting (low light), airborne particles (fog / dust), camera or subject motion (motion blur), sensor noise, compression (JPEG), partial occlusion by plant equipment.
<!-- /m27:degrade-prewid-test-tiny-yolov4-coco -->

<!-- m27:events-prewid-test -->
## Dreaded event on video: prewid test (T27.4)

tiny-yolov4-coco on prewid `test`, frame mode, persons ≥ 30 px, operating point 0.1 FPPI per stage. Frame rate **10 frames/s** (argument (assumption)): 592 frames = 1.0 min of video in 80 runs of consecutive frames (a gap in the frame numbers ends a run); presence 51 % of the frames. Per-frame miss rate over the frames with a person: float 9.57 %, integer 10.23 %.

| Stage | N | Events (miss runs ≥ N) | Episodes ≥ N frames | Events / h | Independent / h | Measured / independent |
|---|---|---|---|---|---|---|
| float | 1 | 20 | 50 | 1.22e+03 | 1.62e+03 | 0.75 |
| float | 2 | 5 | 41 | 304 | 130 | 2.35 |
| float | 3 | 2 | 33 | 122 | 10.4 | 11.7 |
| float | 5 | 1 | 24 | 60.8 | 0.0683 | 891 |
| float | 10 | 0 | 7 | 0 | 2.58e-07 | 0 |
| integer | 1 | 19 | 50 | 1.16e+03 | 1.72e+03 | 0.67 |
| integer | 2 | 7 | 41 | 426 | 147 | 2.89 |
| integer | 3 | 3 | 33 | 182 | 12.6 | 14.5 |
| integer | 5 | 1 | 24 | 60.8 | 0.0947 | 642 |
| integer | 10 | 0 | 7 | 0 | 4.99e-07 | 0 |

Runs used: CAM0 210994-211035, CAM0 211043-211094, CAM1 210268-210305, CAM1 213625-213652, CAM1 213659-213672, CAM1 213674-213691, CAM1 213876-213906, CAM2 210373-210417, CAM2 214233-214249, CAM2 214625-214641, CAM2 214814-214823, CAM3 216148-216233 (runs of at least 10 frames).
<!-- /m27:events-prewid-test -->

<!-- m27:events-prewid-lab -->
## Dreaded event on video: prewid lab (T27.4)

tiny-yolov4-coco on prewid `lab`, frame mode, persons ≥ 30 px, operating point 0.1 FPPI per stage. Frame rate **10 frames/s** (argument (assumption)): 3000 frames = 5.0 min of video in 1 runs of consecutive frames (a gap in the frame numbers ends a run); presence 52 % of the frames. Per-frame miss rate over the frames with a person: float 11.28 %, integer 9.11 %.

| Stage | N | Events (miss runs ≥ N) | Episodes ≥ N frames | Events / h | Independent / h | Measured / independent |
|---|---|---|---|---|---|---|
| float | 1 | 32 | 5 | 384 | 1.89e+03 | 0.204 |
| float | 2 | 23 | 5 | 276 | 212 | 1.3 |
| float | 3 | 17 | 5 | 204 | 23.8 | 8.56 |
| float | 5 | 11 | 4 | 132 | 0.302 | 438 |
| float | 10 | 6 | 4 | 72 | 5.44e-06 | 1.32e+07 |
| integer | 1 | 43 | 5 | 516 | 1.56e+03 | 0.331 |
| integer | 2 | 25 | 5 | 300 | 142 | 2.12 |
| integer | 3 | 18 | 5 | 216 | 12.9 | 16.8 |
| integer | 5 | 11 | 4 | 132 | 0.106 | 1.24e+03 |
| integer | 10 | 3 | 4 | 36 | 6.6e-07 | 5.45e+07 |

Runs used: CAMLAB01 0-2999 (runs of at least 10 frames).
<!-- /m27:events-prewid-lab -->

<!-- m27:deel-prewid -->
## From per-image errors to failures per hour (T27.5)

DEEL (Delseny 2021, §4.2.4.4) redone with the measured rates: dangerous failure = a person present in the zone and not reported for N consecutive frames.

| Assumption | Value | Source |
|---|---|---|
| Frame rate | 10 frames/s | argument (assumption) |
| Presence in the zone | 10 % of the time | assumption (`--presence`) |
| Per-frame miss probability p | 0.09114 | measured (tiny-yolov4-coco int, threshold 0.0953, prewid lab, frame mode) |
| Frames unreported before harm (N) | 5 | `--n`, or reaction budget 0.5 s × fps |
| Measured N-run rate on video | 132 /h of video | T27.4, presence 52 % of the frames |
| Independence | misses independent from frame to frame | row `independent` |
| Measured correlation | runs counted on consecutive frames | row `measured` |
| Every miss of N frames is dangerous | no detection by other means | conservative |
| SIL bounds | PFH < 10⁻⁵ / 10⁻⁶ / 10⁻⁷ per hour for SIL 1 / 2 / 3 | IEC 61508-1 table 3, high demand or continuous mode |

| Model | Dangerous failures / h | vs SIL 1 | vs SIL 2 | vs SIL 3 |
|---|---|---|---|---|
| independent | 0.0206 | +3.3 | +4.3 | +5.3 |
| measured | 25.2 | +6.4 | +7.4 | +8.4 |

Gaps in orders of magnitude (log10 of rate / bound; positive = bound missed). Present frames per hour: 3.6e+03.
<!-- /m27:deel-prewid -->

<!-- m27:unflagged-prewid-lab-tiny-yolov4-coco -->
## Unflagged dangerous misses: tiny-yolov4-coco, prewid lab (T27.7)

tiny-yolov4-coco integer on prewid `lab`, operating point 0.1 FPPI (threshold 0.0953). Unflagged miss: a person missed by the detector **and** in an image the monitor does not flag (per person); a presence episode of at least N frames holding a run of N or more missed frames not flagged within its first 1 frame(s) (per episode, T27.4); over all the persons (episodes). 95 % Clopper-Pearson interval; the alarm rate of each monitor is given apart, never folded in.

| Monitor | Unit | Unflagged / total | Rate % | 95 % CI | Alarms % |
|---|---|---|---|---|---|
| detector alone | person | 148 / 1574 | 9.40 | [8.01, 10.95] | 0.0 |
| tiny-yolov4-coco_prewid-lab__boxes__0.05__apply | person | 148 / 1574 | 9.40 | [8.01, 10.95] | 4.9 |
| tiny-yolov4-coco_prewid-lab__failure-miss-int__0.05__apply | person | 141 / 1574 | 8.96 | [7.59, 10.48] | 5.0 |
| tiny-yolov4-coco_prewid-lab__failure-tcp-int__0.05__apply | person | 143 / 1574 | 9.09 | [7.71, 10.61] | 5.0 |
| tiny-yolov4-coco_prewid-lab__mahalanobis__0.05__apply | person | 145 / 1574 | 9.21 | [7.83, 10.75] | 5.0 |
| tiny-yolov4-coco_prewid-lab__max_score__0.05__apply | person | 141 / 1574 | 8.96 | [7.59, 10.48] | 5.0 |
| tiny-yolov4-coco_prewid-lab__norm__0.05__apply | person | 148 / 1574 | 9.40 | [8.01, 10.95] | 5.0 |
| tiny-yolov4-coco_prewid-lab__patterns__0.05__apply | person | 148 / 1574 | 9.40 | [8.01, 10.95] | 2.5 |
| detector alone | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
| tiny-yolov4-coco_prewid-lab__boxes__0.05__apply | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
| tiny-yolov4-coco_prewid-lab__failure-miss-int__0.05__apply | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
| tiny-yolov4-coco_prewid-lab__failure-tcp-int__0.05__apply | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
| tiny-yolov4-coco_prewid-lab__mahalanobis__0.05__apply | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
| tiny-yolov4-coco_prewid-lab__max_score__0.05__apply | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
| tiny-yolov4-coco_prewid-lab__norm__0.05__apply | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
| tiny-yolov4-coco_prewid-lab__patterns__0.05__apply | episode (N = 5) | 3 / 4 | 75.00 | [19.41, 99.37] | – |
<!-- /m27:unflagged-prewid-lab-tiny-yolov4-coco -->

<!-- m27:events-mot17-train -->
## Dreaded event on video: mot17 train (T27.4)

tiny-yolov4-coco on mot17 `train`, track mode, persons ≥ 30 px, operating point 0.1 FPPI per stage. Frame rate **14, 25, 30 frames/s by sequence** (seqinfo.ini): 5316 frames = 3.6 min of video in 7 runs of consecutive frames (a gap in the frame numbers ends a run); presence 100 % of the frames. Per-frame miss rate over the person-frames: float 78.72 %, integer 80.48 %.

| Stage | N | Events (miss runs ≥ N) | Episodes ≥ N frames | Events / h | Independent / h | Measured / independent |
|---|---|---|---|---|---|---|
| float | 1 | 2677 | 552 | 4.5e+04 | 3.2e+05 | 0.141 |
| float | 2 | 1938 | 546 | 3.26e+04 | 2.5e+05 | 0.13 |
| float | 3 | 1608 | 544 | 2.7e+04 | 1.96e+05 | 0.138 |
| float | 5 | 1242 | 531 | 2.09e+04 | 1.2e+05 | 0.174 |
| float | 10 | 868 | 511 | 1.46e+04 | 3.55e+04 | 0.411 |
| integer | 1 | 2847 | 552 | 4.79e+04 | 3e+05 | 0.159 |
| integer | 2 | 2038 | 546 | 3.43e+04 | 2.41e+05 | 0.142 |
| integer | 3 | 1648 | 544 | 2.77e+04 | 1.93e+05 | 0.144 |
| integer | 5 | 1266 | 531 | 2.13e+04 | 1.24e+05 | 0.172 |
| integer | 10 | 869 | 511 | 1.46e+04 | 4.07e+04 | 0.359 |

Runs used: MOT17-02 1-600, MOT17-04 1-1050, MOT17-05 1-837, MOT17-09 1-525, MOT17-10 1-654, MOT17-11 1-900, MOT17-13 1-750 (runs of at least 10 frames).
<!-- /m27:events-mot17-train -->

<!-- m27:conformal-prewid-test-tiny-yolov4-coco -->
## Conformal person threshold: tiny-yolov4-coco, prewid test (T29.2)

tiny-yolov4-coco on prewid `test` (592 images): person score threshold calibrated on one half for a target miss rate α, checked on the other half, over 20 random splits; halves made of blocks of 25 consecutive frames sent to one side at random, so neighbouring frames do not leak across. A person is missed when the greedy VOC matching gives it no detection above the threshold (the miss of R@0.1). **Per person**: split conformal, bound k / (n + 1) with k = ⌊α (n + 1)⌋ and in brackets the 5-95 % range of Beta(k, n + 1 − k), the spread a calibration set of n persons should give; it assumes the persons exchangeable, which the persons of one frame are not. **Per image**: conformal risk control (Angelopoulos 2022) on the share of each image's persons missed, exchangeable images only. Values: mean [min, max] over the splits; miss in %. **Box margin**: conformal (1 − α) quantile of the relative enlargement that makes the matched detection contain the person, and its coverage on the test half.

| Stage | α | Method | Splits with a threshold | Threshold | Test miss % | Bound (Beta 5-95 %) | Splits with miss ≤ α | Test FPPI | Box margin | Box coverage % |
|---|---|---|---|---|---|---|---|---|---|---|
| float | 5.0 | per person (split conformal) | 11 / 20 | 0.0293 [0.002494, 0.03938] | 8.3 [1.8, 10.7] | 4.6 (2.3-7.5) | 1 / 11 | 1.018 [0.290, 6.480] | 0.208 [0.174, 0.249] | 96.0 [91.5, 98.9] |
| float | 5.0 | per image (risk control) | 20 | 0.1095 [0.02461, 0.2419] | 6.5 [0.7, 14.2] | ≤ 5.0 | 10 / 20 | 0.155 [0.003, 0.660] | – | – |
| float | 10.0 | per person (split conformal) | 20 | 0.07789 [0.02802, 0.1341] | 10.5 [2.7, 16.8] | 9.8 (6.4-13.8) | 9 / 20 | 0.138 [0.020, 0.549] | 0.134 [0.103, 0.163] | 89.3 [80.1, 96.4] |
| float | 10.0 | per image (risk control) | 20 | 0.2937 [0.1269, 0.6071] | 13.2 [3.0, 32.9] | ≤ 10.0 | 9 / 20 | 0.008 [0.000, 0.033] | – | – |
| integer | 5.0 | per person (split conformal) | 11 / 20 | 0.02276 [0.001638, 0.04671] | 8.1 [3.5, 10.2] | 4.6 (2.3-7.5) | 1 / 11 | 1.819 [0.297, 9.070] | 0.197 [0.161, 0.228] | 95.4 [90.5, 98.9] |
| integer | 5.0 | per image (risk control) | 20 | 0.105 [0.01879, 0.2226] | 6.3 [0.8, 15.0] | ≤ 5.0 | 10 / 20 | 0.202 [0.010, 1.353] | – | – |
| integer | 10.0 | per person (split conformal) | 20 | 0.08437 [0.02631, 0.1865] | 10.4 [2.7, 20.8] | 9.8 (6.4-13.8) | 8 / 20 | 0.175 [0.017, 0.822] | 0.138 [0.120, 0.159] | 89.1 [80.7, 95.8] |
| integer | 10.0 | per image (risk control) | 20 | 0.2857 [0.1216, 0.5925] | 12.5 [2.8, 30.1] | ≤ 10.0 | 10 / 20 | 0.012 [0.000, 0.047] | – | – |

**Under the degradations of T27.3** (per-person threshold calibrated on all the clean images; the degraded images are no longer exchangeable with them, so nothing bounds their miss rate):

| Stage | α | Condition | Threshold | Miss % | FPPI |
|---|---|---|---|---|---|
| float | 5.0 | clean (calibration) | 0.0184 | 4.7 | 1.079 |
| float | 5.0 | low_light:1 | 0.0184 | 20.7 **> α** | 2.542 |
| float | 5.0 | low_light:2 | 0.0184 | 34.7 **> α** | 2.811 |
| float | 5.0 | low_light:3 | 0.0184 | 82.8 **> α** | 1.971 |
| float | 5.0 | fog:1 | 0.0184 | 10.2 **> α** | 0.426 |
| float | 5.0 | fog:2 | 0.0184 | 16.3 **> α** | 0.196 |
| float | 5.0 | fog:3 | 0.0184 | 39.7 **> α** | 0.091 |
| float | 5.0 | motion_blur:1 | 0.0184 | 10.8 **> α** | 0.922 |
| float | 5.0 | motion_blur:2 | 0.0184 | 46.9 **> α** | 0.826 |
| float | 5.0 | motion_blur:3 | 0.0184 | 88.9 **> α** | 1.042 |
| float | 5.0 | sensor_noise:1 | 0.0184 | 7.6 **> α** | 1.456 |
| float | 5.0 | sensor_noise:2 | 0.0184 | 23.6 **> α** | 1.436 |
| float | 5.0 | sensor_noise:3 | 0.0184 | 47.2 **> α** | 1.992 |
| float | 5.0 | jpeg:1 | 0.0184 | 9.6 **> α** | 1.211 |
| float | 5.0 | jpeg:2 | 0.0184 | 28.6 **> α** | 1.532 |
| float | 5.0 | jpeg:3 | 0.0184 | 90.7 **> α** | 0.409 |
| float | 5.0 | occlusion:1 | 0.0184 | 21.3 **> α** | 0.978 |
| float | 5.0 | occlusion:2 | 0.0184 | 58.3 **> α** | 0.990 |
| float | 5.0 | occlusion:3 | 0.0184 | 92.7 **> α** | 0.855 |
| float | 10.0 | clean (calibration) | 0.06993 | 9.6 | 0.083 |
| float | 10.0 | low_light:1 | 0.06993 | 26.5 **> α** | 0.473 |
| float | 10.0 | low_light:2 | 0.06993 | 45.8 **> α** | 0.508 |
| float | 10.0 | low_light:3 | 0.06993 | 88.3 **> α** | 0.203 |
| float | 10.0 | fog:1 | 0.06993 | 15.2 **> α** | 0.046 |
| float | 10.0 | fog:2 | 0.06993 | 27.7 **> α** | 0.029 |
| float | 10.0 | fog:3 | 0.06993 | 51.0 **> α** | 0.012 |
| float | 10.0 | motion_blur:1 | 0.06993 | 17.2 **> α** | 0.091 |
| float | 10.0 | motion_blur:2 | 0.06993 | 57.1 **> α** | 0.081 |
| float | 10.0 | motion_blur:3 | 0.06993 | 95.6 **> α** | 0.074 |
| float | 10.0 | sensor_noise:1 | 0.06993 | 13.7 **> α** | 0.184 |
| float | 10.0 | sensor_noise:2 | 0.06993 | 30.3 **> α** | 0.188 |
| float | 10.0 | sensor_noise:3 | 0.06993 | 62.7 **> α** | 0.139 |
| float | 10.0 | jpeg:1 | 0.06993 | 17.5 **> α** | 0.140 |
| float | 10.0 | jpeg:2 | 0.06993 | 41.4 **> α** | 0.250 |
| float | 10.0 | jpeg:3 | 0.06993 | 96.5 **> α** | 0.022 |
| float | 10.0 | occlusion:1 | 0.06993 | 32.9 **> α** | 0.076 |
| float | 10.0 | occlusion:2 | 0.06993 | 72.9 **> α** | 0.147 |
| float | 10.0 | occlusion:3 | 0.06993 | 96.2 **> α** | 0.071 |
| integer | 5.0 | clean (calibration) | 0.005653 | 4.7 | 3.731 |
| integer | 5.0 | low_light:1 | 0.005653 | 15.5 **> α** | 7.745 |
| integer | 5.0 | low_light:2 | 0.005653 | 30.3 **> α** | 6.578 |
| integer | 5.0 | low_light:3 | 0.005653 | 79.9 **> α** | 4.338 |
| integer | 5.0 | fog:1 | 0.005653 | 7.3 **> α** | 2.243 |
| integer | 5.0 | fog:2 | 0.005653 | 10.8 **> α** | 1.507 |
| integer | 5.0 | fog:3 | 0.005653 | 28.0 **> α** | 0.703 |
| integer | 5.0 | motion_blur:1 | 0.005653 | 7.6 **> α** | 3.649 |
| integer | 5.0 | motion_blur:2 | 0.005653 | 36.2 **> α** | 4.500 |
| integer | 5.0 | motion_blur:3 | 0.005653 | 80.5 **> α** | 3.647 |
| integer | 5.0 | sensor_noise:1 | 0.005653 | 6.4 **> α** | 4.785 |
| integer | 5.0 | sensor_noise:2 | 0.005653 | 16.9 **> α** | 5.285 |
| integer | 5.0 | sensor_noise:3 | 0.005653 | 34.4 **> α** | 7.539 |
| integer | 5.0 | jpeg:1 | 0.005653 | 7.6 **> α** | 4.508 |
| integer | 5.0 | jpeg:2 | 0.005653 | 20.4 **> α** | 5.537 |
| integer | 5.0 | jpeg:3 | 0.005653 | 85.4 **> α** | 2.551 |
| integer | 5.0 | occlusion:1 | 0.005653 | 14.9 **> α** | 3.422 |
| integer | 5.0 | occlusion:2 | 0.005653 | 46.9 **> α** | 3.238 |
| integer | 5.0 | occlusion:3 | 0.005653 | 81.9 **> α** | 2.823 |
| integer | 10.0 | clean (calibration) | 0.06949 | 9.6 | 0.105 |
| integer | 10.0 | low_light:1 | 0.06949 | 29.4 **> α** | 0.556 |
| integer | 10.0 | low_light:2 | 0.06949 | 48.7 **> α** | 0.645 |
| integer | 10.0 | low_light:3 | 0.06949 | 91.3 **> α** | 0.280 |
| integer | 10.0 | fog:1 | 0.06949 | 15.2 **> α** | 0.061 |
| integer | 10.0 | fog:2 | 0.06949 | 26.8 **> α** | 0.034 |
| integer | 10.0 | fog:3 | 0.06949 | 47.2 **> α** | 0.014 |
| integer | 10.0 | motion_blur:1 | 0.06949 | 15.5 **> α** | 0.132 |
| integer | 10.0 | motion_blur:2 | 0.06949 | 55.4 **> α** | 0.135 |
| integer | 10.0 | motion_blur:3 | 0.06949 | 95.6 **> α** | 0.182 |
| integer | 10.0 | sensor_noise:1 | 0.06949 | 14.6 **> α** | 0.199 |
| integer | 10.0 | sensor_noise:2 | 0.06949 | 30.3 **> α** | 0.198 |
| integer | 10.0 | sensor_noise:3 | 0.06949 | 63.0 **> α** | 0.164 |
| integer | 10.0 | jpeg:1 | 0.06949 | 17.8 **> α** | 0.169 |
| integer | 10.0 | jpeg:2 | 0.06949 | 41.1 **> α** | 0.282 |
| integer | 10.0 | jpeg:3 | 0.06949 | 96.5 **> α** | 0.027 |
| integer | 10.0 | occlusion:1 | 0.06949 | 32.1 **> α** | 0.086 |
| integer | 10.0 | occlusion:2 | 0.06949 | 69.1 **> α** | 0.159 |
| integer | 10.0 | occlusion:3 | 0.06949 | 96.8 **> α** | 0.074 |
<!-- /m27:conformal-prewid-test-tiny-yolov4-coco -->

<!-- m27:conformal-prewid-lab-tiny-yolov4-coco -->
## Conformal person threshold: tiny-yolov4-coco, prewid lab (T29.2)

tiny-yolov4-coco on prewid `lab` (3000 images): person score threshold calibrated on one half for a target miss rate α, checked on the other half, over 20 random splits; halves made of blocks of 100 consecutive frames sent to one side at random, so neighbouring frames do not leak across. A person is missed when the greedy VOC matching gives it no detection above the threshold (the miss of R@0.1). **Per person**: split conformal, bound k / (n + 1) with k = ⌊α (n + 1)⌋ and in brackets the 5-95 % range of Beta(k, n + 1 − k), the spread a calibration set of n persons should give; it assumes the persons exchangeable, which the persons of one frame are not. **Per image**: conformal risk control (Angelopoulos 2022) on the share of each image's persons missed, exchangeable images only. Values: mean [min, max] over the splits; miss in %. **Box margin**: conformal (1 − α) quantile of the relative enlargement that makes the matched detection contain the person, and its coverage on the test half.

| Stage | α | Method | Splits with a threshold | Threshold | Test miss % | Bound (Beta 5-95 %) | Splits with miss ≤ α | Test FPPI | Box margin | Box coverage % |
|---|---|---|---|---|---|---|---|---|---|---|
| float | 5.0 | per person (split conformal) | 9 / 20 | 0.04843 [0.005033, 0.1606] | 14.1 [8.8, 22.6] | 4.9 (3.7-6.3) | 0 / 9 | 1.253 [0.051, 3.727] | 0.252 [0.181, 0.290] | 91.1 [80.1, 97.0] |
| float | 5.0 | per image (risk control) | 14 / 20 | 0.169 [0.001683, 0.5251] | 9.8 [2.3, 26.8] | ≤ 5.0 | 3 / 14 | 2.479 [0.005, 12.379] | – | – |
| float | 10.0 | per person (split conformal) | 14 / 20 | 0.1602 [0.001395, 0.4457] | 16.7 [3.3, 36.7] | 9.9 (8.2-11.7) | 4 / 14 | 2.248 [0.005, 13.401] | 0.186 [0.097, 0.256] | 84.3 [48.6, 96.3] |
| float | 10.0 | per image (risk control) | 20 | 0.3512 [0.06528, 0.7799] | 13.1 [1.3, 45.1] | ≤ 10.0 | 7 / 20 | 0.023 [0.001, 0.240] | – | – |
| integer | 5.0 | per person (split conformal) | 14 / 20 | 0.03685 [0.001123, 0.2001] | 8.8 [3.3, 20.9] | 4.9 (3.7-6.3) | 5 / 14 | 5.526 [0.035, 14.307] | 0.254 [0.176, 0.304] | 91.7 [76.6, 97.7] |
| integer | 5.0 | per image (risk control) | 20 | 0.1441 [0.003133, 0.526] | 7.4 [1.0, 28.8] | ≤ 5.0 | 7 / 20 | 0.713 [0.007, 5.040] | – | – |
| integer | 10.0 | per person (split conformal) | 20 | 0.1332 [0.002835, 0.4349] | 12.7 [2.2, 39.9] | 9.9 (8.2-11.7) | 9 / 20 | 0.722 [0.007, 6.595] | 0.194 [0.108, 0.237] | 86.3 [52.3, 96.3] |
| integer | 10.0 | per image (risk control) | 20 | 0.3487 [0.1442, 0.777] | 13.4 [1.5, 46.8] | ≤ 10.0 | 7 / 20 | 0.013 [0.001, 0.025] | – | – |
<!-- /m27:conformal-prewid-lab-tiny-yolov4-coco -->

<!-- m27:vote-prewid-lab-tiny-yolov4-coco -->
## Temporal vote: tiny-yolov4-coco, prewid lab (T29.3)

tiny-yolov4-coco on prewid `lab`, persons ≥ 30 px. k-of-n vote of `yolo.infer.temporal` (IoU association ≥ 0.3, no Re-ID; a track is reported when detected in k of its last n frames, with its last box), at the T27.4 threshold (0.1 FPPI of the detector alone) and at a threshold re-tuned so the voted output stays at 0.1 FPPI. 1/1 is the detector alone. **Events**: miss runs of at least N frames (count, and per hour) as T27.4. **Added latency**: frames from the first report of the detector alone to the first report with the vote, per episode (mean / max; a hold can report earlier when it bridges a gap).

| Stage | k/n | Threshold of | Threshold | FPPI | Per-frame miss % | Events N = 3 (/h) | N = 5 (/h) | N = 10 (/h) | Added latency (frames) |
|---|---|---|---|---|---|---|---|---|---|
| integer | 1/1 | T27.4 | 0.0953 | 0.097 | 9.11 | 18 (216) | 11 (132) | 3 (36) | +0.00 / +0 |
| integer | 1/1 | 0.1 FPPI | 0.0946 | 0.097 | 9.11 | 18 (216) | 11 (132) | 3 (36) | +0.00 / +0 |
| integer | 1/2 | T27.4 | 0.0953 | 0.134 | 7.71 | 17 (204) | 10 (120) | 1 (12) | +0.00 / +0 |
| integer | 1/2 | 0.1 FPPI | 0.107 | 0.100 | 7.97 | 18 (216) | 10 (120) | 1 (12) | +0.25 / +1 |
| integer | 1/3 | T27.4 | 0.0953 | 0.160 | 6.18 | 14 (168) | 8 (96) | 1 (12) | +0.00 / +0 |
| integer | 1/3 | 0.1 FPPI | 0.118 | 0.100 | 7.20 | 16 (192) | 9 (108) | 1 (12) | +0.25 / +1 |
| integer | 2/2 | T27.4 | 0.0953 | 0.061 | 10.58 | 20 (240) | 13 (156) | 5 (60) | +1.25 / +3 |
| integer | 2/2 | 0.1 FPPI | 0.0762 | 0.100 | 10.01 | 18 (216) | 12 (144) | 5 (60) | +1.00 / +3 |
| integer | 2/3 | T27.4 | 0.0953 | 0.088 | 9.56 | 21 (252) | 12 (144) | 3 (36) | +1.00 / +2 |
| integer | 2/3 | 0.1 FPPI | 0.0884 | 0.100 | 9.56 | 21 (252) | 12 (144) | 3 (36) | +1.00 / +2 |
| integer | 3/3 | T27.4 | 0.0953 | 0.045 | 11.79 | 17 (204) | 14 (168) | 7 (84) | +6.50 / +20 |
| integer | 3/3 | 0.1 FPPI | 0.0689 | 0.098 | 11.22 | 19 (228) | 14 (168) | 5 (60) | +4.00 / +14 |
| integer | 3/5 | T27.4 | 0.0953 | 0.087 | 9.31 | 15 (180) | 13 (156) | 5 (60) | +5.00 / +15 |
| integer | 3/5 | 0.1 FPPI | 0.0866 | 0.100 | 9.31 | 15 (180) | 13 (156) | 5 (60) | +5.00 / +15 |
<!-- /m27:vote-prewid-lab-tiny-yolov4-coco -->

<!-- m27:vote-mot17-train-tiny-yolov4-coco -->
## Temporal vote: tiny-yolov4-coco, mot17 train (T29.3)

tiny-yolov4-coco on mot17 `train`, persons ≥ 30 px. k-of-n vote of `yolo.infer.temporal` (IoU association ≥ 0.3, no Re-ID; a track is reported when detected in k of its last n frames, with its last box), at the T27.4 threshold (0.1 FPPI of the detector alone) and at a threshold re-tuned so the voted output stays at 0.1 FPPI. 1/1 is the detector alone. **Events**: miss runs of at least N frames (count, and per hour) as T27.4. **Added latency**: frames from the first report of the detector alone to the first report with the vote, per episode (mean / max; a hold can report earlier when it bridges a gap).

| Stage | k/n | Threshold of | Threshold | FPPI | Per-frame miss % | Events N = 3 (/h) | N = 5 (/h) | N = 10 (/h) | Added latency (frames) |
|---|---|---|---|---|---|---|---|---|---|
| integer | 1/1 | T27.4 | 0.727 | 0.099 | 80.48 | 1648 (2.77e+04) | 1266 (2.13e+04) | 869 (1.46e+04) | +0.00 / +0 |
| integer | 1/1 | 0.1 FPPI | 0.727 | 0.099 | 80.48 | 1648 (2.77e+04) | 1266 (2.13e+04) | 869 (1.46e+04) | +0.00 / +0 |
| integer | 1/2 | T27.4 | 0.727 | 0.156 | 78.73 | 1521 (2.56e+04) | 1203 (2.02e+04) | 824 (1.39e+04) | -1.15 / +0 |
| integer | 1/2 | 0.1 FPPI | 0.774 | 0.098 | 81.05 | 1513 (2.54e+04) | 1198 (2.01e+04) | 840 (1.41e+04) | +2.09 / +89 |
| integer | 1/3 | T27.4 | 0.727 | 0.227 | 77.67 | 1415 (2.38e+04) | 1143 (1.92e+04) | 804 (1.35e+04) | -1.43 / +0 |
| integer | 1/3 | 0.1 FPPI | 0.817 | 0.100 | 82.85 | 1360 (2.29e+04) | 1121 (1.89e+04) | 823 (1.38e+04) | +9.54 / +345 |
| integer | 2/2 | T27.4 | 0.727 | 0.065 | 82.36 | 1654 (2.78e+04) | 1282 (2.16e+04) | 871 (1.46e+04) | +4.72 / +150 |
| integer | 2/2 | 0.1 FPPI | 0.679 | 0.100 | 80.73 | 1638 (2.75e+04) | 1278 (2.15e+04) | 879 (1.48e+04) | +0.35 / +77 |
| integer | 2/3 | T27.4 | 0.727 | 0.104 | 80.56 | 1498 (2.52e+04) | 1179 (1.98e+04) | 840 (1.41e+04) | +3.78 / +150 |
| integer | 2/3 | 0.1 FPPI | 0.728 | 0.100 | 80.66 | 1495 (2.51e+04) | 1175 (1.98e+04) | 842 (1.42e+04) | +4.15 / +150 |
| integer | 3/3 | T27.4 | 0.727 | 0.050 | 83.68 | 1754 (2.95e+04) | 1290 (2.17e+04) | 882 (1.48e+04) | +10.47 / +394 |
| integer | 3/3 | 0.1 FPPI | 0.667 | 0.100 | 81.22 | 1770 (2.98e+04) | 1277 (2.15e+04) | 894 (1.5e+04) | +2.57 / +108 |
| integer | 3/5 | T27.4 | 0.727 | 0.125 | 80.62 | 1390 (2.34e+04) | 1134 (1.91e+04) | 822 (1.38e+04) | +7.30 / +357 |
| integer | 3/5 | 0.1 FPPI | 0.751 | 0.100 | 81.81 | 1384 (2.33e+04) | 1128 (1.9e+04) | 820 (1.38e+04) | +11.98 / +400 |
<!-- /m27:vote-mot17-train-tiny-yolov4-coco -->
