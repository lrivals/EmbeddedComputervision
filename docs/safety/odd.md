# Operational design domain (M31, T31.0)

**Target standard**: to be given by ARCYS (IEC 61496, with TS 61496-4-3 for vision-based
protective devices, is expected). Until it is given, the ranges below are proposals drawn from
the measurements of the repo, not from the standard.

**Use case**: a fixed camera watches a dangerous industrial zone (a machine cell, a robot or
press envelope). The function reports whether a person is inside the zone; a person not
reported in time is the dreaded event ([requirements.md](requirements.md)).

Every parameter is linked to the evidence that measures it: a dataset, one of the six
degradations of `tools/degrade.py` ([T27.3](../tasks/M27-safety-metrics.md), severities 1 to 3),
or "not covered". Recall figures are R@0.1 (recall at 0.1 false positive per image) of
Tiny-YOLOv4 COCO (no fine-tuning) on PREWID test, float / integer, from
[results/person_safety.md](../../results/person_safety.md).

## Parameters

| # | Parameter | Proposed ODD range | Evidence | Measured | Status |
|---|---|---|---|---|---|
| O1 | Zone size, camera height | zone up to ≈ 10 m × 10 m, camera 3 to 6 m high, oblique view | PREWID: 4 industrial cameras + 1 lab camera, fixed, oblique ([M38](../tasks/M38-prewid.md)) | per camera R@0.1 87.3-93.3 % (float) | partial: geometry of the PREWID cameras not measured |
| O2 | Person height in the image | ≥ 80 px in the ODD; 30-80 px degraded; < 30 px outside | box-height bins of T27.1 | ≥ 80 px: 97.1 / 97.1 %; 30-80 px: 82.4 / 80.9 %; < 30 px: 0 / 0 (3 persons only) | covered (≥ 30 px); < 30 px not covered |
| O3 | Person distance | the distance at which a person stays ≥ 80 px high; to derive from O1 and the lens | O2 + camera geometry | — | not covered (needs the camera intrinsics of the ARCYS site) |
| O4 | Lighting | at least the light of severity 1 (k = 0.5, γ = 1.5) | `degrade.py low_light:1-3`; ExDark (T27.3, pending) | 66.2 / 61.5 %, 48.1 / 42.0 %, 10.8 / 7.9 % at severities 1 / 2 / 3 | partial: even severity 1 costs 25 points; the integer model loses more than float (−4.7, −6.1 points) |
| O5 | Dust, fog, smoke | transmission t ≥ 0.7 (`fog:1`) | `degrade.py fog:1-3` | 87.8 / 87.5 %, 80.5 / 77.8 %, 60.6 / 64.7 % | covered at severity 1 |
| O6 | Motion blur (camera vibration, running person) | blur ≤ 1 % of the width (`motion_blur:1`) | `degrade.py motion_blur:1-3` | 83.7 / 84.3 %, 43.7 / 43.7 %, 4.4 / 3.8 % | covered at severity 1 |
| O7 | Sensor noise | σ ≤ 0.03 (`sensor_noise:1`) | `degrade.py sensor_noise:1-3` | 84.8 / 84.5 %, 67.1 / 68.5 %, 36.2 / 36.2 % | covered at severity 1 |
| O8 | Compression of the video link | JPEG quality ≥ 30, or uncompressed | `degrade.py jpeg:1-3` | 80.5 / 80.5 %, 53.1 / 53.6 %, 6.7 / 5.8 % | covered at severity 1; uncompressed preferred |
| O9 | Occlusion by plant equipment | ≤ 25 % of the person hidden (`occlusion:1`) | `degrade.py occlusion:1-3` | 68.8 / 68.2 %, 19.8 / 21.9 %, 3.8 / 4.1 % | partial: severity 1 already costs 22 points |
| O10 | Posture (standing, bending, crouching, lying) | standing and walking only | PREWID has no posture label | — | not covered |
| O11 | PPE (helmet, high-visibility vest, coverall) | as on the PREWID sites | PREWID, uncontrolled; construction PPE datasets ([M43](../tasks/M43-construction-ppe.md)) | — | not covered (no per-PPE measure) |
| O12 | Number of persons | 0 to a few; empty frames included | PREWID test: 289 / 592 empty frames | R@0.1 with FPPI on the empty frames: 91.3 / 91.3 % | covered |
| O13 | Frame rate | 10 frames/s | assumption of T27.4 / T27.5; frame bound of Tiny-YOLOv2 92 ms ([latency_bound.md](../../results/latency_bound.md)) | — | assumed: the PREWID frames are not a 10 frames/s video (runs of consecutive frame numbers) |
| O14 | Network and input | Tiny-YOLOv4 COCO, integer, 416 × 416 letterbox | [results/person_safety.md](../../results/person_safety.md), [results/equivalence.md](../../results/equivalence.md) | C-sim == integer on 100 images | covered |

## Reading

- **The degradations combine**: each one is measured alone. The ODD at severity 1 of every axis
  is a box whose corners (e.g. low light and dust together) are not measured.
- **Low light and occlusion are the narrow axes**: they cost more than 20 points at their
  lowest severity. Either the ODD requires lighting and a camera placement that keep them at
  zero, or the detector needs fine-tuning on these conditions (M41-M43).
- **The monitor of M29 is the ODD guard**: an image outside the ODD must be flagged at run time.
  The shift AUROC of the T29.0 monitors is 0.81 (max score, ExDark) and 0.93 (activation
  patterns, FLIR) ([results/ood_monitor.md](../../results/ood_monitor.md)). That is not enough
  for a guard, and none of them has been measured on the six degradations yet (T29.1).
- **Not covered** (O3, O10, O11): they need data from the ARCYS site (camera geometry, posture
  and PPE labels). The ODD limits proposed by the explanation work (T36.6) are added here when
  they exist.
