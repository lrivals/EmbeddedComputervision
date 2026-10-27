# M41 — Safe and Unsafe Behaviours: auto-labelled factory CCTV (optional)

Goal: bring the *Video Dataset for Safe and Unsafe Behaviours* (factory CCTV clips, CC BY
4.0) into the M11 chain as a second industrial person set, next to
[PREWID](M38-prewid.md). The clips carry a **behaviour label per clip** and no box: the
clips are cut into image sequences, and the `person` boxes come from the
[M22](M22-autolabel.md) labeller, checked on PREWID first (same domain, boxes made by hand)
and then by hand on a val subset, as TrashNet in [M32](M32-trashnet-pets.md).

The data chain is the one of the other datasets (prep, loader in `DATASETS`, datasheet, M14
and M16 notebooks, Drive, fine-tuning). The safety use is a **complementary** one: per
behaviour class, does the detector (and the early heads of [M40](M40-early-alerts.md)) see
the worker in an unsafe situation? The false-alarm rate on empty scenes stays measured on
PREWID: these clips are selected around behaviours, so they are not a sample of what a
fixed camera sees in a day.

| What it tests | Why |
|---|---|
| a second industrial domain | another factory, other cameras, 1920 × 1080: PREWID-trained models meet unseen plants |
| auto-labels on industrial footage | first use of the M22 labeller on factory CCTV; PREWID gives its score before any label is trusted |
| frame sampling of clips | how many frames of a clip are worth keeping (T41.1): too dense gives near-duplicates, too sparse misses short events |
| workers in unsafe situations | per behaviour class: a worker at a machine, behind an open panel, near a loaded forklift, is the case a safety function must not miss |
| out-of-domain baseline | `person` maps to VOC / COCO / CrowdHuman `person`: every pre-trained network gets a score with no training |

**State**: T41.0 in progress (2026-10-27): `get_datasets.sh` entries written, Summary
filled from the dataset pages; the data is not on the PC yet (manual download). T41.2 done
(2026-10-27): `grounding-dino-tiny` at 0.5 for the labels. Comes out of a dataset search for industrial
footage next to PREWID (2026-10-26); [M42](M42-soda.md) and
[M43](M43-construction-ppe.md) come from the same search. All tasks run on the PC (Colab
for the GPU runs).

## Summary

From the dataset pages (2026-10-26), to check on the local copy in T41.0.

| Property | Value |
|---|---|
| Name | Video Dataset for Safe and Unsafe Behaviours |
| Source | Mendeley Data, <https://data.mendeley.com/datasets/xjmtb22pff/1> (V1, 2024-06-11, doi:10.17632/xjmtb22pff.1); card and FiftyOne copy on Hugging Face `Voxel51/Safe_and_Unsafe_Behaviours` |
| Licence | CC BY 4.0 (attribution required in the datasheet, the notebooks and the article). Cite: Önal, O., Dandıl, E. (2024), *Video dataset for the detection of safe and unsafe behaviours in workplaces*, Data in Brief 56, 110756, doi:10.1016/j.dib.2024.110756 |
| Clips | 691, CCTV of production facilities, 1920 × 1080 at 24 fps |
| Classes | 8 behaviour classes (Hugging Face card, to check on the local copy). Unsafe: Safe Walkway Violation, Unauthorized Intervention, Opened Panel Cover, Carrying Overload with Forklift. Safe: Safe Walkway, Authorized Intervention, Closed Panel Cover, Safe Carrying |
| Annotations | one behaviour class per clip; **no box**, no frame-level label |
| Official split | train 566 / test 125 clips (tags of the Hugging Face copy; the Mendeley page does not list files): folder layout to check on the local copy |
| Pre-trained mapping | after T41.3: `person` → VOC / COCO / CrowdHuman `person` |

Counts after extraction and labelling (frames, empty frames, boxes, per split and per
behaviour class) are filled in by T41.1 and T41.3.

## Preprocessing

```bash
tools/get_datasets.sh safe_unsafe                       # checks the hand-copied folder
python tools/prep_datasets.py safe_unsafe --extract     # clips → frames (T41.1)
python tools/autolabel.py --folder data/safe_unsafe/frames --classes person \
    --model grounding-dino-tiny --conf 0.5 --out data/safe_unsafe/autolabel --resume
python tools/prep_datasets.py safe_unsafe               # labels + split → annotations_*.json
tools/get_datasets.sh check                             # "safe_unsafe : prêt"
```

### Clips to image sequences (`--extract`, T41.1)

A clip at 24 fps gives 24 near-identical frames per second: labelling and training on all
of them costs time and weighs the static scenes too much. Sampling too sparsely misses a
worker who crosses the field of view in under a second. Three rules, applied per clip:

1. **Base stride**: one frame every `k` frames (default 2 fps, `k = 12` at 24 fps).
2. **Near-duplicate skip**: a sampled frame is dropped if its mean absolute difference to
   the last kept frame (grayscale, downscaled to 160 px wide) is under `--dup-thr`: a
   static scene gives few frames, a busy one keeps the base rate.
3. **Maximum gap**: a frame is always kept once `--gap-max` (default 1 s) has passed since
   the last kept one, and the first and last frame of each clip are always kept: no
   1 s window of a clip is left without a frame, whatever rule 2 says.

- **File names** `<clip>_f<frame:05d>.jpg` keep the frame number, so the kept frames of a
  clip are an ordered sequence (for "worker present, not reported for N frames" runs, as in
  [T27.4](M27-safety-metrics.md)); the timestamp is `frame / fps`.
- **Metadata**: clip id, behaviour class, frame number, timestamp go into the
  `annotations_*.json` entries.
- **Calibration** (T41.1): on about 20 clips spread over the 8 classes, the labeller runs on
  **every** frame (dense reference). For each setting of `k`, `--dup-thr`, `--gap-max`:
  share of the person-present intervals of at least 0.5 s with no kept frame (target 0),
  frames kept per clip, and share of kept pairs under the duplicate threshold. The chosen
  setting and its numbers go into the Summary.

### Boxes and split

- **Boxes**: from the labeller (`labels/<frame>.txt`, `autolabel.json` for the provenance);
  a box under 4 px on a side is reported, as in M38.
- **Empty frames**: a frame with no box at the threshold is kept as empty only after review
  (T41.4 for val; the "needs review" order of the Folder tab puts them first).
- **Split by clip, never by frame**: frames of one clip are near duplicates. Stratified on
  the 8 behaviour classes; the official split, if any, wins.
- **val is hand-checked**: every val frame is reviewed in the editor (T41.4); train keeps
  the labeller's boxes with the review of the "needs review" frames only.
- **Output**: `data/safe_unsafe/annotations_{train,val,test}.json` (corners, labels, clip,
  behaviour, frame, `edited` flag), and `build/safe_unsafe_preview.png`.

### Known limits

- **Labels are the labeller's**: their errors become targets (as in M21 and M32); the
  hand-checked val says what they are worth, and the test split inherits the labeller's
  misses unless it is reviewed too (T41.4 notes).
- **Selected clips**: built around behaviours, so empty frames are rare and not
  representative: no FPPI on empty scenes from this set.
- **Large frames**: 1920 × 1080 letterboxed to the network input makes far workers small;
  the stats notebook gives the box heights at the network input.

## Conventions

Those of M11, M14, M16, M22, M32 and M38 apply. Files to touch:

| File | Addition |
|---|---|
| `python/yolo/data/datasets.py` | `SAFE_UNSAFE_CLASSES = ["person"]`, `load_safe_unsafe`, `DATASETS["safe_unsafe"]`, `MAPPINGS[("safe_unsafe", "voc")]`, `MAPPINGS[("safe_unsafe", "coco")]`; `<split>:<behaviour>` filter as `<split>:<camera>` of PREWID |
| `tools/prep_datasets.py` | `safe_unsafe --extract` (sampling rules above), `safe_unsafe` (labels → corners, split by clip, preview) |
| `tools/get_datasets.sh` | `MARKER[safe_unsafe]`, `HOWTO` (manual download, licence, source), `safe_unsafe` command, `check` |
| `tools/data_stats.py` | `FICHES["safe_unsafe"]` (licence, attribution, "boxes from the M22 labeller") |
| `tools/notebooks/matrice.py`, `gabarits.py`, `jeux.py` | `TRAINABLE`, `SPLIT_IMAGES`, `DATA_MARKERS`, `TASKS`, `STATS_TASKS`; `QUESTIONS["safe_unsafe"]`; `LABELS`, `GROUND` |
| `tools/m11.sh` | profiles `safe_unsafe-prep`, `safe_unsafe-train` |
| tests | `test_prep_datasets.py::test_extract_sampling` (synthetic clip: static part, a short crossing, the max-gap rule), `test_prep_safe_unsafe`, `test_datasets.py::test_safe_unsafe` |

### Stats question

| Dataset | Own question (`stats` notebook) | Task |
|---|---|---|
| safe_unsafe | How many workers per frame and how tall at the network input, per behaviour class, and how many frames does a clip keep after sampling? | T41.6 |

---

## A. Data

### [ ] T41.0 — Inventory and `tools/get_datasets.sh`
- **Spec**: — · **Depends on**: T11.0 · **Runs on**: PC · **Size**: S
- **Deliverables**: `MARKER`, `HOWTO` (Mendeley page, CC BY 4.0 attribution, authors and
  DOI), `safe_unsafe` command; the Summary checked against the local copy (clip count,
  resolution, frame rate, the 8 class names, official split, folder layout)
- **Acceptance**: `tools/get_datasets.sh check` lists `safe_unsafe`; the Summary has no
  "to check" left
- **Notes**: manual download, as PREWID; the folder name keeps the archive's name.
- **Progress** (2026-10-27): `MARKER`, `HOWTO` (source, DOI, licence, citation),
  `safe_unsafe` command (clip count and folder tree of
  `data/Video Dataset for Safe and Unsafe Behaviours/`, the expected archive name) and
  `check` entry. Classes, clip counts and split filled from the Hugging Face card.
- **Remaining**: the download (by hand); the Summary checked on the local copy (resolution,
  frame rate, folder layout, the archive's folder name in `SAFE_UNSAFE_RAW`).

### [ ] T41.1 — Clips to image sequences
- **Spec**: — · **Depends on**: T41.0 · **Runs on**: PC · **Size**: M
- **Deliverables**: `tools/prep_datasets.py safe_unsafe --extract` with `--stride`,
  `--dup-thr`, `--gap-max` (rules of § Preprocessing); `test_extract_sampling`; the
  calibration table (20 clips, dense labeller reference) in this file
- **Acceptance**: the chosen setting misses no person-present interval of 0.5 s or more on
  the calibration clips; frames kept per clip and duplicate share are reported; every clip
  gives at least its first and last frame
- **Notes**: the dense reference uses the labeller of T41.2 at its threshold
  (`grounding-dino-tiny`, 0.5).

## B. Labels

### [x] T41.2 — Labeller check on industrial footage (PREWID)
- **Spec**: §8.3 · **Depends on**: T22.10, T38.2 · **Runs on**: PC · **Size**: S
- **Deliverables**: `tools/autolabel_eval.py --dataset prewid` (`grounding-dino-tiny`,
  `owlv2-base`; 100 frames of `test`, fixed seed; also `lab`) →
  `results/autolabel_prewid-test.md`; a row "industrial CCTV" in the T22.10 table of
  [M22](M22-autolabel.md) and a threshold in [docs/autolabel.md](../autolabel.md)
- **Acceptance**: person AP50, P / R by threshold and s / image for both labellers; a
  labeller and a threshold advised for factory CCTV, or the domain declared unfit
- **Notes**: PREWID boxes are hand-made, so this is the labeller's score on the domain,
  as the Pet trimaps of T32.7. Empty frames count: false boxes on them are the labeller's
  false alarms.
- **Done** (2026-10-27, `results/autolabel_prewid-test.md`, `results/autolabel_prewid-lab.md`,
  100 frames each, seed 0, CPU):

  | labeller | split | AP50 | AP75 | P / R @ 0.5 | P / R @ 0.55 | s / image |
  |---|---|---|---|---|---|---|
  | `grounding-dino-tiny` | test | 0.989 | 0.989 | 0.95 / 0.98 | 0.95 / 0.98 | 30.6 |
  | `grounding-dino-tiny` | lab | 0.964 | 0.961 | 0.94 / 0.91 | 1.00 / 0.83 | 22.5 |
  | `owlv2-base` | test | 0.990 | 0.902 | 1.00 / 0.57 @ 0.2 | 0.00 / 0.00 | 23.0 |
  | `owlv2-base` | lab | 0.898 | 0.884 | 1.00 / 0.06 @ 0.2 | 0.00 / 0.00 | 19.1 |

  - **Advised**: `grounding-dino-tiny` at **0.5**: the domain is fit, better than VOC
    (0.832). At 0.55 the new `lab` camera loses 17 % of the persons, at 0.4 the precision
    falls to 0.64 on `lab`.
  - **OWLv2** ranks the persons as well (AP50 0.990 on `test`) but its scores stay under
    about 0.3: no fixed threshold gives both P and R, so it is not advised.
  - Row "industrial CCTV" in the T22.10 table of [M22](M22-autolabel.md), threshold in
    [docs/autolabel.md](../autolabel.md).

### [ ] T41.3 — Batch labelling
- **Spec**: — · **Depends on**: T41.1, T41.2 · **Runs on**: PC (CUDA if present) ·
  **Size**: M
- **Deliverables**: `tools/autolabel.py --folder data/safe_unsafe/frames --classes person
  --model grounding-dino-tiny --conf 0.5 --out data/safe_unsafe/autolabel --resume`; counts (frames,
  boxes, frames with no box) per behaviour class in the Summary
- **Acceptance**: every extracted frame has a label file; `autolabel.json` holds the model,
  version, prompt and threshold
- **Notes**: long run (CPU: several s / image): run it as the T22.10 runs (parallel
  processes, `OMP_NUM_THREADS`), resume with `--resume`.

### [ ] T41.4 — Hand check of val
- **Spec**: — · **Depends on**: T41.3 · **Runs on**: PC · **Size**: M
- **Deliverables**: every val frame reviewed in the Folder tab ("needs review" first);
  P / R of the raw auto-labels against the reviewed val written in the Summary
- **Acceptance**: val frames all carry `edited` or "seen" in `autolabel.json`; the P / R
  is reported
- **Notes**: if the P / R is well under the T41.2 numbers, review test too before T41.7,
  else its AP measures agreement with the labeller.

## C. Registry and notebooks

### [ ] T41.5 — Split, loader and mappings
- **Spec**: §8.3 · **Depends on**: T41.4 · **Runs on**: PC · **Size**: S
- **Deliverables**: `tools/prep_datasets.py safe_unsafe` (split by clip, preview);
  `SAFE_UNSAFE_CLASSES`, `load_safe_unsafe`, `DATASETS["safe_unsafe"]`, `MAPPINGS`;
  `test_prep_safe_unsafe`, `test_safe_unsafe`
- **Acceptance**: `load("safe_unsafe")` gives the test counts of the Summary; no clip in
  two splits (test); `test:<behaviour>` works

### [ ] T41.6 — Datasheet, notebooks, stats run and Drive
- **Spec**: — · **Depends on**: T41.5 · **Runs on**: PC · **Size**: S
- **Deliverables**: `FICHES["safe_unsafe"]`; `matrice.py`, `gabarits.py`, `jeux.py`
  entries; profiles `safe_unsafe-prep`, `safe_unsafe-train`; `safe_unsafe_stats` run with
  outputs, answer in [stats-jeux.md](stats-jeux.md); archive on the Drive (frames and
  annotations, not the mp4)
- **Acceptance**: `tools/data_stats.py --dataset safe_unsafe --sample 8` and
  `python -m tools.notebooks all --check` pass
- **Notes**: as T38.4, do not run `python -m tools.notebooks all` while other executed
  notebooks hold outputs.

## D. Person detection and safety

### [ ] T41.7 — Out-of-domain baseline and PREWID transfer
- **Spec**: §8.3 · **Depends on**: T41.5, T38.7 · **Runs on**: PC · **Size**: S
- **Deliverables**: person AP50 on `test` of the pre-trained networks (settings of T38.6)
  and of the PREWID fine-tuned checkpoint; per behaviour class; table in this file
- **Acceptance**: the table is written, with the behaviour classes where the networks do
  worst named

### [ ] T41.8 — Fine-tuning on auto-labels
- **Spec**: §7 · **Depends on**: T41.6, T41.7 · **Runs on**: PC (Colab for the GPU runs) ·
  **Size**: M
- **Deliverables**: `tools/m11.sh safe_unsafe-prep`, `_train` notebook; AP50 on the
  hand-checked val and on `test`; AP50 on PREWID `test` of the result (cross-plant)
- **Acceptance**: fine-tuned AP50 above the best baseline of T41.7; the PREWID score of
  the model trained on PREWID + Safe and Unsafe against PREWID alone is reported
- **Notes**: auto-labels as targets: report the val AP against the reviewed boxes only.

### [ ] T41.9 — Unsafe situations in the safety metrics
- **Spec**: §8.3 · **Depends on**: T41.8, T27.0, T27.4 · **Runs on**: PC (float, integer)
  · **Size**: S
- **Deliverables**: section in `results/person_safety.md`: missed workers per behaviour
  class, at the operating point fixed on PREWID empty frames (not on this set); the
  dreaded event of T27.4 on the kept-frame sequences of the clips; the same rows for the
  early heads of T40.0 when they exist
- **Acceptance**: the table is written; the classes where workers are missed most are
  named, and the operating point's origin (PREWID) is stated
- **Notes**: frame gaps are those of T41.1, not the camera's: N frames is a time given by
  the sampling.

## Out of scope

- Behaviour classification and action recognition (the clip labels stay metadata).
- Boxes for forklifts, machines or panels.
- FPPI on empty scenes from this set (PREWID measures it).
- Tracking and re-identification.

## Recommended order

T41.0 → T41.2 (labeller check, no extraction needed) → T41.1 → T41.3 → T41.4 → T41.5 →
T41.6, then T41.7 at once. T41.8 after T38.7, T41.9 once T27.0 and T27.4 exist.
