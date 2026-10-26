# M38 — PREWID: workers in legacy industrial plants (optional)

Goal: bring PREWID (Pre-Industry Worker Image Dataset for Industrial Environments, CC BY
4.0) into the M11 chain and use it as the **industrial person set** that
[M27](M27-safety-metrics.md) lacks. The data chain is the one of the other datasets (prep,
loader in `DATASETS`, datasheet, M14 and M16 notebooks, Drive, fine-tuning). The safety
part runs the M27 metrics on PREWID: missed workers at a fixed false-alarm rate, per
distance and per camera, at the three stages.

PREWID is made of fixed IP cameras in legacy plants: four plant cameras at 640 × 480 and a
1920 × 1080 laboratory sequence, every frame boxed (class `person` only, YOLO format). Empty
frames make up almost half of the set. They are what a zone-watching camera sees most of
the time, and they are where false alarms are counted.

M27 says the literature has "no industrial person dataset", and T27.6 waits for the ARCYS
site images. PREWID is the public stand-in until those arrive; the ARCYS images later go
through the same loader slot. Three complementary sets follow the same chain:
[M41](M41-safe-unsafe-behaviours.md) (factory CCTV, auto-labelled), [M42](M42-soda.md)
(construction sites) and [M43](M43-construction-ppe.md) (persons and PPE).

| What it tests | Why |
|---|---|
| industrial domain | machines, racks, cluttered backgrounds, plant lighting: none of the M11 / M18 sets look like this |
| missed workers at fixed FPPI | person-only ground truth with many empty frames: the M27 metrics apply directly, without a class mapping |
| false alarms on empty scenes | 1,926 empty plant frames and 1,426 empty lab frames: FPPI is measured on true negatives |
| worker distance | median box height from 71 px (CAM2) to 152 px (CAM3) at 480 rows; a few under 20 px |
| unseen viewpoint | four cameras: leave-one-camera-out says whether the model learnt workers or backgrounds |
| lab → plant shift | the lab sequence is a controlled baseline; its score against the plant cameras measures the gap |
| sequences | consecutive frames: the dreaded event "worker present, not reported for N frames" of T27.4 on ground-level footage |
| out-of-domain baseline | `person` maps to VOC / COCO / CrowdHuman `person`: every pre-trained network gets a score with no training |

**State**: 7 of 12 tasks done (2026-10-26): T38.0-T38.6. Data chain, loader, datasheet,
notebooks, stats run, archive on the Drive, and the out-of-domain baseline (best: YOLOv4-tiny
and YOLOv7-tiny COCO at about 90 person AP50 on `test` with no training). Open: T38.7
(fine-tuning, GPU), T38.8-T38.10 (wait for T27.0, T27.1, T27.3, T27.4: `person_safety.py`,
`degrade.py` and `dreaded_event.py` do not exist yet), T38.11 (waits for a T38.7
checkpoint). The archive is in `data/PREWID PRE-industry Worker Image Dataset for Indus/`
(downloaded by hand). All tasks run on the PC.

## Summary

Counted on the local copy (2026-10-25).

| Property | Value |
|---|---|
| Name | PREWID, Pre-Industry Worker Image Dataset for Industrial Environments |
| Source | Mendeley Data, <https://data.mendeley.com/datasets/xgt8kyxyrz/1>, doi:10.17632/xgt8kyxyrz.1, V1 published 2026-06-03 |
| Authors | Javier Silvestre-Blanes, Teresa Albero-Albero, Jorge Zafrilla-Muñoz, Ruben Perez Llorens (Universitat Politècnica de València) |
| Citation | Silvestre-Blanes, J.; Albero-Albero, T.; Zafrilla-Muñoz, J.; Perez Llorens, R. (2026), "PREWID: PRE-industry Worker Image Dataset for Industrial Environments", Mendeley Data, V1, doi:10.17632/xgt8kyxyrz.1 |
| Licence | CC BY 4.0 (attribution required in the datasheet, the notebooks and the article) |
| Plants | not stated as a number; the page speaks of "the industrial plant" for CAM0-CAM3: one plant assumed |
| Frame rate | not published for the plant cameras; the lab mp4 of the local copy is damaged (`moov atom not found`, 40 MB, likely a truncated download), so its rate cannot be read either |
| Folder | `data/PREWID PRE-industry Worker Image Dataset for Indus/PREWID/`: `01_Images/`, `02_GroundTruth/`, `Readme.txt`; 1.2 GB |
| Images | 7,003 JPEG: CAM0 1,001, CAM1 1,000, CAM2 1,000, CAM3 1,002 at 640 × 480; CAMLAB01 3,000 at 1920 × 1080; plus `CAMLAB01_Sequence.mp4` (source of the lab frames) |
| Classes | 1: `person` (class id 0) |
| Annotations | one YOLO txt per frame in `02_GroundTruth/GT_<camera>/` (normalised centre and size; an empty file = no worker); the same boxes in `groundtruth.csv` (2,248 rows, CAM0-3) and `groundtruth_CAMLAB01.csv` (1,574 rows), `;`-separated, one row per box |
| Pairing | every image has its txt, and every txt has its image |
| Official split | none; the split of T38.1 below |
| Pre-trained mapping | `person` → VOC / COCO / CrowdHuman `person` |

| Camera | Frames | Empty | Boxes | Max per frame | Box height px, median (min-max) |
|---|---|---|---|---|---|
| CAM0 | 1,001 | 626 | 391 | 2 | 89 (42-272) |
| CAM1 | 1,000 | 303 | 753 | 3 | 111 (20-280) |
| CAM2 | 1,000 | 404 | 657 | 3 | 71 (16-236), 5 under 20 px |
| CAM3 | 1,002 | 593 | 447 | 3 | 152 (39-279) |
| CAMLAB01 | 3,000 | 1,426 | 1,574 | 1 | 392 (27-740) at 1080 rows |

Splits written by `tools/prep_datasets.py prewid` (T38.1); 0 disagreement with the CSVs,
0 box under 4 px:

| Split | Frames | Empty | Boxes | Per camera (frames) |
|---|---|---|---|---|
| train | 2,821 | 1,348 | 1,581 | CAM0 705, CAM1 709, CAM2 704, CAM3 703 |
| val | 590 | 289 | 324 | CAM0 150, CAM1 144, CAM2 146, CAM3 150 |
| test | 592 | 289 | 343 | CAM0 146, CAM1 147, CAM2 150, CAM3 149 |
| lab | 3,000 | 1,426 | 1,574 | CAMLAB01 |

### Readme against the files

The `Readme.txt` of the archive does not match the files on three points; the files are
the reference:

- it says **PNG, 8-bit grayscale**; the files are **JPEG, RGB, with colour** (channels
  differ by up to 200 levels on a sampled frame of each camera);
- it calls the lab ground-truth folder `GT_CAM_LAB_01`; it is `GT_CAMLAB01`;
- it gives no frame rate. The plant frame names (`210430_CAM0.jpg`) are **frame
  counters, not times** (in 36 to 76 % of the names, depending on the camera, the last two digit pairs go above 59);
  about 90 % of the neighbours are consecutive numbers, with gaps between runs. The lab
  frames come from a video whose frame rate can be read from the mp4.

## Preprocessing

```bash
tools/get_datasets.sh prewid                   # checks the hand-copied folder, then the prep
python tools/prep_datasets.py prewid --preview build/prewid_preview.png
tools/get_datasets.sh check                    # "prewid : prêt"
```

### Boxes and split (`tools/prep_datasets.py prewid`)

- **Boxes**: read from the per-frame txt files of `GT_<camera>/` (not the CSVs): YOLO
  centre / size × image size → corners in pixels. The CSVs are only a check: same number
  of boxes per frame, same values to 10⁻⁵.
- **Empty frames are kept** as images with no box. Unlike the rejects of M32, they are
  data: M27 counts the false alarms on them.
- **Split by time, not at random**: consecutive frames are near duplicates, so a random
  split puts the same worker in the same pose on both sides. Per plant camera, the frames
  sorted by number are cut into contiguous blocks: first 70 % `train`, next 15 % `val`,
  last 15 % `test` (`PREWID_SPLIT`). Every camera is then present in every split, at
  different times.
- **Empty frames and worker frames are cut separately** (found at T38.1): each camera
  starts with long blocks of empty frames (CAM0: 108, 64, then 402 in a row; CAM3: 171 then
  390) before the runs with workers. One time cut over all frames gave `test` 7 empty frames
  out of 600, too few to count false alarms. So the empty frames and the worker frames of a
  camera are each cut 70 / 15 / 15 by frame number; the cuts in the worker frames move
  forward to the next gap in the numbers (`time_split(..., snap=True)`), so one run of a
  worker never sits on two sides. The empty frames are cut exactly: a static background on
  both sides of a cut is no leak for a person detector.
- **Lab sequence**: CAMLAB01 is its own `lab` split, not mixed into train / val / test. It is
  an evaluation of the lab → plant gap, and an optional extra training set (variant
  `--lab-train`).
- **Leave-one-camera-out**: the prep also writes `annotations_loco_<camera>_{train,val,test}.json`
  (the 4 plant cameras, each one held out in turn: train / val are those of the other three
  cameras, test is every frame of the held-out one) for T38.7; the loader reads them as the
  splits `loco_<camera>_train` etc. This costs nothing at prep time and
  avoids a second preprocessing.
- **Checks**: the box is clipped to the image, and a box under 4 px on a side is reported
  (none expected: the smallest is 16 px high). Every box has class 0. The CSV disagreements
  and small boxes go to `data/prewid/checks.json`.
- **Per camera**: the loader takes `<split>:<camera>` (`test:CAM2`), so every tool with
  `--split` gives a per-camera result.
- **Output**: `data/prewid/annotations_{train,val,test,lab}.json` (corners, labels, camera,
  frame number), `data/prewid/images/<split>/` as hard links to the raw JPEGs (as ASL and
  Pet).

### Known limits

- **One class**: no PPE (helmet, vest), no machine, no forklift. The set measures "is there
  a worker", not "is the worker protected".
- **Four plant cameras** in one plant (to check in T38.0): the backgrounds repeat in train
  and test. The time split hides little of that, which is why leave-one-camera-out exists.
- **Frame rate unknown** for the plant cameras: the per-hour rates of T27.5 need it. The
  authors' page (T38.0) gives none; assumptions are written next to every per-hour number.
  The local lab mp4 is damaged; downloading it again may give the lab rate.
- **Annotation policy**: on the preview of T38.1 (`build/prewid_preview.png`, 10 frames),
  partly visible workers are boxed: the lab worker entering at the left edge, a worker half
  hidden by a pillar (CAM0), and far workers at the back of the aisle seen as heads and
  shoulders (CAM2, boxes under 20 px wide). So a miss on a partly visible worker is a real
  miss.
- **Lab boxes**: at most one worker per frame, and a large one (median 392 px): an easy
  baseline, not a stress test.

## Conventions

Those of M11, M14, M16, M18, M21 and M32 apply. Files to touch:

| File | Addition |
|---|---|
| `python/yolo/data/datasets.py` | `PREWID_CLASSES = ["person"]`, `load_prewid`, `DATASETS["prewid"]` (evaluation on `test`, also `lab`), `MAPPINGS[("prewid", "voc")]`, `MAPPINGS[("prewid", "coco")]` |
| `tools/prep_datasets.py` | `prewid` (txt → corners, CSV check, time-block split, `lab`, leave-one-camera-out files, preview) |
| `tools/get_datasets.sh` | `MARKER[prewid]=prewid/annotations_val.json`, `HOWTO` (manual download, licence, source), `prewid` command, `check` |
| `tools/data_stats.py` | `FICHES["prewid"]` (licence and attribution included) |
| `tools/notebooks/matrice.py` | `TRAINABLE`, `SPLIT_IMAGES`, `DATA_MARKERS`, `TASKS`, `STATS_TASKS` |
| `tools/notebooks/gabarits.py` | `QUESTIONS["prewid"]` |
| `tools/notebooks/jeux.py` | `LABELS`, `GROUND` (fixed camera, elevated view) |
| `tools/m11.sh` | profiles `prewid-prep`, `prewid-train` |
| tests | `test_datasets.py::test_prewid`, `test_prep_datasets.py::test_prep_prewid` (synthetic folder: 2 cameras, empty frames, a CSV that disagrees) |

`notebooks/prewid/`: `prewid_stats`, the out-of-domain notebooks of the pre-trained
networks (through `MAPPINGS`), and the fine-tuned `_infer`, `_train`, `_sweep`.

### Stats question

| Dataset | Own question (`stats` notebook) | Task |
|---|---|---|
| prewid | How far are the workers from each camera (box height per camera), and what share of the frames is empty? | T38.3 |

---

## A. Data

### [x] T38.0 — Inventory and `tools/get_datasets.sh`
- **Spec**: — · **Depends on**: T11.0 · **Runs on**: PC · **Size**: S
- **Deliverables**: `MARKER`, `HOWTO` (where to download, the CC BY 4.0 attribution), a
  `prewid` command that finds the hand-copied folder under `data/` and runs the prep;
  `prewid` in `check`. Summary completed: source URL or DOI, authors, citation, number of
  plants, frame rate of the plant cameras if published; frame rate of
  `CAMLAB01_Sequence.mp4` read from the file.
- **Acceptance**: `tools/get_datasets.sh check` gives `prewid : prêt` after the prep; the
  frame counts of the Summary are found again by the command; the Source row of the
  Summary is filled in.
- **Notes**: no automatic download until the source is known. The folder name keeps the
  truncated name of the archive (`… for Indus`); the command accepts it as is.
- **Done** (2026-10-26): `tools/get_datasets.sh prewid` prints 1,001 / 1,000 / 1,000 /
  1,002 / 3,000 frames and runs the prep; `check` gives `prewid : prêt`. Source, authors,
  citation from the Mendeley page (Summary). Frame rate: none published, and the lab mp4 of
  the local copy is damaged, so it stays unknown.

### [x] T38.1 — Boxes and split
- **Spec**: §5.1 · **Depends on**: T38.0 · **Runs on**: PC · **Size**: M
- **Deliverables**: `tools/prep_datasets.py prewid` (per-frame txt → corners, CSV check,
  70 / 15 / 15 contiguous blocks per plant camera, `lab` split, leave-one-camera-out files,
  hard links, report per camera and split); `--preview` (2 frames per camera, with at least
  one empty frame and one with several workers); `test_prep_prewid`.
- **Acceptance**: 7,003 frames written, 2,248 + 1,574 boxes, 0 disagreement with the CSVs;
  the preview shows every box on a worker. The annotation policy for partly visible
  workers is written under Known limits.
- **Notes**: the counts per split go into the Summary.
- **Done** (2026-10-26): 7,003 frames, 2,248 + 1,574 boxes, 0 disagreement, 0 box under
  4 px (about 2 s). The single time cut was replaced by separate cuts of empty and worker
  frames (Boxes and split, above). Preview `build/prewid_preview.png`; annotation policy
  under Known limits. Tests: `test_prewid_frame_and_split`, `test_prep_prewid`.

### [x] T38.2 — Loader and mappings
- **Spec**: §8.3 · **Depends on**: T38.1 · **Runs on**: PC · **Size**: S
- **Deliverables**: `PREWID_CLASSES`, `load_prewid`, `DATASETS["prewid"]`,
  `MAPPINGS[("prewid", "voc")]` and `("prewid", "coco")` (`person` → `person`);
  `test_prewid`; `test_mappings_valid` passes.
- **Acceptance**: `load("prewid")` gives the test counts of the Summary, empty frames
  included; `python tools/eval_voc.py --dataset prewid` runs with the VOC weights.
- **Done** (2026-10-26): `load("prewid")` gives the 592 test frames (289 empty); split
  `<split>:<camera>` for per-camera runs; `test_prewid`, `test_mappings_valid` pass;
  `eval_voc.py` runs (T38.6).

## B. Registry and notebooks

### [x] T38.3 — Datasheet
- **Spec**: — · **Depends on**: T38.2 · **Runs on**: PC · **Size**: S
- **Deliverables**: `FICHES["prewid"]` in `tools/data_stats.py` (with the attribution
  line), `LABELS` and `GROUND` in `tools/notebooks/jeux.py`, the stats question in
  `QUESTIONS` of `tools/notebooks/gabarits.py`.
- **Acceptance**: `tools/data_stats.py --dataset prewid --sample 8` passes.
- **Done** (2026-10-26): passes; PREWID is a ground view (`GROUND`).

### [x] T38.4 — Matrix, profiles, notebook generation
- **Spec**: — · **Depends on**: T38.3 · **Runs on**: PC · **Size**: S
- **Deliverables**: `matrice.py` entries; profiles `prewid-prep` (anchors, cfg) and
  `prewid-train` in `tools/m11.sh`; `notebooks/prewid/` and `notebooks/README.md`
  regenerated with `python -m tools.notebooks prewid`.
- **Acceptance**: `python -m tools.notebooks all --check` passes, or fails only for the
  reasons listed in T32.9 that predate M38.
- **Notes**: as in T32.9, do not run `python -m tools.notebooks all` while other executed
  notebooks would be rewritten.
- **Done** (2026-10-26): `prewid-prep` run (k = 6 anchors `15,37 20,56 29,79 40,110 82,88
  63,149`, mean IoU 0.778, `results/anchors_prewid.md`; cfg
  `build/m11/cfg/tiny-yolov3-prewid.cfg`); 11 notebooks in `notebooks/prewid/`. `python -m
  tools.notebooks prewid` also rewrote `analyse_jeux`, `analyse_balayages` and
  `figures_live` without their outputs; the executed versions were kept (they were stale
  before M38, as listed in T32.9). `--check` lists no PREWID notebook; its other failures
  predate M38.

### [x] T38.5 — Stats run and Drive
- **Spec**: — · **Depends on**: T38.4 · **Runs on**: PC · **Size**: S
- **Deliverables**: `notebooks/prewid/prewid_stats.ipynb` run with its outputs; a "PREWID"
  section in [stats-jeux.md](stats-jeux.md) with the answer to the question;
  `tools/get_datasets.sh push prewid`; a row in [donnees-drive.md](donnees-drive.md).
- **Notes**: the archive holds the annotation files and the hard-linked images, not the mp4.
- **Done** (2026-10-26): notebook executed; answer in stats-jeux.md (CAM2 farthest, median
  71 px; empty frames 45 % of the plant frames, same share in every split);
  `prewid.tar` (1.1 GB) on the Drive.

## C. Person detection and safety

### [x] T38.6 — Out-of-domain baseline
- **Spec**: §8.3 · **Depends on**: T38.2 · **Runs on**: PC · **Size**: S
- **Deliverables**: person AP50 on `test` and `lab` of the pre-trained networks without
  fine-tuning: `tiny-yolov2-voc`, `tiny-yolov3-coco`, `tiny-yolov3-crowdhuman` (when its
  checkpoint exists, [T15.14](M15-campagne-entrainement.md)), and the recent models of
  M24 / M26 where their float chain runs; table in this file, per camera.
- **Acceptance**: the table is written, with the camera where the networks do worst named.
- **Notes**: a quick first result, before any training: it says how far the industrial
  domain is from VOC / COCO.
- **Done** (2026-10-26): `build/m38/baseline/run.py` (settings of the `_infer` notebooks:
  stretch 416 for v2-v4, YOLOv5 letterbox for v7; VOC 11-point AP, IoU 0.5). Person AP50:

  | Network | test | CAM0 | CAM1 | CAM2 | CAM3 | lab |
  |---|---|---|---|---|---|---|
  | tiny-yolov2-voc | 25.5 | 16.9 | 26.8 | 22.3 | 49.7 | 45.2 |
  | tiny-yolov3-coco | 63.7 | 53.0 | 71.9 | 61.1 | 72.2 | 69.7 |
  | tiny-yolov4-coco | **89.9** | 89.9 | 90.5 | 86.8 | 90.3 | 85.8 |
  | tiny-yolov7-coco | 89.7 | 98.0 | 90.3 | 89.6 | 90.8 | 90.3 |
  | tiny-yolov3-crowdhuman | pending: no checkpoint yet (T15.14) | | | | | |

  **Worst camera**: CAM2 for the two best networks (86.8 and 89.6), the camera with the
  farthest workers (median 71 px, Stats question); CAM0 for v2 and v3. The two older
  Tiny-YOLO lose a lot on the plant (v2: 25.5), the two recent ones very little: the gap
  depends on the network as much as on the domain. Scores near 90 with the 11-point AP mean
  a precision close to 1 up to a recall around 0.9, with some workers never found; T38.8
  will count those misses at fixed FPPI. The lab is not easier than the plant for v4 (85.8)
  although its workers are larger; not analysed yet. YOLOv5n / v8n (M33 / M35) have no
  PREWID notebook in the matrix yet. Fine-tuning (T38.7) has to beat 89.9.

### [ ] T38.7 — Fine-tuning, sweep and unseen cameras
- **Spec**: §7 · **Depends on**: T38.5, T38.6 · **Runs on**: PC (Colab for the GPU runs) ·
  **Size**: L
- **Deliverables**: `tools/m11.sh prewid-prep`, then the `_sweep` and `_train` notebooks,
  tracked in [M15](M15-campagne-entrainement.md): fine-tuning from COCO and from CrowdHuman;
  AP50 on `test` and `lab`; leave-one-camera-out (train on 3 plant cameras, test on the
  4th) for the best recipe; variant with `lab` added to training.
- **Acceptance**: fine-tuned AP50 on `test` above the best baseline of T38.6; the
  leave-one-camera-out AP for each camera is written next to the in-camera AP.
- **Notes**: about 2,800 plant training frames: start from pre-trained weights, few epochs.
  Horizontal flip on. A large gap between the in-camera and leave-one-camera-out results
  means the model learnt the backgrounds.

### [ ] T38.8 — M27 metrics on PREWID
- **Spec**: §8.3 · **Depends on**: T38.7, T27.0, T27.1 · **Runs on**: PC (float, integer,
  C-sim gcc) · **Size**: M
- **Deliverables**: `tools/person_safety.py --dataset prewid`; a PREWID section in
  `results/person_safety.md`: recall at 0.1 and 0.01 FPPI, log-average miss rate, recall
  per box-height bin (distance to the camera) and per camera, at the three stages, for the
  fine-tuned network and the best pre-trained one; FPPI measured on the empty frames alone,
  next to the FPPI on all frames.
- **Acceptance**: the table is written; the loss per stage (quantization, engine) is
  attributed as in T27.1; the operating point of T27.2 (target recall 0.95 / 0.99) is
  placed on PREWID with the FPPI it costs.
- **Notes**: this is the industrial row M27 is missing; it answers T27.6 with public
  data until the ARCYS images arrive.

### [ ] T38.9 — Degraded conditions and grayscale input
- **Spec**: §8.3 · **Depends on**: T38.8, T27.3 · **Runs on**: PC · **Size**: S
- **Deliverables**: the degradations of `tools/degrade.py` (T27.3) on PREWID `test`,
  recall at 0.1 FPPI per degradation and severity, float and integer; the same with the
  frames converted to grayscale and fed as three equal channels.
- **Acceptance**: section in `results/person_safety.md`; the grayscale cost is given (the
  readme describes the target as grayscale cameras, so a plant camera may not deliver
  colour).

### [ ] T38.10 — Missed workers on sequences
- **Spec**: §8.3 · **Depends on**: T38.8, T27.4 · **Runs on**: PC · **Size**: S
- **Deliverables**: `tools/dreaded_event.py` on the consecutive runs of each plant camera
  and on the lab sequence: rate of "worker present, not reported during N consecutive
  frames" (N = 1…10), float and integer; per-frame and per-event rates side by side.
- **Acceptance**: section in `results/person_safety.md`; the runs used (frame numbers
  with no gap) and the frame-rate assumption written next to every per-hour number.
- **Notes**: T27.4 asks for a ground-level person video set; PREWID gives industrial
  footage. A gap in the frame numbers ends a run.

### [ ] T38.11 — Replay in the webcam demo
- **Spec**: — · **Depends on**: T38.7, M20 · **Runs on**: PC · **Size**: S
- **Deliverables**: `python tools/webcam.py --source "<…>/CAMLAB01_Sequence.mp4" --model
  ckpt:<run>/checkpoint.npz --cfg build/m11/cfg/tiny-yolov3-prewid.cfg --dataset prewid`
  and the same on a plant camera folder (`--source <…>/01_Images/CAM1`); a short note in
  [docs/webcam.md](../webcam.md).
- **Acceptance**: both replays run headless with `--save`, and a few saved frames are
  checked by eye.

## Out of scope

- PPE classes (helmet, vest, gloves): PREWID has none; see [M43](M43-construction-ppe.md).
- Tracking and re-identification of workers across frames or cameras.
- Pose, segmentation, action recognition.
- Training on the mp4 frames beyond the 3,000 already extracted.

## Recommended order

T38.0 → T38.1 → T38.2, then T38.6 at once (a baseline with no training). Then T38.3 →
T38.4 → T38.5, T38.7, and the safety tasks T38.8 → T38.9 / T38.10 once T27.0 and T27.1
exist. T38.11 last.
