# M21 — ASL Alphabet: hand pseudo-boxes for a classification dataset (optional)

Goal: bring the Kaggle [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
dataset (American Sign Language letters) into the M11 chain, like the M18 datasets: a
loader in `DATASETS`, a datasheet, the M14 and M16 notebooks (stats, fine-tuning, sweep,
inference) and the archive on the Drive. The fine-tuned model then reads sign letters live
in the M20 webcam demo.

ASL Alphabet is a **classification** dataset: one label per image, no box. The YOLO head
needs boxes, so this milestone makes them on disk with a hand landmark detector
(MediaPipe). That is what the dataset tests in the chain:

| What it tests | Why |
|---|---|
| pseudo-labels | boxes come from another model, not from a human: their errors become training targets |
| many close classes | 28 classes told apart by finger positions, one object per image |
| negatives | `nothing` images (background only) carry no box |
| low resolution | 200×200 images, upscaled ×2 to the 416 input |
| domain shift | one signer, a few indoor backgrounds: the webcam (another person, another room) is out of domain |

**State**: 6 of 12 tasks done (2026-10-12). Pseudo-boxes, loader, datasheet, registry and
the 4 notebooks are in place. The detection rate is below its target (T21.3, open). The
`stats` notebook run (T21.7), the Drive upload (T21.8),
the fine-tuning (T21.9), the webcam check (T21.10) and the `SIZE=224` variant (T21.11)
remain.

## Summary

Counts from `tools/prep_datasets.py asl` (2026-10-12).

| Split | Images | Boxes | Left out (no hand found) | Source |
|---|---|---|---|---|
| train | 69 984 | 67 434 | 3 966 (5.4 %) | frames n° 1 to 2 550 of each class |
| val | 6 538 | 6 088 | 6 512 (51.7 %) | frames n° 2 551 to 3 000 of each class (evaluation split) |
| test | 25 | 24 | 3 (A, E, N) | the official test images, one per class, no `del` |

| Property | Value |
|---|---|
| Classes | 28: A to Z, `del`, `space` (`nothing` = negative image, not a class) |
| Resolution | 200×200 RGB JPEG, with a 2 px blue frame |
| Raw images | 29 × 3 000 train, 28 test (1.2 GB) |
| Objects per image | 1 (letters), 0 (`nothing`) |
| Licence | GPL-2.0 according to the Kaggle page (to confirm) |
| Pre-trained mapping | none: VOC and COCO have no hand class |

Hand detection rate: 87.5 % of the letter images get a box. Hardest classes:
N (67.5 % kept), `space` (73.4 %), B (78.5 %), `del` (79.4 %), M (80.2 %); F keeps 99.7 %. In 119 of the 3 000 `nothing` images the detector still sees a hand;
these images stay negatives.

Left-out letter images, by reason: no hand found 8 563, landmarks out of the frame 482,
box under 24 px 1 436 (10 481 in all).

**Val is the hard end of the recording.** The detector loses 52 % of the val letter
images but 5.4 % of the train ones. Its failure rate per block of 100 frames stays
between 1 and 14 % up to n° 2 500, then jumps to 46-71 % for n° 2 501 to 2 900: these
frames are darker (mean luminance about 125 against 145 for the middle of the recording,
measured on one frame per class). So `val` is a separate, harder stretch of video,
which keeps it free of near-duplicates of train, but it holds the "easy half" of that
stretch, unevenly across classes: from 78 images (X) to 450 (F, `nothing`).
Alternative if this hurts (T21.9): val as interleaved blocks of 150 frames spread over
the whole recording, representative of train but closer to it.

## Preprocessing

```bash
pip install -e "python[asl]"                 # mediapipe (never imported by yolo/)
tools/get_datasets.sh kaggle-download asl    # Kaggle API → data/ASL Alphabet/, then prep
# or, with the raw folder already in data/:
tools/get_datasets.sh kaggle                 # runs tools/prep_datasets.py asl if needed
python tools/prep_datasets.py asl --preview build/asl_preview.png   # boxes to check by eye
tools/get_datasets.sh check                  # "asl : prêt"
```

### Raw format

- `data/ASL Alphabet/asl_alphabet_train/asl_alphabet_train/<class>/<class><n>.jpg`,
  n = 1 to 3 000, 29 folders (A–Z, `del`, `space`, `nothing`).
- `data/ASL Alphabet/asl_alphabet_test/asl_alphabet_test/<class>_test.jpg`: 28 images
  (no `del`).
- One signer, filmed with a webcam in front of a few indoor backgrounds (window, wall),
  with varied lighting. Consecutive numbers are consecutive video frames: near duplicates.
- Every image has a 2 px blue frame. It is the same for all classes, so it gives no clue
  about the class; it is left in place (no image rewrite).

### Hand pseudo-boxes (`tools/prep_datasets.py asl`)

- **Detector**: MediaPipe `HandLandmarker` (Tasks API), model `hand_landmarker.task`
  downloaded once to `weights/`. One hand per image (`num_hands=1`), detection and
  presence thresholds at 0.1 (`ASL_CONF`).
- **Cascade** (`ASL_STEPS`): the palm detector misses many hands at 200 px. Each image is
  tried upscaled ×2, then upscaled ×2 with 100 px of edge padding (hands cut by the
  border). The first hit wins. Measured on 20 random images per class (560 letter
  images): 87 % found by the first try, 92 % by both. A third, mirrored try added 1 %
  but its hands were mostly wrong (44 % of its train boxes under 30 px, 76 % in val):
  it was dropped.
- **Box**: min/max of the 21 landmarks, widened by 15 % of their extent on each side
  (`ASL_MARGIN`: landmarks are joint centres, the skin goes beyond), clipped to the image.
  The box covers the hand from the wrist up, not the forearm.
- **Sanity checks** (`asl_check`): a first full run showed about 1 box in 6 on the preview
  tiny and stuck to the border, from landmarks placed mostly outside the frame. A
  detection is now dropped when fewer than 90 % of its landmarks lie in the image
  (`ASL_MIN_INSIDE`, reason `outside`) or when its box is under 24 px on a side
  (`ASL_MIN_SIDE`, reason `small`).
- **Letter image with no valid hand**: left out, listed with its reason (`no hand`,
  `outside`, `small`) in `data/asl/rejected.json`. It is never kept as a negative image:
  the hand is there, only the detector missed it.
- **`nothing` image**: kept as a negative (no box), even if the detector finds a hand
  (counted in the report, field `hand`).
- **Split**: frames n° ≥ 2 551 of each class (`ASL_VAL_FROM`, 15 %) form `val`, the rest
  `train`. A random split would put neighbour frames on both sides. The official test set
  (28 images) is the `test` split: indicative only, too small for a stable mAP.
- **Output**: `data/asl/annotations_{train,val,test}.json`, a compact list (image, size,
  class, frame n°, corners, labels, handedness, handedness score, cascade step), and
  `data/asl/images/{train,val,test}/` as **hard links** to the raw JPEGs (no copy).
- **Cache**: the detections (landmarks, handedness, score, step) go to
  `data/asl/detections.json`, the slow part (about 10 min on 18 cores). The annotation
  files are rebuilt from it at each run, so changing `ASL_MARGIN`, `ASL_MIN_INSIDE` or
  `ASL_MIN_SIDE` takes seconds. Delete the cache to detect again (new `ASL_STEPS` or
  `ASL_CONF`).

### Known limits of the pseudo-boxes

- The detector errs on dark or blurred hands and on closed fists (A, E, M, N, S, T): the
  rejected images are biased towards these classes, and the model sees fewer of them.
- A wrong landmark set (hand on the background) gives a wrong box with the right class.
  `--preview` draws 2 val images per class for a visual check (T21.3).
- `val` is scored against pseudo-boxes too: its mAP measures agreement with MediaPipe,
  not with a human annotator.

## Conventions

Those of M11, M14, M16 and M18 apply. Files touched:

| File | Addition |
|---|---|
| `python/yolo/data/datasets.py` | `ASL_CLASSES`, `ASL_VAL_FROM`, `load_asl`, `DATASETS["asl"]` (no `MAPPINGS`) |
| `tools/prep_datasets.py` | `asl`: `HandDetector`, `landmarks_to_box`, `prep_asl`, `asl_preview` |
| `tools/get_datasets.sh` | `MARKER`, `HOWTO`, `KAGGLE_ID` (`grassknoted/asl-alphabet`), `m21()`, `check` |
| `tools/data_stats.py` | `FICHES["asl"]` |
| `tools/notebooks/matrice.py` | `TRAINABLE`, `SPLIT_IMAGES`, `DATA_MARKERS`, `TASKS`, `STATS_TASKS` |
| `tools/notebooks/gabarits.py` | `QUESTIONS["asl"]` |
| `tools/notebooks/jeux.py` | `LABELS`, `GROUND` (close-up view) |
| `tools/m11.sh` | profiles `asl-prep` and `asl-train` |
| `python/pyproject.toml` | extra `asl` (mediapipe) |
| tests | `test_datasets.py::test_asl`, `test_prep_datasets.py` (`test_prep_asl`…) |

No out-of-domain notebook: without a hand class in VOC or COCO, `build_registry` makes
only the fine-tuned ones. `notebooks/asl/` holds `asl_stats`, `tiny-yolov3-asl_infer`,
`tiny-yolov3-asl_train` and `tiny-yolov3-asl_sweep`.

### Stats question

| Dataset | Own question (`stats` notebook) | Task |
|---|---|---|
| asl | How much of the input does the hand cover, and which classes lose the most images to the pseudo-box step? | T21.1 |

---

## A. Data

### [x] T21.0 — `tools/get_datasets.sh`
- **Depends on**: T11.0 · **Size**: S
- **Deliverables**: `MARKER[asl]=asl/annotations_val.json`, `HOWTO`, `KAGGLE_ID` and
  `KAGGLE_DIR` (`ASL Alphabet`); `m21()` called by `kaggle` (prep if the raw folder is
  there and the marker is missing); `asl` in `check` and in the list of manual datasets.
- **Acceptance**: `tools/get_datasets.sh check` gives `asl : prêt`.

### [x] T21.1 — Hand pseudo-boxes
- **Depends on**: T21.0 · **Size**: M
- **Deliverables**: `tools/prep_datasets.py asl` (cascade, box, sanity checks, rejects,
  split, hard links, detection cache, report); extra `asl` in `python/pyproject.toml`;
  tests `test_landmarks_to_box`, `test_asl_split`, `test_asl_check`, `test_prep_asl`
  (fake detector, no MediaPipe needed).
- **Acceptance**: the three annotation files and `rejected.json` written; a second run
  reuses `detections.json`.

### [x] T21.2 — Loader
- **Depends on**: T21.1 · **Size**: S
- **Deliverables**: `ASL_CLASSES` (28), `ASL_VAL_FROM`, `load_asl`, `DATASETS["asl"]`
  (evaluation on `val`, calibration and fine-tuning on `train`); `test_asl`;
  `test_mappings_valid` accepts a dataset without mapping.
- **Acceptance**: `load("asl")` gives the val counts of the summary.

### [ ] T21.3 — Visual check of the boxes
- **Depends on**: T21.1 · **Size**: S
- **Deliverables**: `python tools/prep_datasets.py asl --preview build/asl_preview.png`;
  detection rate per class in the summary.
- **Acceptance**: ≥ 90 % of the letter images kept, every class ≥ 75 %; boxes on the
  preview tight on the hand. Otherwise tune `ASL_STEPS`, `ASL_CONF` or `ASL_MARGIN`.
- **Result**: 87.5 % of the letter images kept, N lowest at 67.5 %: the target is
  **not met**. The first run (with a mirrored third try and no sanity checks) kept
  91.1 %, but about 1 box in 6 on the preview was a tiny box on the border. The checks
  traded 3.6 points of kept images for boxes that sit on the hand: on the second
  preview, all 56 letter boxes cover the hand; a few are loose or shifted (D, L, M, P,
  the raised finger sometimes cut).
- **Notes**: to raise N, B, M, E and A (closed fists, the hardest hands for MediaPipe),
  try `ASL_CONF` below 0.1 or a third try at ×3 with padding; check the preview again.

## B. Registry and notebooks

### [x] T21.4 — Datasheet
- **Depends on**: T21.2 · **Size**: S
- **Deliverables**: `FICHES["asl"]` in `tools/data_stats.py`; `LABELS` and `GROUND` in
  `tools/notebooks/jeux.py`.
- **Remains**: confirm the licence on the Kaggle page.

### [x] T21.5 — Matrix and profiles
- **Depends on**: T21.4 · **Size**: S
- **Deliverables**: `matrice.py` (`TRAINABLE`, `SPLIT_IMAGES`, marker, tasks);
  `QUESTIONS["asl"]` in `gabarits.py`; profiles `asl-prep` and `asl-train` in
  `tools/m11.sh`.

### [x] T21.6 — Notebook generation
- **Depends on**: T21.5 · **Size**: S
- **Deliverables**: `notebooks/asl/` (4 notebooks), `notebooks/README.md` regenerated.
- **Acceptance**: `python -m tools.notebooks all --check` passes; `tools/data_stats.py`
  passes on `asl` (`--sample 8`).

## C. Runs

### [ ] T21.7 — `stats` notebook run
- **Depends on**: T21.6 · **Size**: S
- **Deliverables**: `notebooks/asl/asl_stats.ipynb` run and versioned with its outputs;
  section in [stats-jeux.md](stats-jeux.md) (`tools/data_stats.py --report`).
- **Notes**: run again after `tools/m11.sh asl-prep`, so that collisions use the dataset's
  anchors.

### [ ] T21.8 — Drive upload
- **Depends on**: T21.1 · **Size**: S
- **Deliverables**: `tools/get_datasets.sh push asl`; row in
  [donnees-drive.md](donnees-drive.md).
- **Notes**: the archive follows the hard links of `data/asl/images/` (≈ 1 GB) and holds
  the annotation files: Colab does not need MediaPipe.

### [ ] T21.9 — Fine-tuning and sweep
- **Depends on**: T21.7, T21.8 · **Size**: L
- **Deliverables**: `tools/m11.sh asl-prep` (anchors, cfg with 28 classes), then the
  `_sweep` and `_train` notebooks (batch × subset, then one full run), tracked in
  [M15](M15-campagne-entrainement.md); mAP on `val` and on the 28 `test` images.
- **Notes**:
  - Horizontal flip stays on: a mirrored sign is the same letter signed with the left hand.
  - J and Z are moving signs; the dataset only has one frame of each.
  - Near-duplicate frames: an epoch is worth far fewer independent images than its count.
  - A classification-like score (is the top detection of each image the right letter?)
    reads more directly than the mAP here; to add to the `infer` notebook if needed.

### [ ] T21.10 — Webcam demo
- **Depends on**: T21.9, M20 · **Size**: S
- **Deliverables**: `python tools/webcam.py --model ckpt:<run>/checkpoint.npz
  --cfg build/m11/cfg/tiny-yolov3-asl.cfg --dataset asl` live; a short note in
  [docs/webcam.md](../webcam.md) (letters read, conditions).
- **Notes**: expect a drop against `val`: one signer and a few backgrounds in training.
  Fine-tuning on a few hundred frames of one's own hands (same prep script) is the
  natural next step.

### [ ] T21.11 — Input at 224 (optional)
- **Depends on**: T21.9 · **Size**: M
- **Deliverables**: `SIZE=224x224 tools/m11.sh asl-prep` and `asl-train`; mAP and MACs
  against 416.
- **Notes**: 200 px images gain nothing from a ×2 upscale; at 224 the network runs about
  3.4 × fewer MACs, a strong argument for the board.

## Out of scope

- A classification head (top-1 accuracy, softmax loss): the chain stays a detector.
- Hand landmarks as targets (keypoints): the head outputs boxes only.
- Moving signs (J, Z) as sequences.

## Recommended order

T21.0 → T21.1 → T21.2 → T21.4 → T21.5 → T21.6 are done. T21.3 (detection rate) can be
improved at any time from the cache. Then T21.7 and T21.8, T21.9, and last T21.10 and
T21.11.
