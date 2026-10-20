# M32 — TrashNet and Oxford-IIIT Pet: auto-labelled detection sets (optional)

Goal: bring two public datasets that have **no usable object box** into the M11 chain,
as M21 did for ASL Alphabet: a loader in `DATASETS`, a datasheet, the M14 and M16
notebooks (stats, fine-tuning, sweep, inference), the archive on the Drive, then a
fine-tuned Tiny-YOLO in the M20 webcam demo.

- [TrashNet](https://github.com/garythung/trashnet): photos of one piece of waste on a
  plain background, 6 classes. It is a **classification** set: the class is the folder,
  there is no box. The boxes come from the M22 labeller (Grounding DINO), which T22.10
  measured as usable at ground level. Demo: waste sorting on the webcam or the board.
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/): photos of cats and dogs,
  37 breeds. Its only box is a tight box around the **head**; every image also has a
  hand-made pixel mask of the animal (trimap). The whole-body box comes from that mask,
  exactly and for free; the M22 labeller is run on the same images and scored against
  the mask boxes, which measures how tight the M22 boxes are (T32.7).

This milestone comes out of T22.11 (ranks 2 and 3 of the datasets worth labelling).

| What it tests | Why |
|---|---|
| auto-labels as training targets (TrashNet) | boxes from an open-vocabulary model, not from a human: as in M21, their errors become targets; a hand-checked val subset says what they are worth |
| a reference for the labeller (Pet) | mask boxes made by hand: the first measurement of M22 box tightness on a domain where it is usable |
| fine-grained classes (Pet) | 37 breeds told apart by fur, ears and muzzle, one or two objects per image |
| class imbalance (TrashNet) | `trash` has 137 images, `paper` 594 |
| domain shift | plain backgrounds and centred objects (TrashNet), pet photos from the web (Pet): the webcam is out of domain for both |
| out-of-domain baseline (Pet) | breeds map to VOC / COCO `cat` and `dog`: the pre-trained networks get a score before any fine-tuning |

**State**: 0 of 14 tasks done (2026-10-20). Document prepared from T22.11; no data
downloaded yet.

## Summary

Numbers from the dataset pages (2026-10-20); the counts after preprocessing are to be
filled by T32.1 and T32.5.

### TrashNet

| Property | Value |
|---|---|
| Source | [github.com/garythung/trashnet](https://github.com/garythung/trashnet) (`dataset-resized.zip`), mirror on Hugging Face `garythung/trashnet` |
| Licence | MIT (repository) |
| Images | 2527 JPEG, 512 × 384 (resized by the authors) |
| Classes | 6: glass 501, paper 594, cardboard 403, plastic 482, metal 410, trash 137 |
| Objects per image | 1, roughly centred, on a white or light plain background |
| Official split | none |
| Existing boxes | third-party: [OctavDM/Trashnet_annotations](https://github.com/OctavDM/Trashnet_annotations), 2521 images (6 removed), licence to check |
| Pre-trained mapping | none worth it (COCO `bottle`, `cup` cover a fraction of glass / plastic only) |

### Oxford-IIIT Pet

| Property | Value |
|---|---|
| Source | [robots.ox.ac.uk/~vgg/data/pets](https://www.robots.ox.ac.uk/~vgg/data/pets/): `images.tar.gz` (≈ 800 MB), `annotations.tar.gz` (≈ 18 MB) |
| Licence | CC BY-SA 4.0 for the annotations; the image copyright stays with the owners (photos from the web) |
| Images | 7349 JPEG, sizes vary (most about 500 × 375) |
| Classes | 37 breeds: 12 cats, 25 dogs, about 200 images each; the file name gives the breed (`Abyssinian_12.jpg`; capitalised = cat) |
| Annotations | head ROI (`annotations/xmls/`, Pascal VOC XML, trainval images only), trimap of every image (`annotations/trimaps/*.png`: 1 = pet, 2 = background, 3 = boundary; to verify at download), `list.txt`, `trainval.txt` (3680), `test.txt` (3669) |
| Objects per image | 1 most of the time; a few images show two animals or more (one mask for all) |
| Pre-trained mapping | breed → VOC / COCO `cat` or `dog` |

## Preprocessing

```bash
pip install -e "python[autolabel]"            # TrashNet only: the M22 labeller (torch)
tools/get_datasets.sh trashnet                # download + tools/prep_datasets.py trashnet
tools/get_datasets.sh pets                    # download + tools/prep_datasets.py pets
python tools/prep_datasets.py trashnet --preview build/trashnet_preview.png
python tools/prep_datasets.py pets --preview build/pets_preview.png
tools/get_datasets.sh check                   # "trashnet : prêt", "pets : prêt"
```

### TrashNet boxes (`tools/prep_datasets.py trashnet`)

- **Labeller**: `grounding-dino-tiny` through `get_labeller` of
  `tools/autolabel_models.py` (imported by the prep script only; `yolo/` never imports
  torch). CPU: about 5 s per image with 2 prompts, so ≈ 3.5 h for 2527 images, or
  ≈ 1.5 h as 3 processes of 6 threads (T22.10).
- **Prompts**: one image = one object of a known class, so the labeller only has to
  place the box. Each class folder gets its own prompt plus a generic one, e.g.
  `glass` → `glass bottle or jar`, `plastic` → `plastic bottle or container`, `metal` →
  `metal can`, `paper` → `paper`, `cardboard` → `cardboard`, `trash` → `waste`, and
  `object` for all (`TRASHNET_PROMPTS`). The prompt that wins does not matter: the
  class written is the folder's.
- **Box**: the best-scoring box of the image over all its prompts, if its score is
  ≥ `TRASHNET_CONF` (start at 0.3: one object per image, so a low threshold costs little
  precision). Sanity check: a box covering more than 98 % of the image (the labeller
  boxed the whole photo) is rejected, reason `whole image`.
- **Rejects**: no box or a rejected box → the image is left out and listed with its
  reason in `data/trashnet/rejected.json` (as ASL); never kept as a negative.
- **Split**: no official split. Stratified by class, fixed seed: 70 % train, 15 % val,
  15 % test (`TRASHNET_SPLIT`). Photos of the same object from several angles may sit
  on both sides; to check on the preview (near-duplicates).
- **Cache**: the raw predictions go to `data/trashnet/autolabel.json` (the format of
  `tools/autolabel_io.py`, SHA-1 keyed): changing the threshold or the sanity checks
  rebuilds the annotation files in seconds, without the labeller.
- **Output**: `data/trashnet/annotations_{train,val,test}.json` and
  `data/trashnet/images/<split>/` as hard links to the raw JPEGs (as ASL).

### Pet whole-body boxes (`tools/prep_datasets.py pets`)

- **Box from the trimap**: pixels 1 (pet) and 3 (boundary) form the mask; the box is
  the min / max of the mask. Made by hand, so it is the reference of this dataset; no
  labeller is needed for the main path.
- **Several animals**: the trimap holds all of them in one mask. Connected components of
  the mask (ignoring specks under 1 % of the image) → one box per component when they
  are apart. Touching animals stay one component: these images are listed
  (`multi`) and split with the M22 labeller (`cat` / `dog` boxes inside the mask), or
  left out if that fails. The class of every box is the breed of the file name (one
  breed per image).
- **Sanity checks**: the head box (when the XML exists) must lie inside the body box
  (≥ 90 % of its area), else reason `head outside`; a mask under 2 % of the image →
  reason `small`.
- **Split**: the official `test.txt` is the `test` split (evaluation); `trainval.txt` is
  split 90 / 10 into `train` / `val`, stratified by breed, fixed seed.
- **Output**: `data/pets/annotations_{train,val,test}.json` (corners, labels, head box
  when known, number of components), `data/pets/images/<split>/` as hard links.

### Known limits

- **TrashNet**: the boxes are auto-labels. A wrong box (a shadow, the background
  sheet, part of the object) is a training target with the right class. The val and
  test splits are scored against auto-labels too, until a hand-checked subset exists
  (T32.3): their mAP measures agreement with the labeller, not with a human.
- **TrashNet**: one centred object on a plain background in every image. A model trained
  on it finds waste well on a table, much less in a bin or in the street.
- **Pet**: the mask is a segmentation, not a box annotation: the box includes the
  whole silhouette, tail and legs, which is the "whole body" asked for. Leashes and
  people are background.
- **Pet**: 37 fine-grained classes at 416 px on Tiny-YOLO: expect a low breed mAP and a
  high cat / dog accuracy; report both (T32.12).

## Conventions

Those of M11, M14, M16, M18 and M21 apply. Files to touch:

| File | Addition |
|---|---|
| `python/yolo/data/datasets.py` | `TRASHNET_CLASSES`, `PETS_CLASSES` (37, file-name order), `load_trashnet`, `load_pets`, `DATASETS["trashnet"]`, `DATASETS["pets"]`, `MAPPINGS[("pets", "voc")]`, `MAPPINGS[("pets", "coco")]` |
| `tools/prep_datasets.py` | `trashnet` (labeller, best box, rejects, split, cache, preview), `pets` (trimap boxes, components, head check, split, preview) |
| `tools/get_datasets.sh` | `MARKER`, `HOWTO`, download of both (GitHub zip / HF, the two Oxford archives), `check` |
| `tools/data_stats.py` | `FICHES["trashnet"]`, `FICHES["pets"]` |
| `tools/notebooks/matrice.py` | `TRAINABLE`, `SPLIT_IMAGES`, `DATA_MARKERS`, `TASKS`, `STATS_TASKS` |
| `tools/notebooks/gabarits.py` | `QUESTIONS["trashnet"]`, `QUESTIONS["pets"]` |
| `tools/notebooks/jeux.py` | `LABELS`, `GROUND` (close-up view) |
| `tools/m11.sh` | profiles `trashnet-prep`, `trashnet-train`, `pets-prep`, `pets-train` |
| tests | `test_datasets.py::test_trashnet`, `::test_pets`, `test_prep_datasets.py` (fake labeller, synthetic trimap with two components) |

`notebooks/trashnet/`: `trashnet_stats` and the fine-tuned `_infer`, `_train`, `_sweep`
(no mapping, so no out-of-domain notebook, as ASL). `notebooks/pets/`: the same plus the
out-of-domain notebooks of the pre-trained networks, through `MAPPINGS`.

### Stats questions

| Dataset | Own question (`stats` notebook) | Task |
|---|---|---|
| trashnet | How much of the image does one piece of waste cover, and which classes lose the most images to the labeller? | T32.8 |
| pets | How much larger is the body box than the head box, and how many images hold more than one animal? | T32.8 |

---

## A. TrashNet data

### [ ] T32.0 — Download and `tools/get_datasets.sh`
- **Depends on**: T11.0 · **Size**: S
- **Deliverables**: `MARKER[trashnet]=trashnet/annotations_val.json`, `HOWTO`, a
  `trashnet` command (download of `dataset-resized.zip` into `data/TrashNet/`, then the
  prep if the marker is missing); `trashnet` in `check`.
- **Acceptance**: `tools/get_datasets.sh check` gives `trashnet : prêt` after the prep;
  the 6 folders hold 2527 images.

### [ ] T32.1 — Boxes from the M22 labeller
- **Depends on**: T32.0, T22.10 · **Size**: M
- **Deliverables**: `tools/prep_datasets.py trashnet` (prompts, best box, sanity check,
  rejects, stratified split, `autolabel.json` cache, hard links, report);
  `--preview`; tests with the fake labeller of `python/tests/test_autolabel.py`.
- **Acceptance**: ≥ 95 % of the images get a box, every class ≥ 90 %; a second run
  reuses the cache without the labeller.
- **Notes**: run as 3 processes on class folders if the CPU time hurts (T22.10). Fill
  the counts of the Summary.

### [ ] T32.2 — Loader
- **Depends on**: T32.1 · **Size**: S
- **Deliverables**: `TRASHNET_CLASSES`, `load_trashnet`, `DATASETS["trashnet"]`
  (evaluation on `test`, calibration and fine-tuning on `train`); `test_trashnet`.
- **Acceptance**: `load("trashnet")` gives the test counts of the Summary.

### [ ] T32.3 — Box check: preview and reference
- **Depends on**: T32.1 · **Size**: S
- **Deliverables**: preview (2 images per class); 100 val images checked and corrected
  by hand in the M22 editor (`make autolabel`, "Folder" tab), kept as a hand-checked
  val subset; IoU of the auto-labels against it. If the licence of
  OctavDM/Trashnet_annotations allows it, the same IoU against those 2521 boxes.
- **Acceptance**: median IoU ≥ 0.85 against the hand-checked boxes; otherwise tune the
  prompts or `TRASHNET_CONF` and run again from the cache.

## B. Oxford-IIIT Pet data

### [ ] T32.4 — Download and `tools/get_datasets.sh`
- **Depends on**: T11.0 · **Size**: S
- **Deliverables**: `MARKER[pets]=pets/annotations_val.json`, `HOWTO`, a `pets` command
  (the two archives into `data/Oxford-IIIT Pet/`, then the prep); `pets` in `check`.
- **Acceptance**: 7349 images, 7349 trimaps, `trainval.txt` and `test.txt` found; the
  trimap values (1 / 2 / 3) confirmed on a few images and written in the Summary.

### [ ] T32.5 — Whole-body boxes from the trimaps
- **Depends on**: T32.4 · **Size**: M
- **Deliverables**: `tools/prep_datasets.py pets` (mask → box, components, head check,
  split, report with the `multi`, `head outside` and `small` counts); `--preview` with the
  head box drawn in another colour; tests on a synthetic trimap (one animal, two apart,
  two touching).
- **Acceptance**: ≥ 98 % of the images get a box; on the preview, every body box
  covers its animal and contains its head box.
- **Notes**: the `multi` images that touch go through the M22 labeller only if they are
  more than about 1 %; otherwise leave them out.

### [ ] T32.6 — Loader and mappings
- **Depends on**: T32.5 · **Size**: S
- **Deliverables**: `PETS_CLASSES`, `load_pets`, `DATASETS["pets"]` (evaluation on
  `test`), `MAPPINGS[("pets", "voc")]` and `("pets", "coco")` (each breed → `cat` or
  `dog`); `test_pets`; `test_mappings_valid` passes.
- **Acceptance**: `python tools/eval_voc.py --dataset pets` runs with the VOC weights
  (out-of-domain cat / dog mAP).

### [ ] T32.7 — M22 labeller against the mask boxes
- **Depends on**: T32.5, T22.10 · **Size**: S
- **Deliverables**: `python tools/autolabel_eval.py --dataset pets --n 200` with the
  prompts `cat, dog` (breeds mapped to their species); mAP@0.5, mAP@0.75 and the IoU
  distribution of the matched boxes; a row in the T22.10 table of
  [M22](M22-autolabel.md).
- **Acceptance**: the row is written. It answers the question VOC could not: how tight
  the M22 boxes are against boxes from a human-made mask.
- **Notes**: `autolabel_eval.py` scores against the dataset classes; a species view
  (37 → 2) is needed here, through the `MAPPINGS` of T32.6.

## C. Registry and notebooks

### [ ] T32.8 — Datasheets
- **Depends on**: T32.2, T32.6 · **Size**: S
- **Deliverables**: `FICHES["trashnet"]`, `FICHES["pets"]` in `tools/data_stats.py`;
  `LABELS` and `GROUND` in `tools/notebooks/jeux.py`; the two stats questions in
  `QUESTIONS` of `tools/notebooks/gabarits.py`.

### [ ] T32.9 — Matrix, profiles, notebook generation
- **Depends on**: T32.8 · **Size**: S
- **Deliverables**: `matrice.py` entries; profiles `trashnet-prep`, `trashnet-train`,
  `pets-prep`, `pets-train` in `tools/m11.sh`; `notebooks/trashnet/`, `notebooks/pets/`,
  `notebooks/README.md` regenerated.
- **Acceptance**: `python -m tools.notebooks all --check` passes; `tools/data_stats.py`
  passes on both (`--sample 8`).

## D. Runs

### [ ] T32.10 — `stats` notebook runs
- **Depends on**: T32.9 · **Size**: S
- **Deliverables**: `notebooks/trashnet/trashnet_stats.ipynb` and
  `notebooks/pets/pets_stats.ipynb` run with their outputs; sections in
  [stats-jeux.md](stats-jeux.md).

### [ ] T32.11 — Drive upload
- **Depends on**: T32.1, T32.5 · **Size**: S
- **Deliverables**: `tools/get_datasets.sh push trashnet`, `push pets`; rows in
  [donnees-drive.md](donnees-drive.md).
- **Notes**: the archives hold the annotation files: Colab needs neither torch nor the
  trimaps.

### [ ] T32.12 — Fine-tuning and sweep
- **Depends on**: T32.10, T32.11 · **Size**: L
- **Deliverables**: `tools/m11.sh trashnet-prep` / `pets-prep` (anchors, cfg), then the
  `_sweep` and `_train` notebooks, tracked in [M15](M15-campagne-entrainement.md); mAP on
  `test`; for TrashNet also on the hand-checked subset of T32.3; for Pet also the
  species mAP (37 → 2) and the out-of-domain baseline of T32.6.
- **Notes**: horizontal flip stays on for both. TrashNet is small (≈ 1770 train
  images): start from the VOC or COCO weights, few epochs, watch `trash` (137 images).

### [ ] T32.13 — Webcam demo
- **Depends on**: T32.12, M20 · **Size**: S
- **Deliverables**: `python tools/webcam.py --model ckpt:<run>/checkpoint.npz --cfg
  build/m11/cfg/tiny-yolov3-<dataset>.cfg --dataset <dataset>` live, for each; a short
  note in [docs/webcam.md](../webcam.md) (classes read, conditions).
- **Notes**: TrashNet: hold one object in front of a plain wall first, then a cluttered
  desk. Pet: a phone screen showing pet photos is enough to try.

## Out of scope

- A segmentation head trained on the trimaps: the chain stays a detector.
- A classification head (top-1 accuracy): TrashNet and the breeds are scored as
  detection.
- Cleaning TrashNet beyond the rejects (the 6 images removed by the third-party version)
  unless T32.3 adopts that version.
- Other waste datasets with boxes (TACO and similar): a later milestone if TrashNet works.

## Recommended order

Pet first: its boxes need no labeller (T32.4 → T32.5 → T32.6), and T32.7 gives M22 its
missing tightness measurement. TrashNet next (T32.0 → T32.1 → T32.2 → T32.3), with the
threshold informed by T32.7. Then T32.8 → T32.9 for both, T32.10 and T32.11, T32.12, and
last T32.13.
