# M18 — Drone and thermal datasets: AU-AIR, DroneVehicle, HIT-UAV, UAVDT (optional)

Goal: plug four datasets seen from a drone into the chain of [M11](M11-jeux-de-donnees.md),
like VisDrone, FLIR, ExDark and CrowdHuman. For each one: a loader in `DATASETS`, a
datasheet, the notebooks of [M14](M14-notebooks.md) and [M16](M16-presentation-jeux.md)
(statistics, out-of-domain inference, fine-tuning), and the archive on the Drive. This
document mostly says **what preprocessing each dataset needs** before entering the chain,
and why.

Each dataset stresses a part of the chain that VisDrone only partly covers:

| Dataset | What it stresses |
|---|---|
| auair | 1920×1080 frames reduced to 416: tiny objects, class imbalance (Car ≈ 78 %) |
| dronevehicle | oriented boxes turned into upright boxes, dense vehicle scenes, night in RGB |
| hituav | single-channel thermal seen from 60 to 130 m (L00 at cin = 1, like FLIR), very small objects |
| uavdt | non-square 1024×540, very redundant video frames, objects among the smallest of the datasets |

**State**: 8 tasks done out of 11. Data ready on the PC (`tools/get_datasets.sh check`),
loaders, preprocessing, datasheets and 24 notebooks generated. Remaining: running the
`_stats` notebooks (T18.8) and the upload to the Drive (T18.9). The fine-tunings are
tracked in M15 (T18.10).

## Summary

The counts come from the loaders (`tools/data_stats.py`, revision `16c9b6f`). The values
"seen by the network" are taken at 416×416 in letterbox, with the COCO anchors, before the
anchors specific to each dataset (`tools/m11.sh <dataset>-prep`).

| Dataset | Images (train / test) | Objects | Classes | Resolution | Objects/image (mean) | Small < 32² at 416 | Licence |
|---|---|---|---|---|---|---|---|
| auair | 27,880 / 4,943 (val) | 131,977 | 8 | 1920×1080 | 4.0 | 79.8 % | CC BY-NC-SA 2.0 |
| dronevehicle | 12,118 / 2,608 (+ 2,599 val) | 278,078 | 2 | 640×512 (after crop) | 16.0 | 65.4 % | CC BY-NC-SA 4.0 |
| hituav | 2,008 / 571 (+ 287 val) | 24,751 | 4 | 640×512, 1 channel | 8.6 | 84.2 % | to check |
| uavdt | 24,143 / 16,592 | 798,795 | 3 | 1024×540 | 19.6 | 96.9 % | research only |

Mappings to the pre-trained weights (`MAPPINGS`, out-of-domain):

| Dataset | → VOC | → COCO, in addition | Without an equivalent |
|---|---|---|---|
| auair | Human→person, Car and Van→car, Motorbike, Bicycle, Bus | Truck→truck | Trailer |
| dronevehicle | small-vehicle→car | large-vehicle→truck (buses and trucks merged) | — |
| hituav | Person, Car, Bicycle | — | OtherVehicle |
| uavdt | car, bus | truck | — |

<a id="prétraitements"></a>

## Preprocessing

Rule: the loader reads the raw format when it is fast. Preprocessing on disk only happens
if the image itself must change (DroneVehicle) or if reading the raw format is too slow
(UAVDT). The raw folders keep their original name in `data/`.
`tools/get_datasets.sh kaggle` links (or preprocesses) them to `data/<dataset>`, the only
path the code reads.

```bash
tools/get_datasets.sh kaggle    # links auair and hituav; prep_datasets.py dronevehicle uavdt
tools/get_datasets.sh check     # 4 datasets "prêt"
```

### AU-AIR — no preprocessing on disk

- **Raw**: `data/AU-AIR dataset/AU-AIR dataset/{annotations.json,images/}`, linked as
  `data/auair`.
- **Format**: a single JSON, one entry per frame (`image_name`, `bbox` in
  `top/left/width/height` pixels, `class` index into `categories`), with the altitude, the
  attitude and the date of each frame.
- **Pitfalls of the JSON**, tolerated by `load_auair`:
  - the width key is spelled `"image_width:"` (colon included);
  - 5,972 frames are named `…_xx_…` instead of `…_x_…` (the files exist under that name).
- **Split**: the dataset has no official one, and its 32,823 frames come from 8 videos. A
  random draw would put neighbouring frames in train and in val. The split is therefore
  done **by video**: `AUAIR_VAL` gathers 3 whole videos (4,943 frames, 15 %), the rest goes
  to train. `val` serves as the evaluation split.
- **Removals**: 54 boxes empty after clipping (`make_sample`).
- Val only has 5 Motorbike and 66 Bus: the AP of these classes is very noisy there.

### DroneVehicle — crop and bounding boxes (`tools/prep_datasets.py dronevehicle`)

- **Raw**: `data/DroneVehicle Dataset/DroneVehiclesDatasetYOLO/{train,val,test}/{images,labels}`.
  It is the Roboflow "YOLOv11-OBB" version (M. Mandal). It only keeps the RGB half of the
  original dataset (not the infrared) and re-splits the images of the original train (no.
  00001 to 17990) into 12,118 / 2,599 / 2,608. The classes are merged into small-vehicle
  (car, van) and large-vehicle (bus, truck, freight car).
- **White frame**: each image is 840×712, i.e. the 640×512 image surrounded by a 100 px
  white frame (checked on a sample). Left in place, this frame would take up 45 % of the
  input pixels. The preprocessing **crops** it and rewrites the image at 640×512 (JPEG
  quality 95).
- **Oriented boxes**: a line is `class x1 y1 … x4 y4`, normalised on 840×712. The YOLO
  head only produces upright boxes ("Out of scope" of M11). Each box therefore becomes its
  **bounding box**: min/max of the 4 corners, shifted by 100 px, clipped to the image then
  renormalised. A bounding box is larger than the oriented box for a vehicle at an angle,
  hence stronger overlaps between neighbours.
- **Output**: `data/dronevehicle/<split>/{images,labels}/<no>.{jpg,txt}` in YOLO
  `class cx cy w h`. The name loses the Roboflow suffix `_jpg.rf.<hash>`. 2.2 GB,
  ≈ 20 s on 16 cores, and the script is idempotent.
- `--keep-border` keeps the 840×712 images as they are and only converts the boxes.
- The dataset contains RGB night images, almost black: a lead for the INT8 calibration,
  like ExDark.

### HIT-UAV — no preprocessing on disk

- **Raw**: `data/HIT-UAV: A High-altitude Infrared Thermal Dataset/hit-uav/{images,labels}/{train,val,test}`,
  linked as `data/hituav`. The original `dataset.yaml` points to `/tmp/dataset` and is not
  read.
- **Format**: YOLO txt (normalised `class cx cy w h`), 640×512 JPEG images in mode `L`
  (single channel), read by `parse_yolo_txt`.
- **DontCare** (class 4, 148 boxes) is removed by the loader, as in KITTI. Person, Car,
  Bicycle and OtherVehicle remain (148 objects, almost empty).
- **Single channel**: the fine-tuning runs with `CH=1` (cfg `tiny-yolov3-hituav-c1.cfg`,
  L00 copied from the RGB weights by summing over the channels), like FLIR (T11.7). The
  pre-trained weights receive the grey image copied onto 3 channels (`letterbox`,
  `image_mode`).
- The file name encodes the capture conditions: `<period>_<altitude m>_<angle
  °>_…_<no>`. The altitude goes from 60 to 130 m and the angle from 30 to 90°. The first
  field separates two periods (917 images in 0, 1,949 in 1): day or night, meaning to be
  confirmed on the dataset page.

### UAVDT — M sequences only and compact JSON (`tools/prep_datasets.py uavdt`)

- **Raw**: `data/UAVDT Dataset/{train,test}/{img,ann,meta}`. It is the Supervisely export
  of DatasetNinja: one JSON per image, with `points.exterior` boxes (bottom-right corner
  included), object tags (occlusion, out of view, target identifier) and image tags (day,
  night, fog, altitude, view).
- **S sequences set aside**: the test set of the export mixes the 20 M sequences of the DET
  benchmark (16,592 frames) and the S sequences of single-object tracking (37,084 frames,
  a single box of class `vehicle`). The latter would skew the mAP and are not kept. The
  train set only contains M sequences (30 sequences, 24,143 frames).
- **Reading**: rereading 40,000 JSON files at each load would take several minutes. The
  preprocessing writes `data/uavdt/annotations_{train,test}.json`, a compact list (image,
  size, continuous corners, classes, sequence, image tags, occlusion and out-of-view per
  object). It also creates `data/uavdt/images/{train,test}/` as **hard links** to the M
  images, without a copy, which keeps the S sequences out of the Drive archive.
- **Ignored regions**: the original dataset has them (`*_gt_ignore.txt`), the export does
  not. A detection in an ignored area therefore counts as a false alarm, as for VisDrone
  before T15.10.
- **Redundancy**: consecutive frames are almost identical (≈ 800 per sequence). An epoch
  over 24,143 images is worth much less than 24,143 independent images, which must be kept
  in mind for the batch × subset sweep of M15.
- **Train ↔ test gap**: the test objects are smaller (total variation of the area at the
  input 0.32, the largest of the four datasets).

## Shared conventions

Those of M11 (loader, `MAPPINGS`), M14 (generated notebooks, `--check`) and M16
(datasheet, `stats`) apply unchanged. Adding a dataset means touching these files:

| File | Addition |
|---|---|
| `python/yolo/data/datasets.py` | classes, loader, `DATASETS` entry, `MAPPINGS` to VOC and COCO |
| `tools/prep_datasets.py` | preprocessing on disk, if any |
| `tools/get_datasets.sh` | `MARKER`, `HOWTO`, link or preprocessing in `m18()`, list of `check` |
| `tools/data_stats.py` | `FICHES`; `raw_annotations` if the loader removes boxes |
| `tools/notebooks/matrice.py` | `TRAINABLE`, `TRAIN_INPUT` (1 channel), `SPLIT_IMAGES`, `DATA_MARKERS`, `REGISTRATION`, `TASKS`, `STATS_TASKS`, `READS_HEADERS` |
| `tools/notebooks/gabarits.py` | specific question of the `stats` notebook (`QUESTIONS`) |
| `tools/m11.sh` | profiles `<dataset>-prep` and `<dataset>-train` |
| tests | `test_datasets.py`, `test_prep_datasets.py` |

then `python -m tools.notebooks <dataset>` (or `make notebooks`).

<a id="questions-par-jeu"></a>

### Questions per dataset

| Dataset | Specific question (`stats` notebook) | Pointer |
|---|---|---|
| auair | what is left of the objects of a 1920×1080 frame reduced to 416, class by class? | T18.1 |
| dronevehicle | vehicles per image against the 256 slots, and their size at 416 | T18.2 |
| hituav | distribution of the thermal levels, and what the INT8 of L00 keeps of it | T18.3 |
| uavdt | `letterbox` or `stretch` for 1024×540: how many objects fall under 8 px of height? | T18.4 |

---

## A. Data

### [x] T18.0 — Trees and `tools/get_datasets.sh`
- **Depends on**: T11.0 · **Size**: S
- **Deliverables**: `MARKER` and `HOWTO` of the four datasets; function `m18()` called by
  `kaggle` (links `auair` and `hituav`, `prep_datasets.py` for `dronevehicle` and `uavdt`
  if their marker file is missing); the four datasets in `check`.
- **Acceptance**: `tools/get_datasets.sh check` gives the four datasets "prêt"; `pack`
  archives `data/<dataset>` with links followed.
- **Notes**: no `KAGGLE_ID` until the identifiers of the mirrors are noted. On Colab, the
  way in is therefore the Drive archive (T18.9).

### [x] T18.1 — AU-AIR
- **Depends on**: T18.0 · **Size**: S
- **Deliverables**: `AUAIR_CLASSES`, `load_auair`, `AUAIR_VAL` (split by video);
  `test_auair`.
- **Acceptance**: 27,880 + 4,943 images, 131,977 objects (132,031 raw, minus the 54 empty
  boxes found by `raw_annotations`).

### [x] T18.2 — DroneVehicle
- **Depends on**: T18.0 · **Size**: M
- **Deliverables**: `tools/prep_datasets.py dronevehicle` (crop, OBB → bounding box);
  `parse_yolo_txt`, `load_dronevehicle`; tests `test_obb_to_yolo_border`,
  `test_prep_dronevehicle`, `test_dronevehicle`.
- **Acceptance**: 17,325 images at 640×512, 278,078 objects (as many as raw lines); boxes
  checked by eye on day and night images.

### [x] T18.3 — HIT-UAV
- **Depends on**: T18.0 · **Size**: S
- **Deliverables**: `HITUAV_CLASSES` (without DontCare), `load_hituav`;
  `test_yolo_txt_and_hituav`.
- **Acceptance**: 2,008 / 287 / 571 images, 24,751 objects (DontCare removed); images read
  as a single channel by the `CH=1` fine-tuning.

### [x] T18.4 — UAVDT
- **Depends on**: T18.0 · **Size**: M
- **Deliverables**: `tools/prep_datasets.py uavdt` (M sequences, compact JSON, hard
  links); `load_uavdt`; `test_prep_uavdt`, `test_uavdt`.
- **Acceptance**: train 24,143 images, 422,911 objects; test 16,592 images, 375,884
  objects; 37,084 S frames set aside.

## B. Registry and notebooks

### [x] T18.5 — Datasheets and mappings
- **Depends on**: T18.1 to T18.4 · **Size**: S
- **Deliverables**: four `FICHES` entries of `tools/data_stats.py`; `raw_annotations` for
  auair (raw boxes) and hituav (DontCare); `MAPPINGS` to VOC and COCO;
  `test_m18_mappings`.
- **Remaining**: licence of HIT-UAV to note from the dataset page.

### [x] T18.6 — Matrix and profiles
- **Depends on**: T18.5 · **Size**: S
- **Deliverables**: `matrice.py` (the four datasets in `TRAINABLE`, hituav with 1
  channel, `SPLIT_IMAGES`, marker files, tasks); `QUESTIONS` of `gabarits.py`; profiles
  `<dataset>-prep` and `<dataset>-train` of `tools/m11.sh` (`CH=1` for hituav).

### [x] T18.7 — Generating the notebooks
- **Depends on**: T18.6 · **Size**: S
- **Deliverables**: `notebooks/{auair,dronevehicle,hituav,uavdt}/`, 6 notebooks per
  dataset: `<dataset>_stats`, `tiny-yolov2-voc_infer`, `tiny-yolov3-coco_infer`,
  `tiny-yolov3-<dataset>_{infer,train,sweep}`; `notebooks/README.md` regenerated;
  `analyse_balayages.ipynb` rerun (its list of sweeps changed).
- **Acceptance**: `python -m tools.notebooks all --check` passes for these notebooks.
  `data_stats.py` passes on the four datasets (`--sample 8`), and so does `eval_voc.py`
  (COCO weights, 20 images, as a simple trial of the chain).

## C. Running

### [x] T18.8 — `stats` notebooks executed
- **Depends on**: T18.7 · **Size**: S
- **Deliverables**: the four `<dataset>_stats.ipynb` executed and versioned with their
  outputs; sections in [stats-jeux.md](stats-jeux.md) (`tools/data_stats.py --report`).
- **Notes**: redo after `tools/m11.sh <dataset>-prep`, so that the collisions use the
  anchors of the dataset and not those of COCO.
- **Done** (2026-10-27): the four notebooks regenerated (their template was stale since
  M26) and executed with the cfgs of `<dataset>-prep` (`build/m11/cfg/`): auair 69 s,
  dronevehicle 60 s, hituav 13 s, uavdt 151 s. Sections in [stats-jeux.md](stats-jeux.md);
  the split counts equal the official ones. Small boxes at 416: auair 79.8 %, dronevehicle
  65.4 %, hituav 84.2 %, uavdt 96.9 %; lost targets (collisions) 8.9 %, 2.9 %, 7.8 %,
  14.7 % (uavdt test 20.8 %).

### [x] T18.9 — Upload to the Drive
- **Depends on**: T18.0 · **Size**: S
- **Deliverables**: `tools/get_datasets.sh push auair dronevehicle hituav uavdt`; rows of
  the table of [donnees-drive.md](donnees-drive.md).
- **Notes**: ≈ 2.4 + 2.2 + 0.2 + 6.5 GB (`du -shL`). UAVDT is the longest to upload.
  Check the free space (`rclone about gdrive:`).
- **Status (2026-10-08)**: done. The four archives are on `gdrive:EmbeddedCV/data`, with
  the same sizes as in `data_archives/` (auair 2,477,015,040; dronevehicle 2,244,229,120;
  hituav 203,950,080; uavdt 6,855,751,680 bytes). The Drive has 32.7 GiB free.
  `data_archives/` was then emptied; `tools/get_datasets.sh pull <dataset>` restores a
  dataset.

### [ ] T18.10 — Out of domain and fine-tunings
- **Depends on**: T18.8, T18.9 · **Size**: L
- **Deliverables**: mAP of the VOC and COCO weights without fine-tuning (`_infer`
  notebooks, as in T11.2); sweeps and fine-tuned runs (`_sweep`, `_train`), tracked in
  [M15](M15-campagne-entrainement.md).
- **Notes**:
  - HIT-UAV and UAVDT have 84 % and 97 % of objects smaller than 32² at 416: they are the
    best quantified arguments for a larger input or tiles (T15.9, T15.16).
  - On UAVDT, subsample the frames, or rely on the subset of the sweep, rather than on the
    epochs.

## Out of scope

- Oriented boxes of DroneVehicle: the head does not produce them (like DOTA in M11).
- Infrared half of DroneVehicle: missing from this version. If it arrives, it will form a
  dataset of its own, single-channel like HIT-UAV.
- Object tracking (S sequences of UAVDT, target identifiers).

## Suggested order

T18.0 → T18.1 to T18.4 → T18.5 → T18.6 → T18.7 and T18.9 (Drive, for Colab) are done.
Then T18.8, then T18.10.
