# M22 — Photo auto-labelling with manual correction (optional)

Goal: a local tool that pre-labels photos with an existing **open-vocabulary** detector
(the classes are typed as text), shows the boxes, lets the user fix or add boxes by hand,
and exports the labels in the YOLO txt format that the project's loaders already read
(`parse_yolo_txt`, `_load_yolo_dir` in `python/yolo/data/datasets.py`).

Two ways in:

- **Image**: drag & drop one photo → the image is shown with its boxes and classes →
  the user corrects them if needed → "Download labels" gives `<stem>.txt`.
- **Folder**: pick a folder → every image goes through the labeller → review the
  doubtful ones → "Download all" gives one label file per image, in one zip.

The long-term use is to build training sets for new classes without drawing every box by
hand, then fine-tune the embedded Tiny-YOLO on them (M15) and run it on the webcam (M20)
or the board. It generalises the M21 idea (pseudo-boxes from another model) from hands
to any class that can be named.

v1 runs on the PC (CPU, CUDA if present). The open-vocabulary model is a large PyTorch
network: it is a labelling tool, never something that goes on the FPGA.

**State**: 0 of 10 tasks done (2026-10-12). Document written; no code yet.

## User guide (target behaviour)

This section says what the finished tool does. The tasks below build it.

### 1. Launch the program

```bash
pip install -e "python[autolabel]"   # gradio, gradio_image_annotation, transformers, torch (T22.0)
make autolabel                       # = python tools/autolabel.py, opens http://127.0.0.1:7860
```

The first run downloads the model weights into `weights/hf/` (`HF_HOME`); later runs work
offline. The server listens on `127.0.0.1` only: the photos may be private, so no
`share=True` link.

Useful options:

```bash
python tools/autolabel.py --model owlv2-base         # another labeller (table below)
python tools/autolabel.py --device cpu               # force the CPU even when CUDA is present
python tools/autolabel.py --list                     # labellers, device, cached weights, then quit
```

### 2. Choose the classes

A text field takes the class list, comma-separated: `person, dog, bicycle`. The **order
defines the class ids** written in the txt files and in `classes.txt` (`person` = 0,
`dog` = 1, …). The list is fixed for a session or a folder, so the ids stay the same
across all images.

Presets fill the field so that the labels match an existing dataset of the project:

| Preset | Classes |
|---|---|
| `voc` | the 20 `VOC_CLASSES` |
| `coco` | the 80 COCO names |
| `dataset:<name>` | the names of `DATASETS[<name>]` (KITTI, VisDrone, ASL, …) |

Some dataset names are bad text prompts (`tvmonitor`, `pottedplant`, `diningtable`,
`motorbike`). An alias map sends the prompt `tv monitor` / `potted plant` / … to the
model and writes the dataset's class id (T22.7).

### 3. Choose the labeller (model)

| `--model` | Model | Licence | Notes |
|---|---|---|---|
| `grounding-dino-tiny` (default) | Grounding DINO, Swin-T (`IDEA-Research/grounding-dino-tiny`, HF transformers) | Apache-2.0 | best box quality among the light open-vocabulary models; prompt `person. dog. bicycle.` |
| `owlv2-base` | OWLv2 base (`google/owlv2-base-patch16-ensemble`, HF transformers) | Apache-2.0 | one query per class; also supports image queries ("find more like this box", T22.9) |
| `yolo:tiny-yolov2-voc`, `yolo:tiny-yolov3-coco`, `yolo:ckpt:…` | project networks, through `load_model` of `tools/webcam.py` | project | fixed vocabulary (VOC, COCO, a fine-tuned checkpoint), fast, no torch; baseline for T22.7 |

YOLO-World (ultralytics) is left out of v1: AGPL-3.0. Speed on this PC's CPU is to be
measured (T22.7); expect about 1 to 3 s per image for the open-vocabulary models against
≈ 0.2 s for Tiny-YOLO.

### 4. Tab "Image"

1. Drag & drop a photo (JPEG, PNG, WebP; EXIF orientation applied).
2. The labeller runs; the image is shown with its boxes, class names and scores, drawn
   with the colours of `tools/detect.py`.
3. A confidence slider hides the boxes under the threshold without re-running the model
   (the raw predictions are kept).
4. Manual correction: move / resize / delete a box, draw a new one, change its class
   (drop-down of the session's classes). "Re-run" applies a new prompt.
5. "Download labels" gives `<stem>.txt` (and `classes.txt`); "Download image" gives the
   annotated image.

### 5. Tab "Folder"

1. Input: several files dropped at once, a zip, or a local folder path (the server is
   local, so a path such as `~/Pictures/garden/` is fine). Option: recursive.
2. A progress bar while the labeller runs over every image; unreadable files are listed
   and skipped.
3. A gallery sorted by **"needs review"** first: images with no box, then images whose
   best score is lowest. Clicking an image opens the same editor as the "Image" tab.
4. "Download all" gives a zip:

```text
labels/<stem>.txt     one file per image (empty file = image seen, no object)
classes.txt           one class name per line, in id order
autolabel.json        provenance: model, prompt, thresholds, per-image scores, "edited" flag
```

The same batch is available without a display (T22.3):

```bash
python tools/autolabel.py --folder ~/Pictures/garden --classes "cat, dog" \
    --out build/autolabel/garden [--conf 0.3] [--model owlv2-base] [--resume] [--recursive]
```

`--resume` skips the images whose `labels/<stem>.txt` already exists, so a long folder
can be stopped and restarted.

### 6. Label format

One line per box, YOLO convention:

```text
<class id> <cx> <cy> <w> <h>
```

`cx cy w h` are normalised to `[0, 1]` by the size of the image **after** the EXIF
transpose (the size a viewer shows). This is the input of `parse_yolo_txt`, and a folder
`{images/, labels/}` is read as is by `_load_yolo_dir`, so an auto-labelled folder
becomes a `DATASETS` entry in a few lines (T22.9).

## Tasks

### [ ] T22.0 — Dependencies and `--list`
- **Spec**: — · **Depends on**: M0 · **Size**: S
- **Deliverables**: `autolabel = ["gradio>=4.44", "gradio_image_annotation",
  "transformers>=4.44", "torch", "pillow>=10"]` in the optional dependencies of
  `python/pyproject.toml`; `python tools/autolabel.py --list`.
- **Acceptance**: `--list` prints the labellers, the device (CPU / CUDA) and which
  weights are cached; it exits with a message, not a traceback, when an extra is missing.
- **Notes**: `yolo/` must never import `torch`, `transformers` or `gradio` (same rule as
  `cv2` in M20 and matplotlib in M13). Install the CPU wheel of torch on a PC without GPU
  (`--index-url https://download.pytorch.org/whl/cpu`, ≈ 200 MB instead of ≈ 2 GB).
  Pin the versions found to work in the `Done` line.

### [ ] T22.1 — Labeller interface and registry
- **Spec**: §8.1 · **Depends on**: T22.0, T20.1 · **Size**: M
- **Deliverables**: `tools/autolabel_models.py`: `Labeller.predict(pil, classes, conf)
  -> (boxes, scores, labels)` with `boxes` as normalised `(cx, cy, w, h)`, the convention
  of `detect`, so `pixel_boxes` / `draw` of `tools/detect.py` are reused; registry
  `grounding-dino-tiny`, `owlv2-base`, `yolo:<spec>` (wraps `load_model` of
  `tools/webcam.py`).
- **Acceptance**: on `000004.jpg` with the prompt `car`, every car found by
  `tiny-yolov2-voc` is matched by a Grounding DINO box with IoU ≥ 0.5.
- **Notes**: class-agnostic NMS across prompts (one object hit by `dog` and `cat` keeps
  the best score). Grounding DINO wants lower-case classes ending with `.`, and its
  prompt is capped at 256 tokens: split the 80 COCO names into several calls. Large
  photos are downscaled for inference (longest side ≤ 1333) and the boxes are written in
  normalised coordinates, so the original size is kept. Optional per-class thresholds.

### [ ] T22.2 — Label I/O
- **Spec**: — · **Depends on**: — · **Size**: S
- **Deliverables**: `tools/autolabel_io.py`: write / read YOLO txt, `classes.txt`,
  `autolabel.json`, zip of a folder result.
- **Acceptance**: write → `parse_yolo_txt` gives back the same boxes to 1e-6; an empty
  result writes an empty file (a negative image, not a missing one).
- **Notes**: `autolabel.json` keeps the **raw** predictions with their scores, separate
  from the exported labels, so the threshold can be changed later without re-running the
  model. Key the cache by image hash (SHA-1 of the bytes), not by file name.

### [ ] T22.3 — Headless folder batch (CLI)
- **Spec**: — · **Depends on**: T22.1, T22.2 · **Size**: S
- **Deliverables**: `--folder`, `--classes`, `--out`, `--conf`, `--resume`, `--recursive`
  in `tools/autolabel.py`.
- **Acceptance**: a folder of 5 images gives 5 txt files, `classes.txt` and
  `autolabel.json` on a machine without a display; a second run with `--resume` labels
  nothing.
- **Notes**: same function as the "Folder" tab (T22.6), so both give the same files.
  Names that collide in a recursive run (`a/img.jpg`, `b/img.jpg`) keep their relative
  path under `labels/`.

### [ ] T22.4 — Gradio tab "Image"
- **Spec**: — · **Depends on**: T22.1, T22.2 · **Size**: M
- **Deliverables**: Gradio app in `tools/autolabel.py`: class field and presets, model
  choice, drag & drop, annotated preview, confidence slider, download of `<stem>.txt`
  and of the annotated image.
- **Acceptance**: dropping `000004.jpg` with the prompt `car` shows the boxes, and the
  downloaded txt equals the CLI output for the same image.
- **Notes**: the model is loaded once and shared by both tabs; the slider filters the
  kept raw predictions, it does not call the model.

### [ ] T22.5 — Manual correction
- **Spec**: — · **Depends on**: T22.4 · **Size**: M
- **Deliverables**: box editor (`gradio_image_annotation`): add, move, resize, delete,
  change class; "Re-run" with a new prompt; edited images marked `edited: true` in
  `autolabel.json`.
- **Acceptance**: a box drawn by hand appears in the downloaded txt with the chosen class;
  a deleted box no longer appears.
- **Notes**: a re-run must not silently drop the manual boxes: keep them and add the new
  predictions, then NMS. If the component turns out to be too limited, the fallback is a
  small HTML canvas page served by the same process (`gr.HTML` + JS).

### [ ] T22.6 — Gradio tab "Folder"
- **Spec**: — · **Depends on**: T22.3, T22.5 · **Size**: M
- **Deliverables**: multi-file / zip / local path input, progress bar, gallery sorted by
  "needs review", per-image editor, "Download all" zip.
- **Acceptance**: a folder of 20 images gives a zip with 20 txt files, `classes.txt` and
  `autolabel.json`; a correction made in the gallery is in the zip.
- **Notes**: work on a copy under `build/autolabel/<session>/`, never write into the
  user's photo folder. Large folders: process in a generator (`yield` progress) so the
  page stays responsive.

### [ ] T22.7 — Quality and speed measurement
- **Spec**: §8.3 · **Depends on**: T22.3 · **Size**: M
- **Deliverables**: a table in this document: mAP@0.5 and seconds per image of each
  labeller on a VOC 2007 test subset (e.g. 500 images, prompt = the 20 VOC names with the
  alias map), against `tiny-yolov2-voc`; precision and recall at the chosen default `conf`.
- **Acceptance**: the table is filled for `grounding-dino-tiny`, `owlv2-base` and
  `yolo:tiny-yolov2-voc`; a default `conf` is chosen and justified.
- **Notes**: score the labels as detections with `evaluate` of
  `python/yolo/infer/metrics.py`, as `tools/eval_voc.py` does. For labelling, a
  **precision-oriented** threshold is the right default (a wrong box costs more to fix
  than a missing one costs to draw). Open-vocabulary models are zero-shot on VOC, so
  their mAP shows the labelling quality, not a training result.

### [ ] T22.8 — Tests, `make autolabel`, user guide
- **Spec**: — · **Depends on**: T22.6 · **Size**: S
- **Deliverables**: `python/tests/test_autolabel.py` (label I/O round trip, NMS across
  prompts, alias map, folder batch with a fake labeller, `--resume`); `make autolabel`
  and its `make help` line; user guide `docs/autolabel.md`.
- **Acceptance**: `make test-py` is green with and without the `autolabel` extra (tests
  that need torch or gradio are skipped without them).
- **Notes**: the fake labeller returns fixed boxes, so the tests need no weights and
  no network.

### [ ] T22.9 — Outlook: closing the loop
- **Spec**: — · **Depends on**: T22.7, M15 · **Size**: L
- **Deliverables**: plan (and a first trial) for: auto-labelled folder → `DATASETS`
  entry through `_load_yolo_dir` → M15 fine-tuning of Tiny-YOLO → integer model (M4) →
  webcam (M20) and board.
- **Acceptance**: one auto-labelled set (e.g. a few hundred photos of a new class)
  trained into a Tiny-YOLO whose mAP on a hand-checked val split is reported.
- **Notes**: other directions, by expected value: box refinement with SAM (tighter
  boxes); active learning (label first what the current Tiny-YOLO is least sure of);
  OWLv2 image queries for classes that are hard to name; COCO JSON / VOC XML export if
  another tool needs them; keyboard shortcuts in the editor for fast review.

## Design notes

- **Raw predictions vs labels**: the model output (all boxes above a low floor, with
  scores) is stored apart from the exported labels. Thresholds, class subsets and
  manual edits are applied on top, without re-running the model.
- **Stable ids**: the class list is fixed per session / folder and written to
  `classes.txt`. Changing it mid-folder starts a new session.
- **Geometry**: EXIF transpose first, then normalise to the transposed size. Downscale
  for inference only; coordinates stay normalised.
- **Negatives**: an image with no object gets an empty txt, so the loaders count it as a
  background image, as M21 does for `nothing`.
- **Privacy**: local server, no upload, no `share=True`; the models run on the PC.
- **Provenance and licence**: `autolabel.json` names the model, its version and the
  prompt. The licence of the produced labels follows that of the photos; the model's
  licence (Apache-2.0 for v1) does not restrict its outputs.
- **Pseudo-label bias**: as in M21, the errors of the labeller become training targets.
  Review at least the "needs review" images, and keep a hand-checked val split to
  measure what a model trained on auto-labels is really worth (T22.9).
