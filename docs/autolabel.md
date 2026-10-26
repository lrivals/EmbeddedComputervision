# Auto-labelling photos (M22)

`tools/autolabel.py` pre-labels photos with an open-vocabulary detector (you type the
class names), shows the boxes, lets you fix them by hand, and exports YOLO txt labels that
the project's loaders read (`parse_yolo_txt`, `_load_yolo_dir` in
`python/yolo/data/datasets.py`). It is a labelling tool for the PC; the open-vocabulary
models never go on the FPGA. Task list and measurements: [tasks/M22-autolabel.md](tasks/M22-autolabel.md).

## Quick start

```bash
pip install -e "python[autolabel]"   # once: gradio, gradio_image_annotation, transformers, torch
make get-weights                     # once, only for the yolo:* labellers (Darknet weights)
make autolabel                       # starts the app; open http://127.0.0.1:7860 in a browser
```

Stop the app with `Ctrl+C` in its terminal. First session in the browser: type the
classes (`cat, dog`), drop a photo in the **Image** tab, fix the boxes, click
**Download labels**.

## 1. Install and launch

On a PC without GPU, install the CPU wheel of torch before the extra (≈ 200 MB instead of
≈ 2 GB): `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Versions
known to work: torch 2.11, transformers 5.19, gradio 6.30, gradio_image_annotation 0.5.6.

The first use of a labeller downloads its weights into `weights/hf/` (`HF_HOME`,
≈ 0.7 GB for Grounding DINO, ≈ 0.6 GB for OWLv2); later runs work offline. The model is
loaded at the first photo (a few seconds), then shared by both tabs. The server listens on
`127.0.0.1` only and has no share link: the photos stay on the PC.

```bash
python tools/autolabel.py                        # = make autolabel
python tools/autolabel.py --model owlv2-base     # another default labeller (table below)
python tools/autolabel.py --device cpu           # force the CPU even with CUDA
python tools/autolabel.py --preset voc           # class field pre-filled with the 20 VOC names
python tools/autolabel.py --classes "cat, dog" --conf 0.4 --port 7870
python tools/autolabel.py --list                 # labellers, device, cached weights, then quit
```

| Option | Default | Effect |
|---|---|---|
| `--model` | `grounding-dino-tiny` | labeller selected at start (the app has a menu too) |
| `--device` | `auto` | `cpu` or `cuda`; `auto` takes CUDA when present |
| `--classes` / `--preset` | `person, dog` | initial class list (`voc`, `coco`, `dataset:<name>`) |
| `--conf` | by labeller (§3) | export threshold, the slider's start value |
| `--port` | 7860 | port of the local server |
| `--folder`, `--out`, `--resume`, `--recursive` | | batch without a display (§5) |
| `--cfg`, `--dataset` | | config and class names of a `yolo:ckpt:` labeller |

## 2. Classes

The **Classes** field takes a comma-separated list: `person, dog, bicycle`. The **order
gives the class ids** of the txt files and of `classes.txt` (`person` = 0, `dog` = 1, …).
Keep the list fixed for a folder so the ids stay the same across images.

`name=prompt` writes `name` but sends `prompt` to the model: `tvmonitor=television`.
Without `=`, a built-in alias map turns bad prompts into good ones (`tvmonitor` → `tv
monitor`, `pottedplant` → `potted plant`, `diningtable` → `dining table`, `motorbike` →
`motorcycle`, `aeroplane` → `airplane`, `_` and `-` → spaces).

The **Preset** menu fills the field with the classes of a project dataset, so the labels
match it: `voc` (20), `coco` (80), `dataset:<name>` (KITTI, VisDrone, ASL, …).

## 3. Labellers

| `--model` | Model | Licence | Notes |
|---|---|---|---|
| `grounding-dino-tiny` (default) | Grounding DINO Swin-T, `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | best boxes; prompt `person. dog.`, split in several calls past 256 tokens |
| `owlv2-base` | OWLv2 base, `google/owlv2-base-patch16-ensemble` | Apache-2.0 | one query per class; lower scores than Grounding DINO for the same quality |
| `yolo:tiny-yolov2-voc`, `yolo:tiny-yolov3-coco`, `yolo:tiny-yolov4-coco` | project networks (`load_model` of `tools/webcam.py`) | project | fixed vocabulary: only the session classes whose name matches a model class are kept; fast |
| `yolo:ckpt:<run>/checkpoint.npz` | a fine-tuned project network | project | needs `--cfg` (and `--dataset` for the class names) |

The default export threshold depends on the labeller, as their score scales differ:
0.55 for Grounding DINO, 0.4 for OWLv2, 0.5 for the project networks (a precision of about
0.85 on VOC); `--conf` or the slider overrides it. On VOC 2007 test (200 images, CPU),
mAP@0.5 is 0.83 for Grounding DINO (6.7 s / image), 0.86 for OWLv2 (4.8 s) and 0.57 for
Tiny-YOLOv2 (0.3 s); details in T22.7 of [tasks/M22-autolabel.md](tasks/M22-autolabel.md).

## 4. Tab "Image"

The editor fits each photo to the window (width and height, portrait or landscape), so
the whole image and its toolbar are visible without scrolling, down to a 1000 × 600
window. In the "Folder" tab, a click in the gallery scrolls the editor into view.

1. Type the classes, pick the labeller.
2. Drop a photo (JPEG, PNG, WebP). Its EXIF orientation is applied first.
3. The boxes appear with their class. The **Confidence** slider hides or shows boxes
   without re-running the model: every prediction down to 0.05 is kept.
4. Correct by hand in the editor: drag on the image to draw a box, click a box to select
   it, move or resize it with the mouse, `Delete` removes it, double-click changes its
   class. Hand-drawn and changed boxes are kept whatever the slider says.
5. **Re-run** applies a new class list or labeller. Your hand boxes stay; a new prediction
   that overlaps one of them, or a box of the same class you deleted, is dropped.
6. **Download labels** gives `<stem>.txt`, `classes.txt` and `autolabel.json`;
   **Download image** gives the photo with the boxes drawn.

## 5. Tab "Folder"

1. Drop several images or a zip, or type a local folder path (`~/Pictures/garden`); tick
   **Recursive** for sub-folders. Everything is copied under
   `build/autolabel/<session>/images/`: your folder is never written to.
2. **Label all** runs the labeller over every image with a progress bar. Unreadable files
   are listed and skipped.
3. The gallery shows **"needs review" first**: images without a box, then the lowest best
   score. Click an image to open it in the editor below; each edit is saved at once.
4. **Download all** gives a zip, at the threshold of the slider (your hand edits are kept):

```text
labels/<rel>.txt      one file per image (empty file = image seen, no object)
classes.txt           one class name per line, in id order
autolabel.json        provenance: model and version, prompts, threshold, and per image the
                      raw predictions with scores, the hand corrections, the "edited" flag
```

The same batch without a display:

```bash
python tools/autolabel.py --folder ~/Pictures/garden --classes "cat, dog" \
    --out build/autolabel/garden [--conf 0.5] [--model owlv2-base] [--resume] [--recursive]
```

`--resume` skips the images whose txt already exists, so a long folder can be stopped and
restarted (it needs the same classes and model). An image already in `autolabel.json`
with the same bytes (SHA-1) reuses its raw predictions without running the model.
`<rel>` is the path relative to the folder, so `a/img.jpg` and `b/img.jpg` do not collide.

## 6. Label format, use in training

One line per box, YOLO convention: `<class id> <cx> <cy> <w> <h>`, normalised to `[0, 1]`
by the size of the image **after** the EXIF transpose (the size a viewer shows). A folder
`{images/, labels/}` is read by `_load_yolo_dir` as is, so an auto-labelled set becomes a
`DATASETS` entry in a few lines (T22.9).

Labeller errors become training targets: review at least the "needs review" images, and
keep a hand-checked val split to measure what a model trained on auto-labels is worth.

**Industrial footage.** For factory CCTV (`person` only), the labeller and threshold come
from T41.2 ([tasks/M41-safe-unsafe-behaviours.md](tasks/M41-safe-unsafe-behaviours.md)),
measured on PREWID, whose boxes are hand-made; until then use the ground-level default
(`grounding-dino-tiny`, 0.55). Video clips are cut into frames first, with the sampling
rules of T41.1.

## 7. Troubleshooting

| Symptom | Cause, fix |
|---|---|
| `gradio, gradio_image_annotation missing: pip install …` | the extra is not installed: `pip install -e "python[autolabel]"` |
| `Cannot find empty port in range: 7860-7860` | another app (or a previous run) uses it: `--port 7870`, or stop the other run |
| first photo takes a long time | weights downloading (first use) or model loading; later photos ≈ 5 s on CPU |
| no box at all | threshold too high for this labeller or class name not understood: lower the slider, try `name=other words`, or another labeller |
| boxes around groups of objects | a Grounding DINO habit on crowded scenes: delete the group box, or try `owlv2-base` |
| `--resume needs the same …` | the folder was labelled with another class list or labeller: same options, or a new `--out` |
| `yolo:*` labeller unavailable | Darknet weights missing: `make get-weights` |
| work files | `build/autolabel/image/` (Image tab), `build/autolabel/<session>/` (Folder tab: `images/` copies, `result/`, `labels.zip`); safe to delete |
