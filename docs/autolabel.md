# Auto-labelling photos (M22)

`tools/autolabel.py` pre-labels photos with an open-vocabulary detector (you type the
class names), shows the boxes, lets you fix them by hand, and exports YOLO txt labels that
the project's loaders read (`parse_yolo_txt`, `_load_yolo_dir` in
`python/yolo/data/datasets.py`). It is a labelling tool for the PC; the open-vocabulary
models never go on the FPGA. Task list and measurements: [tasks/M22-autolabel.md](tasks/M22-autolabel.md).

## 1. Install and launch

```bash
pip install -e "python[autolabel]"   # gradio, gradio_image_annotation, transformers, torch
make autolabel                       # = python tools/autolabel.py → http://127.0.0.1:7860
```

On a PC without GPU, install the CPU wheel of torch first (≈ 200 MB instead of ≈ 2 GB):
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

The first use of a labeller downloads its weights into `weights/hf/` (`HF_HOME`,
≈ 0.7 GB for Grounding DINO, ≈ 0.6 GB for OWLv2); later runs work offline. The server
listens on `127.0.0.1` only and has no share link: the photos stay on the PC.

```bash
python tools/autolabel.py --model owlv2-base     # another labeller (table below)
python tools/autolabel.py --device cpu           # force the CPU even with CUDA
python tools/autolabel.py --list                 # labellers, device, cached weights, then quit
python tools/autolabel.py --preset voc --conf 0.4 --port 7870
```

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
| `yolo:tiny-yolov2-voc`, `yolo:tiny-yolov3-coco` | project networks (`load_model` of `tools/webcam.py`) | project | fixed vocabulary: only the session classes whose name matches a model class are kept; fast |
| `yolo:ckpt:<run>/checkpoint.npz` | a fine-tuned project network | project | needs `--cfg` (and `--dataset` for the class names) |

The default export threshold depends on the labeller, as their score scales differ:
0.55 for Grounding DINO, 0.4 for OWLv2, 0.5 for the project networks (a precision of about
0.85 on VOC); `--conf` or the slider overrides it. On VOC 2007 test (200 images, CPU),
mAP@0.5 is 0.83 for Grounding DINO (6.7 s / image), 0.86 for OWLv2 (4.8 s) and 0.57 for
Tiny-YOLOv2 (0.3 s); details in T22.7 of [tasks/M22-autolabel.md](tasks/M22-autolabel.md).

## 4. Tab "Image"

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
