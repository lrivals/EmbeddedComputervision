"""Label I/O of the auto-labeller (M22, T22.2): YOLO txt, `classes.txt`, `autolabel.json`.

A result folder holds

    labels/<rel>.txt    one line `<class id> <cx> <cy> <w> <h>` per box, normalised by the
                        size of the image after the EXIF transpose (empty file: no object)
    classes.txt         one class name per line, in id order
    autolabel.json      provenance: model, prompt, thresholds and, per image, the raw
                        predictions (all boxes above the floor, with scores) and an
                        "edited" flag

The raw predictions are kept apart from the exported labels, so a new threshold is applied
without re-running the model; they are keyed by the SHA-1 of the image bytes.
"""

import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PROVENANCE = "autolabel.json"
CLASSES = "classes.txt"


# ---------------------------------------------------------------- images

def load_image(path):
    """RGB PIL image with its EXIF orientation applied (the size a viewer shows)."""
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def list_images(folder, recursive=False):
    """Image files of `folder` (by extension), sorted."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def label_rel_path(image, root):
    """`labels/` path of `image`: its path relative to `root`, suffix `.txt`, so that
    `a/img.jpg` and `b/img.jpg` of a recursive run do not collide."""
    return Path(image).relative_to(root).with_suffix(".txt")


# ---------------------------------------------------------------- YOLO txt

def format_yolo(boxes, labels):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return "".join(f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n"
                   for b, c in zip(np.clip(boxes, 0, 1), labels))


def write_yolo_txt(path, boxes, labels):
    """`(cx, cy, w, h)` normalised boxes → YOLO txt; no box gives an empty file (a negative
    image, not a missing one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_yolo(boxes, labels))


def read_yolo_txt(path):
    """YOLO txt → `(boxes (n, 4) normalised, labels (n,))`."""
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    rows = [r for r in rows if len(r) >= 5]
    boxes = np.array([[float(v) for v in r[1:5]] for r in rows], dtype=np.float64).reshape(-1, 4)
    return boxes, np.array([int(r[0]) for r in rows], dtype=np.int64)


def write_classes(path, classes):
    Path(path).write_text("".join(f"{c}\n" for c in classes))


def read_classes(path):
    return [c.strip() for c in Path(path).read_text().splitlines() if c.strip()]


# ---------------------------------------------------------------- raw predictions

def raw_record(boxes, scores, labels):
    """Raw predictions → JSON-able dict."""
    return {"boxes": np.round(np.asarray(boxes, dtype=np.float64).reshape(-1, 4), 6).tolist(),
            "scores": np.round(np.asarray(scores, dtype=np.float64), 4).tolist(),
            "labels": [int(c) for c in labels]}


# ---------------------------------------------------------------- provenance

def new_provenance(model, version, classes, prompts, conf):
    return {"model": model, "model_version": version, "classes": list(classes),
            "prompts": list(prompts), "conf": conf,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "images": {}, "unreadable": []}


def read_provenance(out_dir):
    path = Path(out_dir) / PROVENANCE
    return json.loads(path.read_text()) if path.exists() else None


def write_provenance(out_dir, prov):
    path = Path(out_dir) / PROVENANCE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prov, indent=1))
    tmp.replace(path)  # a run stopped mid-write leaves the previous file


def image_record(digest, size, raw):
    """Provenance of one image: raw predictions plus the manual corrections, kept apart —
    `removed` (indices of raw boxes deleted or changed by hand) and `manual` (boxes drawn
    or changed by hand, `[cx, cy, w, h, class]`)."""
    return {"sha1": digest, "width": size[0], "height": size[1], "raw": raw,
            "removed": [], "manual": [], "edited": False, "n_boxes": 0, "best_score": 0.0}


def labels_of(record, conf):
    """Exported labels of an image: raw predictions above `conf`, minus the removed ones,
    plus the manual boxes (score 1). Returns `(boxes, scores, labels, raw indices)`, the
    index being -1 for a manual box."""
    raw = record["raw"]
    scores = np.asarray(raw["scores"], dtype=np.float64)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    thr = np.asarray(conf, dtype=np.float64)[labels] if np.ndim(conf) else conf
    keep = scores >= thr
    keep[np.asarray(record.get("removed", []), dtype=np.int64)] = False
    idx = np.nonzero(keep)[0]
    manual = np.asarray(record.get("manual", []), dtype=np.float64).reshape(-1, 5)
    boxes = np.concatenate([np.asarray(raw["boxes"], dtype=np.float64).reshape(-1, 4)[idx],
                            manual[:, :4]])
    return (boxes, np.concatenate([scores[idx], np.ones(len(manual))]),
            np.concatenate([labels[idx], manual[:, 4].astype(np.int64)]),
            np.concatenate([idx, np.full(len(manual), -1)]).astype(np.int64))


def export_labels(record, conf, path):
    """Write the labels of `record` at `conf` to `path`; updates its review fields."""
    boxes, scores, labels, _ = labels_of(record, conf)
    write_yolo_txt(path, boxes, labels)
    record["n_boxes"] = len(labels)
    record["best_score"] = round(float(scores.max()), 4) if len(scores) else 0.0
    return boxes, scores, labels


def review_order(images):
    """Keys of `images` (provenance records), "needs review" first: no box, then lowest
    best score."""
    return sorted(images, key=lambda k: (images[k]["n_boxes"] > 0, images[k]["best_score"], k))


def zip_result(out_dir, dest=None):
    """`labels/`, `classes.txt` and `autolabel.json` of `out_dir` → one zip."""
    out_dir = Path(out_dir)
    dest = Path(dest) if dest else out_dir.with_suffix(".zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name in (CLASSES, PROVENANCE):
            if (out_dir / name).exists():
                z.write(out_dir / name, name)
        for p in sorted((out_dir / "labels").rglob("*.txt")):
            z.write(p, p.relative_to(out_dir).as_posix())
    return dest
