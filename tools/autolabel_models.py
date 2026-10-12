"""Labellers of the auto-labelling tool (M22, T22.1): image + class names → boxes.

    labeller = get_labeller("grounding-dino-tiny")         # or owlv2-base, yolo:<spec>
    boxes, scores, labels = labeller.predict(pil, ["person", "dog"], conf=0.05)

`boxes` are normalised `(cx, cy, w, h)` in `pil`, the convention of `detect`, so
`pixel_boxes` / `draw` of tools/detect.py apply; `labels` index the class list. A class is
written `name` or `name=prompt` (the prompt sent to the model, see `ALIASES`).

torch and transformers are imported here only, lazily, never by `yolo/` (as cv2 in M20).
The Hugging Face weights go to `weights/hf/` (`HF_HOME`).
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("HF_HOME", str(ROOT / "weights" / "hf"))

import numpy as np  # noqa: E402

from yolo.infer.nms import nms  # noqa: E402

EXTRA_HINT = 'pip install -e "python[autolabel]"'
FLOOR = 0.05      # raw predictions kept down to this score (thresholds applied later)
AGNOSTIC_IOU = 0.7  # one object hit by two prompts ("dog", "cat") keeps the best score
MAX_SIDE = 1333   # inference only: larger photos are downscaled, boxes stay normalised
GDINO_MAX_TOKENS = 256

# Dataset class names that are bad text prompts → prompt sent to the model (T22.7).
ALIASES = {
    "aeroplane": "airplane", "diningtable": "dining table", "motorbike": "motorcycle",
    "pottedplant": "potted plant", "tvmonitor": "tv monitor", "person_sitting": "sitting person",
    "othervehicle": "vehicle", "small-vehicle": "car", "large-vehicle": "truck",
}


class MissingExtra(RuntimeError):
    pass


# ---------------------------------------------------------------- classes and prompts

def parse_classes(text):
    """`"person, tvmonitor=tv monitor"` → (names, prompts). Names give the class ids (their
    order) and `classes.txt`; prompts are sent to the model."""
    names, prompts = [], []
    for item in re.split(r"[,\n]", text):
        name, _, prompt = (s.strip() for s in item.partition("="))
        if not name:
            continue
        if name in names:
            raise ValueError(f"class {name!r} given twice")
        names.append(name)
        prompts.append(prompt or prompt_of(name))
    return names, prompts


def prompt_of(name):
    """Text prompt of a class name: alias, else lower case with `_` / `-` as spaces."""
    alias = ALIASES.get(name, ALIASES.get(name.lower()))
    return alias or re.sub(r"[_-]+", " ", name).strip().lower()


def same_class(a, b):
    """Two class names that denote the same thing (`tvmonitor` and `tv monitor`)."""
    return prompt_of(a).replace(" ", "") == prompt_of(b).replace(" ", "")


def chunk_prompts(prompts, n_tokens, budget):
    """Split `prompts` into consecutive groups whose token counts (`n_tokens[i]` + 1 for the
    `.`) fit in `budget`; returns lists of indices."""
    chunks, cur, used = [], [], 0
    for i, n in enumerate(n_tokens):
        if cur and used + n + 1 > budget:
            chunks.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += n + 1
    return chunks + ([cur] if cur else [])


# ---------------------------------------------------------------- boxes

def agnostic_nms(boxes, scores, labels, iou_thr=AGNOSTIC_IOU):
    """NMS over all classes together, by score; returns the kept `(boxes, scores, labels)`."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep], labels[keep]


def xyxy_px_to_cxcywh(xyxy, w, h):
    """Pixel corners in a w × h image → normalised `(cx, cy, w, h)`, clipped to the image."""
    b = np.clip(np.asarray(xyxy, dtype=np.float64).reshape(-1, 4) / [w, h, w, h], 0, 1)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def downscale(img, max_side=MAX_SIDE):
    if max(img.size) <= max_side:
        return img
    s = max_side / max(img.size)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))


def _finish(boxes, scores, labels, conf):
    """Score floor, drop empty boxes, agnostic NMS, sort by score."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    keep = (scores >= conf) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    return agnostic_nms(boxes[keep], scores[keep], labels[keep])


# ---------------------------------------------------------------- labellers

class Labeller:
    name = "?"
    licence = "?"
    version = None

    def predict(self, img, classes, conf=FLOOR):
        """PIL RGB image, class names (or `(names, prompts)`) → normalised
        `(boxes, scores, labels)` above `conf`, after the agnostic NMS."""
        names, prompts = classes if isinstance(classes, tuple) else (
            list(classes), [prompt_of(c) for c in classes])
        if not names:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
        return _finish(*self._predict(img, list(names), list(prompts), conf), conf)

    def _predict(self, img, names, prompts, conf):
        raise NotImplementedError


def need_torch():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MissingExtra(f"{exc.name} is missing: {EXTRA_HINT}") from None
    return torch


def pick_device(device="auto"):
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def hf_cache_dir(repo):
    return Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo.replace('/', '--')}"


def hf_cached(repo):
    snap = hf_cache_dir(repo) / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


class HFLabeller(Labeller):
    """Open-vocabulary model of Hugging Face transformers."""
    repo = None
    licence = "Apache-2.0"

    def __init__(self, device="auto"):
        torch = need_torch()
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device = torch, pick_device(device)
        kw = {"local_files_only": True} if hf_cached(self.repo) else {}  # offline once cached
        self.processor = AutoProcessor.from_pretrained(self.repo, **kw)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.repo, **kw)
        self.model.to(self.device).eval()
        ref = hf_cache_dir(self.repo) / "refs" / "main"
        self.version = f"{self.repo}@{ref.read_text().strip()[:12] if ref.exists() else '?'}"


class GroundingDino(HFLabeller):
    """Grounding DINO: one prompt `a. b. c.` per call (≤ 256 tokens, split otherwise). The
    score of class k for a query is the max of its token probabilities over the span of k,
    then the query keeps its best class."""
    name, repo = "grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny"

    def _predict(self, img, names, prompts, conf):
        img = downscale(img)
        tok = self.processor.tokenizer
        n_tok = [len(tok(p.lower(), add_special_tokens=False)["input_ids"]) for p in prompts]
        all_b, all_s, all_l = [], [], []
        for chunk in chunk_prompts(prompts, n_tok, GDINO_MAX_TOKENS - 2):
            b, s, lab = self._call(img, [prompts[i] for i in chunk], conf)
            all_b.append(b)
            all_s.append(s)
            all_l.append(np.asarray(chunk, dtype=np.int64)[lab])  # chunk index → class id
        return np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_l)

    def _call(self, img, prompts, conf):
        torch = self.torch
        text = " ".join(f"{p.lower()}." for p in prompts)
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        probs = outputs.logits[0].sigmoid().float().cpu().numpy()        # (Q, 256)
        boxes = outputs.pred_boxes[0].float().cpu().numpy()              # (Q, 4) cxcywh
        # Token positions of each prompt: character spans → token ids via the offsets.
        offsets = self.processor.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        cls_scores = np.zeros((len(boxes), len(prompts)))
        pos = 0
        for k, p in enumerate(prompts):
            start, end = pos, pos + len(p)
            pos = end + 2  # ". "
            toks = [t for t, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
            if toks:
                cls_scores[:, k] = probs[:, toks].max(axis=1)
        labels = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(boxes)), labels]
        keep = scores >= conf
        return boxes[keep], scores[keep], labels[keep]


class Owlv2(HFLabeller):
    """OWLv2: one text query per class. The image is padded to a square at the bottom and on
    the right: boxes come in that square and are renormalised to the image."""
    name, repo = "owlv2-base", "google/owlv2-base-patch16-ensemble"

    def _predict(self, img, names, prompts, conf):
        torch = self.torch
        img = downscale(img, 960 * 2)
        inputs = self.processor(text=[[f"a photo of a {p}" for p in prompts]], images=img,
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        side = max(img.size)
        res = self.processor.post_process_grounded_object_detection(
            outputs, threshold=conf, target_sizes=[(side, side)])[0]
        return (xyxy_px_to_cxcywh(res["boxes"].float().cpu().numpy(), *img.size),
                res["scores"].float().cpu().numpy(), res["labels"].cpu().numpy())


class YoloLabeller(Labeller):
    """A network of the project (fixed vocabulary), through `load_model` of tools/webcam.py:
    its classes are matched to the session's by name (`same_class`), the others dropped."""
    licence = "project"

    def __init__(self, spec, cfg=None, dataset=None):
        from webcam import load_model

        self.model = load_model(spec, cfg, dataset)
        self.name = self.version = f"yolo:{spec}"

    def _predict(self, img, names, prompts, conf):
        boxes, scores, labels = self.model.infer(img, conf)
        lut = np.array([next((k for k, n in enumerate(names) if same_class(m, n)), -1)
                        for m in self.model.names], dtype=np.int64)
        mapped = lut[np.asarray(labels, dtype=np.int64)] if len(labels) else np.zeros(0, int)
        keep = mapped >= 0
        return np.asarray(boxes)[keep], np.asarray(scores)[keep], mapped[keep]


REGISTRY = {"grounding-dino-tiny": GroundingDino, "owlv2-base": Owlv2}
YOLO_MODELS = ("yolo:tiny-yolov2-voc", "yolo:tiny-yolov3-coco")
DEFAULT = "grounding-dino-tiny"
# Export threshold by labeller: precision ≈ 0.85 on 200 VOC 2007 test images (T22.7), a
# wrong box costing more to fix than a missing one to draw. The score scales differ.
DEFAULT_CONF = {"grounding-dino-tiny": 0.55, "owlv2-base": 0.4}
YOLO_CONF = 0.5


def default_conf(name):
    return DEFAULT_CONF.get(name, YOLO_CONF)
NOTES = {
    "grounding-dino-tiny": "Grounding DINO Swin-T, best boxes of the light open-vocabulary models",
    "owlv2-base": "OWLv2 base, one query per class",
    "yolo:tiny-yolov2-voc": "project Tiny-YOLOv2, VOC (20), fixed vocabulary",
    "yolo:tiny-yolov3-coco": "project Tiny-YOLOv3, COCO (80), fixed vocabulary",
}


def get_labeller(name, device="auto", cfg=None, dataset=None):
    if name.startswith("yolo:"):
        return YoloLabeller(name[5:], cfg, dataset)
    if name not in REGISTRY:
        raise ValueError(f"unknown labeller {name!r}: {', '.join([*REGISTRY, *YOLO_MODELS])}"
                         " or yolo:ckpt:<path>.npz")
    return REGISTRY[name](device)


def weights_ready(name):
    """Are the weights of `name` on disk?"""
    if name.startswith("yolo:"):
        from webcam import model_files

        return all(p.exists() for p in model_files(name[5:]))
    return hf_cached(REGISTRY[name].repo)
