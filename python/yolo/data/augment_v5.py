"""YOLOv5 / YOLOv7 training augmentation in NumPy (T33.7): mosaic, random perspective, HSV,
horizontal flip, box filtering.

The reference (`utils/dataloaders.py`, `utils/augmentations.py` of YOLOv5) works on uint8
images with OpenCV; here the images are uint8 (H, W, 3) arrays, resized and warped with
Pillow (bilinear), boxes in pixels `(x1, y1, x2, y2)`.

- `load_resized`: the long side resized to S (ratio kept).
- `mosaic4`: canvas 2S × 2S filled with gray 114, centre (x_c, y_c) uniform in [S/2, 3S/2];
  the image and 3 random others fill the four quadrants around it, cropped at the canvas.
- `random_perspective` with the hyperparameters of `hyp.scratch-low.yaml` /
  `hyp.scratch.tiny.yaml` (no rotation, shear or perspective: scale ±0.5, translation 0.1):
  M = T·S·C (C moves the centre of the input to the origin, S scales by s ~ U(1 − 0.5,
  1 + 0.5), T moves it to a point uniform in [0.4, 0.6] of the output); the mosaic is
  warped from 2S to S. The four corners of each box are mapped, the box is their bounding
  box clipped to the output, and kept by `box_candidates`.
- `box_candidates`: kept if both sides ≥ 2 px, area ratio new / (old·s) > 0.1 and aspect
  ratio < 100.
- `augment_hsv`: gains r = 1 + U(−1, 1)·(h, s, v) = (0.015, 0.7, 0.4): hue multiplied (mod
  1), saturation and value multiplied and clipped.
- flip: x → W − x with probability 0.5.

Not implemented (both off or small in the files): rotation, shear, perspective, vertical
flip, mixup (0.05 in `hyp.scratch.tiny.yaml`), copy-paste and paste-in (YOLOv7).
"""

import numpy as np

from yolo.data.augment import hsv_to_rgb, rgb_to_hsv

AUG_V5 = {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1,
          "scale": 0.5, "shear": 0.0, "perspective": 0.0, "fliplr": 0.5, "mosaic": 1.0}
GRAY_114 = 114


def to_xyxy_px(boxes, w, h):
    """Normalized `(cx, cy, w, h)` → pixels `(x1, y1, x2, y2)`."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], axis=1)


def to_cxcywh_norm(xyxy, w, h):
    b = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h,
                     (b[:, 2] - b[:, 0]) / w, (b[:, 3] - b[:, 1]) / h], axis=1)


def load_resized(img, size):
    """PIL image → (uint8 (h, w, 3) with long side `size`, scale r)."""
    from PIL import Image

    img = img.convert("RGB")
    r = size / max(img.width, img.height)
    if r != 1:
        img = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))),
                         Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8), r


def mosaic4(items, size, rng):
    """`items`: 4 (uint8 image, pixel xyxy boxes, labels) resized by `load_resized`.

    Returns the 2S × 2S canvas, its boxes (pixels, not clipped) and labels.
    """
    s = size
    xc, yc = (int(v) for v in rng.uniform(s / 2, 3 * s / 2, 2))
    canvas = np.full((2 * s, 2 * s, 3), GRAY_114, dtype=np.uint8)
    out_b, out_l = [], []
    for k, (img, boxes, labels) in enumerate(items):
        h, w = img.shape[:2]
        if k == 0:    # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif k == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif k == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:         # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] += x1a - x1b
        b[:, [1, 3]] += y1a - y1b
        out_b.append(b)
        out_l.append(np.asarray(labels).reshape(-1))
    return canvas, np.concatenate(out_b), np.concatenate(out_l)


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """Boxes kept after the warp: `box1` before (scaled), `box2` after, xyxy (n, 4)."""
    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def random_perspective(img, boxes, labels, out_size, rng, hyp=AUG_V5):
    """Warp of `img` (uint8 (H, W, 3)) to `out_size` × `out_size` (see the module doc).

    Returns (uint8 image, pixel xyxy boxes, labels) after `box_candidates`.
    """
    from PIL import Image

    h, w = img.shape[:2]
    c = np.eye(3)
    c[0, 2], c[1, 2] = -w / 2, -h / 2
    s = rng.uniform(1 - hyp["scale"], 1 + hyp["scale"])
    sc = np.diag([s, s, 1.0])
    t = np.eye(3)
    t[0, 2] = rng.uniform(0.5 - hyp["translate"], 0.5 + hyp["translate"]) * out_size
    t[1, 2] = rng.uniform(0.5 - hyp["translate"], 0.5 + hyp["translate"]) * out_size
    m = t @ sc @ c
    inv = np.linalg.inv(m)
    warped = Image.fromarray(img).transform(
        (out_size, out_size), Image.AFFINE, tuple(inv[:2].reshape(-1)),
        resample=Image.BILINEAR, fillcolor=(GRAY_114,) * 3)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(b):
        corners = b[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)          # 4 corners per box
        xy = (np.c_[corners, np.ones(len(corners))] @ m.T)[:, :2].reshape(-1, 8)
        x, y = xy[:, 0::2], xy[:, 1::2]
        new = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], axis=1)
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, out_size)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, out_size)
        keep = box_candidates(b * s, new)
        b, labels = new[keep], np.asarray(labels)[keep]
    return np.asarray(warped, dtype=np.uint8), b, np.asarray(labels).reshape(-1)


def augment_hsv(img, rng, hgain=0.015, sgain=0.7, vgain=0.4):
    """uint8 RGB image → uint8 with random HSV gains (see the module doc)."""
    r = rng.uniform(-1, 1, 3) * (hgain, sgain, vgain) + 1
    hsv = rgb_to_hsv(img.astype(np.float64) / 255)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 1)
    return np.round(hsv_to_rgb(hsv) * 255).astype(np.uint8)


def fliplr(img, boxes):
    w = img.shape[1]
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = w - b[:, [2, 0]]
    return img[:, ::-1], b
