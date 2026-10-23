"""Evaluation of the explanation maps (T36.4).

- **pointing game** (Zhang 2016): hit when the map's maximum falls in the explained box,
  grown by `tol` pixels;
- **energy in the box**: share of the positive mass of the map inside the box (a
  "proportional" pointing game, Wang 2020);
- **deletion / insertion** (Petsiuk 2018): the pixels ranked by the map are removed (set to
  grey) or inserted (into a blurred image) in `steps` equal parts; the weight of the explained
  box is computed at each step. AUC over the fraction of pixels, by the trapezoid rule: low
  deletion and high insertion are good.

The weight is the one of D-RISE (`perturb.similarity`: IoU · class score of the best
candidate before the NMS), so the same curves work at the three stages.

Baselines: a random map (uniform noise) and a centre map (Gaussian of standard deviation H/4
at the image centre). Box sizes are binned as in T27.0, by the COCO area thresholds in pixels
of the original image: small < 32², medium < 96², large.
"""

import numpy as np

from yolo.explain.maps import box_blur, box_mask
from yolo.explain.perturb import GREY, weights

SMALL, MEDIUM = 32 ** 2, 96 ** 2
BINS = ("small", "medium", "large")


def size_bin(box, width, height):
    """Bin of a normalised `(cx, cy, w, h)` box in an original image width × height."""
    area = box[2] * width * box[3] * height
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def pointing_game(m, box, tol=15):
    h, w = m.shape
    y, x = np.unravel_index(int(np.argmax(m)), m.shape)
    cx, cy, bw, bh = box
    return bool(abs(x + 0.5 - cx * w) <= bw * w / 2 + tol
                and abs(y + 0.5 - cy * h) <= bh * h / 2 + tol)


def energy_in_box(m, box):
    pos = np.maximum(m, 0)
    total = pos.sum()
    return float(pos[box_mask(box, *m.shape)].sum() / total) if total > 0 else 0.0


def _auc(ys):
    ys = np.asarray(ys, dtype=np.float64)
    return float(((ys[1:] + ys[:-1]) / 2).mean())


def _ranked(m, steps):
    order = np.argsort(-np.ravel(m), kind="stable")
    cuts = np.linspace(0, order.size, steps + 1).round().astype(int)
    return order, cuts


def deletion(stage, x, m, box, cls, steps=16, fill=GREY):
    """(AUC, curve (steps + 1,)): pixels removed in decreasing map order."""
    _, c, h, w = x.shape
    order, cuts = _ranked(m, steps)
    stack = np.repeat(x, steps + 1, axis=0).reshape(steps + 1, c, h * w)
    for k, n in enumerate(cuts):
        stack[k, :, order[:n]] = fill
    ys = weights(stage, stack.reshape(steps + 1, c, h, w), box, cls)
    return _auc(ys), ys


def insertion(stage, x, m, box, cls, steps=16, blur=None):
    """(AUC, curve): pixels of `x` inserted in decreasing map order into a blurred `x`."""
    _, c, h, w = x.shape
    blur = max(1, w // 16) if blur is None else blur
    base = box_blur(x[0], blur).astype(x.dtype).reshape(c, h * w)
    src = x[0].reshape(c, h * w)
    order, cuts = _ranked(m, steps)
    stack = np.repeat(base[None], steps + 1, axis=0)
    for k, n in enumerate(cuts):
        stack[k][:, order[:n]] = src[:, order[:n]]
    ys = weights(stage, stack.reshape(steps + 1, c, h, w), box, cls)
    return _auc(ys), ys


def random_map(shape, rng=0):
    return np.random.default_rng(rng).random(shape)


def centre_map(shape):
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    s = h / 4
    return np.exp(-(((y + 0.5 - h / 2) ** 2 + (x + 0.5 - w / 2) ** 2) / (2 * s * s)))


def scores(stage, x, m, box, cls, steps=16):
    """All the measures of one map: {pointing, energy, deletion, insertion}."""
    return {"pointing": pointing_game(m, box), "energy": energy_in_box(m, box),
            "deletion": deletion(stage, x, m, box, cls, steps)[0],
            "insertion": insertion(stage, x, m, box, cls, steps)[0]}
