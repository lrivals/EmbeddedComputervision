"""Perturbation attributions, black box (T36.2): occlusion and D-RISE.

No gradient: a method only calls a **stage**, a function that maps a stack of preprocessed
images (n, C, H, W) to the candidates of each image before the NMS, `(boxes (K, 4) normalised
(cx, cy, w, h), scores (K, C))`. The three stages of the chain:

- `float_stage`: the float `Network` and `decode`;
- `int_stage`: `IntNetwork` and the table decoding `decode_int` (§9.4);
- `golden_stage`: the C++ golden (`golden_run run`), whose int8 heads are decoded by the same
  `decode_int`. Golden == C-sim layer by layer (M28), so its maps are those of the HLS
  testbench too.

The integer and golden stages see the same int8 heads, hence the same candidates and the same
maps bit for bit: the masks are drawn from a seed, the same at every stage.

Similarity of a candidate j to the explained box b and class c: IoU(b, b_j)·s_j[c], with
s_j[c] = σ(t_o)·p_c its score. It is the D-RISE similarity of Petsiuk 2021, IoU·cos(P, P_j)·O_j,
with the one-hot class vector P, before the division by ‖P_j‖ (the integer decoding gives the
score, not O_j and P_j apart). The weight of a perturbed image is its best similarity. For a
**miss**, b is the ground-truth box: the map says which pixels move the detector towards the
missed person.

- **occlusion** (Zeiler 2014): a `patch` grey square slid by `stride`; map = mean drop of the
  weight over the positions that cover a pixel.
- **D-RISE**: `n` masks, a random s × s binary grid (keep probability p) upsampled bilinearly
  over (H + H/s) × (W + W/s) and cropped at a random shift (RISE, Petsiuk 2018); the image is
  multiplied by the mask; map = Σ_i w_i M_i / (n·p).
"""

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from yolo.explain.maps import resize
from yolo.infer.boxes import iou

CAND_THR = 0.005  # candidates kept before the NMS (the score floor of the OOD tools)
GREY = 0.5


# --------------------------------------------------------------------------- stages
def _keep(boxes, scores, conf):
    k = scores.max(axis=1) > conf
    return boxes[k], scores[k]


def float_stage(net, conf=CAND_THR, chunk=8):
    from yolo.backend import to_numpy
    from yolo.infer.decode import decode

    def run(xs):
        out = []
        for k in range(0, len(xs), chunk):
            heads = {i: to_numpy(v) for i, v in net.forward(xs[k:k + chunk], train=False).items()}
            boxes, _, scores = decode(heads, net.net)
            out += [_keep(b, s, conf) for b, s in zip(boxes, scores)]
        return out
    return run


def int_stage(inet, luts, conf=CAND_THR, chunk=8):
    from yolo.infer.decode import decode_int
    from yolo.quant.quantize import quantize_input

    def run(xs):
        out = []
        for k in range(0, len(xs), chunk):
            heads = inet.forward(quantize_input(xs[k:k + chunk]))
            out += list(decode_int(heads, inet.qm.net, luts, conf))
        return out
    return run


GOLDEN = Path(__file__).resolve().parents[3] / "build" / "golden" / "golden_run"


def golden_heads(model_dir, qx, head_ids, golden=GOLDEN):
    """Int8 heads {id: (1, C, S, S)} of the C++ golden for one int8 input (1, 3, H, W)."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input.npy"
        np.save(inp, np.ascontiguousarray(qx, dtype=np.int8))
        r = subprocess.run([str(golden), "run", str(model_dir), str(inp), tmp],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"golden_run: {r.stderr.strip() or r.returncode}")
        return {i: np.load(Path(tmp) / f"L{i:02d}.npy") for i in head_ids}


def golden_stage(model_dir, qm, luts, conf=CAND_THR, jobs=8, golden=GOLDEN):
    from yolo.infer.decode import decode_int
    from yolo.quant.quantize import quantize_input

    def one(x):
        heads = golden_heads(model_dir, quantize_input(x[None]), qm.outputs, golden)
        return decode_int(heads, qm.net, luts, conf)[0]

    def run(xs):
        with ThreadPoolExecutor(jobs) as ex:
            return list(ex.map(one, list(xs)))
    return run


# ---------------------------------------------------------------------- similarity
def similarity(cands, box, cls):
    """Best IoU(box, b_j)·s_j[cls] over the candidates `(boxes, scores)` of one image."""
    boxes, scores = cands
    if len(boxes) == 0:
        return 0.0
    ov = iou(np.asarray(box, dtype=np.float64)[None], np.asarray(boxes, dtype=np.float64))[0]
    return float((ov * np.asarray(scores, dtype=np.float64)[:, cls]).max())


def weights(stage, xs, box, cls):
    return np.array([similarity(c, box, cls) for c in stage(xs)])


# -------------------------------------------------------------------------- methods
def occlusion(stage, x, box, cls, patch=64, stride=32, fill=GREY):
    """`x` (1, C, H, W); returns {map, passes, base}."""
    _, _, h, w = x.shape
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs_ = list(range(0, max(w - patch, 0) + 1, stride))
    pos = [(y, x0) for y in ys for x0 in xs_]
    stack = np.repeat(x, len(pos), axis=0)
    for k, (y, x0) in enumerate(pos):
        stack[k, :, y:y + patch, x0:x0 + patch] = fill
    base = weights(stage, x, box, cls)[0]
    drop = base - weights(stage, stack, box, cls)
    acc = np.zeros((h, w))
    cnt = np.zeros((h, w))
    for d, (y, x0) in zip(drop, pos):
        acc[y:y + patch, x0:x0 + patch] += d
        cnt[y:y + patch, x0:x0 + patch] += 1
    return {"map": acc / np.maximum(cnt, 1), "passes": len(pos) + 1, "base": base}


def rise_masks(n, h, w, s=16, p=0.5, rng=0):
    """`n` RISE masks (n, h, w) in [0, 1], reproducible from `rng`."""
    rng = np.random.default_rng(rng)
    ch, cw = int(np.ceil(h / s)), int(np.ceil(w / s))
    out = np.empty((n, h, w))
    for k in range(n):
        grid = (rng.random((s, s)) < p).astype(np.float64)
        up = resize(grid, (s + 1) * ch, (s + 1) * cw)
        dy, dx = rng.integers(0, ch), rng.integers(0, cw)
        out[k] = up[dy:dy + h, dx:dx + w]
    return out


def drise(stage, x, box, cls, n=500, s=16, p=0.5, rng=0, batch=32, masks=None):
    """`x` (1, C, H, W); returns {map, passes, weights}. `masks`: precomputed `rise_masks`."""
    _, _, h, w = x.shape
    masks = rise_masks(n, h, w, s, p, rng) if masks is None else masks
    wts = np.concatenate([weights(stage, (x * masks[k:k + batch, None]).astype(x.dtype), box,
                                  cls) for k in range(0, len(masks), batch)])
    sal = np.tensordot(wts, masks, axes=1) / (len(masks) * p)
    return {"map": sal, "passes": len(masks), "weights": wts}


METHODS = {"occlusion": occlusion, "drise": drise}
