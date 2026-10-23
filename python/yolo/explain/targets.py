"""Explanation target: the scalar to explain and its δ on the head outputs (T36.0).

A target is one cell and anchor of one head, and a class c. Its scalar is the score of the
decoding before the threshold and NMS (§8.1):

    s = σ(t_o) · p_c,   p_c = σ(t_c) (`yolo` heads)  or  softmax(t)_c (`region`, v2)

- **detection**: the cell and anchor that produced a box kept by the NMS (`detections`);
- **miss**: the cell and anchor that the assignment rule of the targets (§5.1, the one of
  `yolo.data.targets.build_targets`) makes responsible for a ground-truth box (`miss_target`).
  Explaining the score there says why it stayed low.

δs on the raw head map (N, A·(5+C), S_h, S_w):

    ∂s/∂t_o = σ(t_o)(1 − σ(t_o))·p_c
    ∂s/∂t_k = σ(t_o)·p_c(1 − p_c)·[k = c]         (sigmoid)
    ∂s/∂t_k = σ(t_o)·p_c([k = c] − p_k)           (softmax: every class gets a δ)

BN in eval mode uses the running statistics, which `bn_backward` does not model: explain the
folded network (`explain_net`, `fuse_network` in float64), whose backward is exact.
"""

from dataclasses import dataclass

import numpy as np

from yolo.backend import get_xp, to_numpy
from yolo.data.targets import anchor_owner, anchor_ref, anchors_frac, heads, is_v5, responsible
from yolo.infer.decode import decode
from yolo.infer.nms import CONF_THR, IOU_THR, nms
from yolo.layers.activations import sigmoid
from yolo.models.specs import infer_shapes


@dataclass(frozen=True)
class Target:
    """Head `head` (output layer id), anchor `anchor` (position in the head's mask), cell
    (`i`, `j`), class `cls`. `box`: normalised `(cx, cy, w, h)` of the detection or of the
    ground truth; `score`: the decoded score of a detection; `kind`: "detection" or "miss"."""

    head: int
    anchor: int
    i: int
    j: int
    cls: int
    kind: str = "detection"
    box: tuple = None
    score: float = None


def explain_net(name, weights=None, dtype=np.float64):
    """Folded `Network` of a pretrained network (or a .cfg with its Darknet `weights`)."""
    from pathlib import Path

    from yolo.io.darknet_weights import load_darknet_weights
    from yolo.models.tiny_yolo import PRETRAINED, build
    from yolo.quant.fuse_bn import fuse_network

    root = Path(__file__).resolve().parents[3]
    net = build(name)
    load_darknet_weights(net, weights or root / "weights" / PRETRAINED[name])
    return fuse_network(net, dtype=dtype)


def _check(net):
    if is_v5(net) or net.get("head") == "dfl":  # YOLOv5/v7 and anchor-free heads
        raise ValueError("explanation targets: Darknet v2 / v3 / v4 heads only")


def _softmax_head(net, hid):
    return net["layers"][hid]["type"] == "region"


def _cell(out, net, t):
    """Raw values (N, 5+C) of the target's cell and anchor in head map `out`."""
    c = net["classes"]
    a0 = t.anchor * (5 + c)
    return out[:, a0:a0 + 5 + c, t.i, t.j]


def target_score(outputs, net, t):
    """Score s (N,) of target `t` for every image of the batch; `outputs` {head id: map}."""
    _check(net)
    p = _cell(outputs[t.head], net, t)
    xp = get_xp(p)
    obj = sigmoid(p[:, 4])
    tc = p[:, 5:]
    if _softmax_head(net, t.head):
        e = xp.exp(tc - tc.max(axis=1, keepdims=True))
        pc = (e / e.sum(axis=1, keepdims=True))[:, t.cls]
    else:
        pc = sigmoid(tc[:, t.cls])
    return obj * pc


def target_douts(outputs, net, t):
    """(s (N,), {head id: δs on its map}): the δ that `Network.backward` takes. Each image of
    the batch gets the δ of its own score (the batch scalar is Σ_n s_n)."""
    _check(net)
    out = outputs[t.head]
    xp = get_xp(out)
    c = net["classes"]
    p = _cell(out, net, t)
    obj = sigmoid(p[:, 4])
    tc = p[:, 5:]
    d = xp.zeros_like(p)
    if _softmax_head(net, t.head):
        e = xp.exp(tc - tc.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        pc = prob[:, t.cls]
        d[:, 5:] = -(obj * pc)[:, None] * prob
        d[:, 5 + t.cls] += obj * pc
    else:
        pc = sigmoid(tc[:, t.cls])
        d[:, 5 + t.cls] = obj * pc * (1 - pc)
    d[:, 4] = obj * (1 - obj) * pc
    dmap = xp.zeros_like(out)
    a0 = t.anchor * (5 + c)
    dmap[:, a0:a0 + 5 + c, t.i, t.j] = d
    return obj * pc, {t.head: dmap}


def cell_table(outputs, net):
    """[(head id, anchor, i, j)] in the order of `decode` (heads, then anchor, row, column)."""
    rows = []
    for hid, mask in heads(net):
        _, _, sh, sw = outputs[hid].shape
        rows += [(hid, a, i, j) for a in range(len(mask)) for i in range(sh)
                 for j in range(sw)]
    return rows


def detections(outputs, net, conf_thr=CONF_THR, iou_thr=IOU_THR, n=0):
    """Targets of the boxes of image `n` kept by the threshold and the per-class NMS of
    `filter_and_nms`, with their cell, anchor and decoded score, best score first."""
    _check(net)
    outs = {k: to_numpy(v)[n:n + 1] for k, v in outputs.items()}
    boxes, _, scores = (x[0] for x in decode(outs, net))
    table = cell_table(outs, net)
    found = []
    for c in range(scores.shape[1]):
        cand = np.nonzero(scores[:, c] > conf_thr)[0]
        for k in cand[nms(boxes[cand], scores[cand, c], iou_thr)]:
            hid, a, i, j = table[k]
            found.append(Target(hid, a, i, j, c, "detection", tuple(float(v) for v in boxes[k]),
                                float(scores[k, c])))
    return sorted(found, key=lambda t: -t.score)


def grid_sizes(net, size=None):
    """{head id: (S_h, S_w)} for the input of `net`, or for an input (H, W) `size`."""
    desc = net if size is None else {**net, "input": (net["input"][0], *size)}
    shapes = infer_shapes(desc)
    return {hid: shapes[hid][1][1:] for hid, _ in heads(net)}


def miss_target(net, box, cls, size=None):
    """Target of the cell and anchor responsible for ground-truth `box` (normalised
    `(cx, cy, w, h)`) by the rule of §5.1; `None` when its best anchor belongs to no head."""
    _check(net)
    anchors = anchors_frac(net["anchors"], anchor_ref(net))
    r = responsible(np.asarray(box, dtype=np.float64)[None], anchors,
                    anchor_owner(heads(net)), grid_sizes(net, size))[0]
    if r is None:
        return None
    _, hid, a, i, j = r
    return Target(hid, a, i, j, int(cls), "miss", tuple(float(v) for v in box))


@dataclass
class Explained:
    """`score` (N,); `dx` δs/δx (N, C, H, W); `dlayers` {id: δ of its output}; `acts` the
    outputs of every layer (forward with `all_outputs`)."""

    score: object
    dx: object
    dlayers: dict
    acts: list


def explain(net, x, t, keep=()):
    """Forward in eval mode, then backward of the target's score; `keep`: layer ids whose δ
    is returned (Grad-CAM)."""
    acts = net.forward(x, train=False, all_outputs=True)
    outputs = {i: acts[i] for i in net.outputs}
    score, douts = target_douts(outputs, net.net, t)
    dx, _, kept = net.backward(douts, keep=set(keep))
    return Explained(score, dx, kept, acts)
