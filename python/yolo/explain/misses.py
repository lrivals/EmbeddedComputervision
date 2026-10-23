"""Causes of the missed objects (T36.5), from the head outputs and the post-processing.

A ground-truth box of class c is **missed** when no detection kept by the NMS has class c and
IoU ≥ `match_iou` with it. Its cause is the first test that holds, in this fixed order, so
the causes form a partition:

1. **quantization** (integer stage only): the float model, run on the same image, detects
   it. Tested first: the other causes describe the integer model's own outputs, this one says
   the float model had none of them;
2. **no_anchor**: no anchor shape has IoU ≥ `anchor_iou` with the box (§5.1 shape IoU);
3. **low_objectness**: σ(t_o) < the threshold at every candidate of the box, i.e. at the
   responsible cell and anchor (§5.1) and at every pre-NMS box with IoU ≥ `match_iou`;
4. **class_confusion**: among them, the one of best objectness ≥ threshold has another
   argmax class;
5. **below_threshold**: right class, but its score σ(t_o)·p_c is under the operating point;
6. **nms_suppressed**: a candidate box of class c with IoU ≥ `match_iou` passed the threshold
   before the NMS; the NMS removed it;
7. **mislocalized**: what is left. A box of class c passed the threshold at the responsible
   cell, but none of its boxes reaches IoU ≥ `match_iou`. The task does not list this cause;
   it is needed for the partition.

The cause says what to fix: anchors (k-means again), the detector's data or the threshold
(T29.2), the NMS (crowds), the quantization (QAT).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from yolo.data.targets import anchor_ref, anchors_frac, heads
from yolo.infer.boxes import iou, iou_wh
from yolo.infer.nms import CONF_THR, IOU_THR, filter_and_nms

CAUSES = ("quantization", "no_anchor", "low_objectness", "class_confusion", "below_threshold",
          "nms_suppressed", "mislocalized")
MATCH_IOU = 0.5
ANCHOR_IOU = 0.25


@dataclass
class StageOutputs:
    """One image at one stage: every cell and anchor decoded, in the order of `decode`
    (`boxes` (K, 4), `obj` (K,), `scores` (K, C)), and the detections after the NMS."""

    boxes: np.ndarray
    obj: np.ndarray
    scores: np.ndarray
    dets: tuple


def float_outputs(heads_out, net, conf_thr=CONF_THR, iou_thr=IOU_THR):
    """`heads_out` {id: (1, ·, S, S)} float maps of one image."""
    from yolo.infer.decode import decode

    boxes, obj, scores = (v[0] for v in decode(heads_out, net))
    return StageOutputs(boxes, obj, scores, filter_and_nms(boxes, scores, conf_thr, iou_thr))


def int_outputs(heads_out, net, luts, conf_thr=CONF_THR, iou_thr=IOU_THR):
    """`heads_out` {id: (1, ·, S, S)} int8 maps; table decoding of every cell (the objectness
    threshold of `decode_head_int` is set below the smallest int8 logit)."""
    from yolo.infer.decode import decode_head_int, decode_int

    anchors = anchors_frac(net["anchors"], anchor_ref(net))
    parts = []
    for hid, mask in heads(net):
        layer = net["layers"][hid]
        mode = "v2" if layer["type"] == "region" else "v3"
        floor = 1.0 / (1.0 + np.exp(128.5 * luts[hid].scale))
        parts.append(decode_head_int(heads_out[hid][0], anchors[mask], net["classes"], mode,
                                     luts[hid], floor, layer.get("scale_x_y", 1.0),
                                     bool(layer.get("new_coords"))))
    boxes, obj, scores = (np.concatenate(x, axis=0) for x in zip(*parts))
    dets = filter_and_nms(*decode_int(heads_out, net, luts, conf_thr)[0], conf_thr, iou_thr)
    return StageOutputs(boxes, obj, scores, dets)


def matched(dets, box, cls, match_iou=MATCH_IOU):
    b, _, lab = dets
    keep = np.asarray(lab) == cls
    if not keep.any():
        return False
    return bool((iou(np.asarray(box, dtype=np.float64)[None], np.asarray(b)[keep])[0]
                 >= match_iou).any())


def responsible_index(net, box, grids):
    """Index, in the order of `decode`, of the cell and anchor responsible for `box` (§5.1);
    `grids` {head id: (S_h, S_w)}."""
    from yolo.data.targets import anchor_owner, responsible

    anchors = anchors_frac(net["anchors"], anchor_ref(net))
    r = responsible(np.asarray(box, dtype=np.float64)[None], anchors,
                    anchor_owner(heads(net)), grids)[0]
    _, hid, a, i, j = r
    off = 0
    for h, mask in heads(net):
        sh, sw = grids[h]
        if h == hid:
            return off + (a * sh + i) * sw + j
        off += len(mask) * sh * sw
    raise AssertionError(hid)


def classify_miss(net, box, cls, out, grids, conf_thr=CONF_THR, float_out=None,
                  match_iou=MATCH_IOU, anchor_iou=ANCHOR_IOU):
    """Cause of the miss of ground truth (`box`, `cls`) in `out` (`StageOutputs`); `None` if
    it is detected. `float_out`: the float stage's outputs, for the integer stage."""
    if matched(out.dets, box, cls, match_iou):
        return None
    if float_out is not None and matched(float_out.dets, box, cls, match_iou):
        return "quantization"
    anchors = anchors_frac(net["anchors"], anchor_ref(net))
    if iou_wh(np.asarray(box[2:4], dtype=np.float64)[None], anchors).max() < anchor_iou:
        return "no_anchor"
    ov = iou(np.asarray(box, dtype=np.float64)[None], out.boxes)[0]
    near = np.nonzero(ov >= match_iou)[0]
    cand = np.union1d(near, [responsible_index(net, box, grids)]).astype(int)
    if out.obj[cand].max() < conf_thr:
        return "low_objectness"
    best = cand[np.argmax(out.obj[cand])]
    probs = out.scores[best] / max(out.obj[best], 1e-300)
    if int(np.argmax(probs)) != cls:
        return "class_confusion"
    if out.scores[cand, cls].max() <= conf_thr:
        return "below_threshold"
    if (out.scores[near, cls] > conf_thr).any():
        return "nms_suppressed"
    return "mislocalized"


def shares(causes):
    """{cause: share} over the misses (`None` entries, the detections, left out)."""
    c = Counter(x for x in causes if x is not None)
    n = sum(c.values())
    return {k: (c[k] / n if n else 0.0) for k in CAUSES}
