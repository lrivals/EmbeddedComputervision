"""T3.1 : décodage des têtes (§8.1), cohérent avec l'encodage des cibles (T2.3)."""

import numpy as np

from yolo.data.letterbox import boxes_to_letterbox
from yolo.data.targets import anchors_frac, build_targets, heads
from yolo.infer.decode import decode, decode_head, to_original
from yolo.models.specs import TINY_YOLOV3_VOC

ANCHORS = TINY_YOLOV3_VOC["anchors"]
C = 20


def _logit(x):
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return np.log(x / (1 - x))


def test_decode_inverts_encode():
    rng = np.random.default_rng(0)
    boxes = np.concatenate([rng.uniform(0.05, 0.95, (10, 2)),
                            rng.uniform(0.02, 0.8, (10, 2))], axis=1)
    hl = heads(TINY_YOLOV3_VOC)
    grids = {16: 13, 23: 26}
    t = build_targets([boxes], [np.arange(10)], ANCHORS, hl, grids)
    outputs = {}
    for hid, mask in hl:
        tt, s = t[hid], grids[hid]
        p = np.zeros((1, 3, 5 + C, s, s))
        p[:, :, 0], p[:, :, 1] = _logit(tt["x"]), _logit(tt["y"])
        p[:, :, 2], p[:, :, 3] = tt["tw"], tt["th"]
        p[:, :, 4] = np.where(tt["obj"], 10.0, -10.0)
        p[:, :, 5:] = -10.0
        for n, a, i, j in zip(*np.nonzero(tt["obj"])):
            p[n, a, 5 + tt["cls"][n, a, i, j], i, j] = 10.0
        outputs[hid] = p.reshape(1, -1, s, s)
    dboxes, obj, scores = decode(outputs, TINY_YOLOV3_VOC)
    assert dboxes.shape == (1, 3 * (13 * 13 + 26 * 26), 4)
    assert scores.shape == (1, 3 * (13 * 13 + 26 * 26), C)
    sel = obj[0] > 0.5
    labels = scores[0, sel].argmax(axis=1)
    got = dboxes[0, sel][np.argsort(labels)]
    np.testing.assert_allclose(got, boxes[np.sort(labels)], atol=1e-9)


def test_scores_v3_and_v2():
    rng = np.random.default_rng(1)
    out = rng.standard_normal((2, 2 * (5 + 4), 3, 3))
    anchors = anchors_frac([[50, 60], [100, 80]])
    p = out.reshape(2, 2, 9, 3, 3)
    sig = lambda t: 1 / (1 + np.exp(-t))  # noqa: E731
    _, obj3, s3 = decode_head(out, anchors, 4, "v3")
    _, obj2, s2 = decode_head(out, anchors, 4, "v2")
    # Ancre 1, ligne 2, colonne 0 → indice 1·9 + 2·3 + 0.
    k, (a, i, j) = 15, (1, 2, 0)
    t = p[0, a, :, i, j]
    assert obj3[0, k] == obj2[0, k] == sig(t[4])
    np.testing.assert_allclose(s3[0, k], sig(t[4]) * sig(t[5:]), rtol=1e-14)
    e = np.exp(t[5:])
    np.testing.assert_allclose(s2[0, k], sig(t[4]) * e / e.sum(), rtol=1e-14)
    np.testing.assert_allclose(s2.sum(axis=2), obj2, rtol=1e-14)


def test_decode_box_values():
    out = np.zeros((1, 5 + 1, 2, 2))
    out[0, 2, 1, 0] = np.log(2.0)
    boxes, _, _ = decode_head(out, [[0.1, 0.2]], 1)
    # σ(0) = 0,5 : centre de la cellule ; ligne 1, colonne 0 → indice 2.
    np.testing.assert_allclose(boxes[0, 2], [0.25, 0.75, 0.2, 0.2])


def test_to_original_inverts_letterbox():
    boxes = np.array([[0.3, 0.6, 0.2, 0.1], [0.9, 0.1, 0.05, 0.08]])
    lb = boxes_to_letterbox(boxes, 500, 333, 416)
    np.testing.assert_allclose(to_original(lb, 500, 333, 416), boxes, atol=1e-12)
