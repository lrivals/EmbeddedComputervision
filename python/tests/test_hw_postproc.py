"""T9.1.1 : post-traitement tout entier (référence du noyau `yolo_post`) ; Python == C++."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from yolo.infer import hw_postproc as hp
from yolo.infer.boxes import iou
from yolo.infer.nms import nms
from yolo.io.export import load_model
from yolo.quant.lut import HeadLuts

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "build" / "golden" / "golden_run"
NETS = ["tiny-yolov2-voc", "tiny-yolov3-coco"]
IMAGES = ["000001", "000002", "000003"]


def corners(cx, cy, w, h):
    return (cx - w // 2, cy - h // 2, cx - w // 2 + w, cy - h // 2 + h)


def test_iou_fraction():
    assert hp.iou_fraction(0.45) == (9, 20)
    assert hp.iou_fraction(0.5) == (1, 2)
    assert hp.iou_fraction(0.3) == (3, 10)


def test_overlap_equals_float_iou():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a = corners(*rng.integers(0, 6656, 2), *rng.integers(1, 3000, 2))
        b = corners(*rng.integers(0, 6656, 2), *rng.integers(1, 3000, 2))
        cxcywh = [[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] for x1, y1, x2, y2 in (a, b)]
        ref = iou(np.array(cxcywh[:1], float), np.array(cxcywh[1:], float))[0, 0] > 0.45
        got = hp.overlaps(np.array(a), *(np.array([v]) for v in b), 9, 20)[0]
        assert got == ref


def box(x, score, cls=0, w=100):
    return (x, 0, x + w, w, score, cls)


def test_nms_nosort_basic():
    # b recouvre a avec un meilleur score : le remplace ; c (autre classe) reste.
    a, b, c = box(0, 100), box(10, 200), box(5, 50, cls=1)
    out, ov = hp.nms_nosort([a, b, c], 9, 20)
    assert out == [b, c] and ov == 0
    # d moins bon que la sélectionnée qu'il recouvre : rejeté ; égalité : la première gagne.
    out, _ = hp.nms_nosort([b, box(12, 150), box(14, 200)], 9, 20)
    assert out == [b]


def test_nms_nosort_vs_sorted_differs_on_chains():
    """Chaîne a < b < c, a∩b et b∩c au-delà du seuil, a∩c non : la NMS triée garde c et a ;
    sans tri, c remplace b qui avait éliminé a."""
    a, b, c = box(0, 100), box(30, 200), box(60, 300)
    out, _ = hp.nms_nosort([a, b, c], 9, 20)
    assert out == [c]
    xywh = np.array([[x + 50, 50, 100, 100] for x in (0, 30, 60)], float)
    assert list(nms(xywh, [100, 200, 300], 0.45)) == [2, 0]


def test_nms_nosort_overflow():
    cands = [box(1000 * k, 10) for k in range(5)]
    out, ov = hp.nms_nosort(cands, 9, 20, cap=3)
    assert len(out) == 3 and ov == 2


def _heads(net):
    model_dir = ROOT / "model" / net
    if not (model_dir / "manifest.json").exists():
        pytest.skip("export absent (make export)")
    qm, m = load_model(model_dir)
    luts = {e["id"]: HeadLuts(e["scale"]) for e in m["layers"] if "lut_offset" in e}
    return model_dir, qm.net, luts, [hid for hid, _ in hp.heads(qm.net)]


def _golden(model_dir, conf, iou_thr, files):
    out = subprocess.run([str(GOLDEN), "hwpp", str(model_dir), repr(conf), repr(iou_thr),
                          *map(str, files)], check=True, capture_output=True, text=True).stdout
    boxes, overflow = [], None
    for line in out.splitlines():
        tok = line.split()
        if tok[0] == "box":
            boxes.append(tuple(int(t) for t in tok[1:]))
        elif tok[0] == "overflow":
            overflow = int(tok[1])
    return boxes, overflow


@pytest.mark.parametrize("net", NETS)
def test_dumps_close_to_float_postproc(net):
    """Mêmes détections que le post-traitement de référence (§8.2) sur les dumps, boîtes à
    moins d'un pixel Q4 près."""
    import json

    model_dir, spec, luts, ids = _heads(net)
    if (model_dir / "SYNTHETIC").exists():
        pytest.skip("export synthétique (CI) : têtes aléatoires, scores quasi égaux")
    for image in IMAGES:
        outs = {h: np.load(model_dir / "dumps" / image / f"L{h:02d}.npy")[None] for h in ids}
        ref = json.loads((model_dir / "dumps" / image / "detections.json").read_text())
        boxes, ov = hp.run(hp.make_heads(outs, spec, luts), ref["conf"], ref["iou"])
        b, s, lab = hp.to_detections(boxes)
        assert ov == 0 and list(lab) == ref["labels"]
        assert np.allclose(s, ref["scores"], atol=2 * 2**-16)
        assert np.abs(b - np.array(ref["boxes"]).reshape(-1, 4)).max(initial=0) < 2 / 416


@pytest.mark.parametrize("net", NETS)
def test_python_equals_golden(net):
    if not GOLDEN.exists():
        pytest.skip("golden_run absent (make test-cpp)")
    model_dir, spec, luts, ids = _heads(net)
    rng = np.random.default_rng(1)
    cases = [({h: np.load(model_dir / "dumps" / im / f"L{h:02d}.npy") for h in ids}, conf)
             for im in IMAGES for conf in (0.25, 0.005)]
    for _ in range(6):  # têtes aléatoires : t_o élevé pour beaucoup de candidates
        heads = {h: rng.integers(-128, 128, cases[0][0][h].shape).astype(np.int8) for h in ids}
        cases.append((heads, float(rng.choice([0.005, 0.1, 0.25]))))
    with tempfile.TemporaryDirectory() as tmp:
        for n, (heads, conf) in enumerate(cases):
            files = []
            for h in ids:
                f = Path(tmp) / f"{n}_{h}.npy"
                np.save(f, heads[h])
                files.append(f)
            want = hp.run(hp.make_heads({h: heads[h][None] for h in ids}, spec, luts), conf, 0.45)
            got = _golden(model_dir, conf, 0.45, files)
            assert got == (want[0], want[1]), (n, conf)
