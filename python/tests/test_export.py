"""T4.7 : export du modèle entier — manifest valide, blobs relus == modèle, dumps rejoués."""

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from yolo.io.export import export_model, layout, load_luts, load_model
from yolo.models.tiny_yolo import build
from yolo.quant.calibrate import ActStats, choose_scales
from yolo.quant.fuse_bn import fuse_network
from yolo.quant.int_model import IntNetwork, QuantModel, head_luts
from yolo.quant.quantize import INPUT_SCALE, quantize_input

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "docs" / "manifest.schema.json").read_text())
EXPORTS = [ROOT / "model" / n for n in ("tiny-yolov2-voc", "tiny-yolov3-coco")]


def _qm(name, rng):
    fused = fuse_network(build(name, dtype=np.float64, rng=1))
    stats = ActStats(fused, per_image=1024, rng=0)
    x = rng.uniform(0, 1, (1, 3, 64, 64))
    stats.update(fused.forward(x, train=False, all_outputs=True))
    scales, _, _ = choose_scales(fused, stats)
    return QuantModel.from_fused(fused, INPUT_SCALE, scales), quantize_input(x)


@pytest.fixture(scope="module", params=["tiny-yolov2-voc", "tiny-yolov3-voc"])
def exported(request, tmp_path_factory):
    qm, qx = _qm(request.param, np.random.default_rng(0))
    d = tmp_path_factory.mktemp(request.param)
    return qm, qx, d, export_model(qm, d)


def test_manifest_valid(exported):
    _, _, d, m = exported
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    assert json.loads((d / "manifest.json").read_text()) == m
    for name, size in m["blobs"].items():
        assert (d / name).stat().st_size == size


def test_blobs_reloaded_identical(exported):
    qm, qx, d, _ = exported
    rq, m = load_model(d)
    assert rq.input_scale == qm.input_scale
    for i, c in qm.convs.items():
        r = rq.convs[i]
        assert np.array_equal(r.qW, c.qW) and np.array_equal(r.qb, c.qb)
        assert np.array_equal(r.M0, c.M0) and r.n == c.n
        assert (r.sx, r.sy, r.act) == (c.sx, c.sy, c.act)
    a = IntNetwork(qm).forward(qx, all_outputs=True)
    b = IntNetwork(rq).forward(qx, all_outputs=True)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    luts = load_luts(d, m)
    for hid, h in head_luts(qm).items():
        for name, table in luts[hid].items():
            assert np.array_equal(table, getattr(h, name))


def test_v2_region_and_float_anchors(exported):
    qm, _, _, m = exported
    if qm.net["name"] != "tiny-yolov2-voc":
        pytest.skip("YOLOv2 seulement")
    assert m["layers"][-1]["type"] == "region" and m["layers"][-1]["num"] == 5
    assert m["anchors"][0] == [pytest.approx(34.56), pytest.approx(38.08)]
    # Chaîne simple : ping-pong A/B, tête dans H13, aucun tampon spécial.
    assert set(m["buffers"]) == {"A", "B", "H13"}


def test_layout_rejects_unfused_maxpool():
    net = {"input": (3, 32, 32), "layers": [{"type": "maxpool", "k": 2, "s": 2}]}
    with pytest.raises(ValueError):
        layout(net)


@pytest.mark.parametrize("d", EXPORTS, ids=lambda p: p.name)
def test_real_export_dumps_replay(d):
    """Export réel (`make export`) : manifest valide, dumps == passe avant du modèle relu."""
    if not (d / "manifest.json").exists():
        pytest.skip(f"{d.name} non exporté (make export)")
    qm, m = load_model(d)
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    dumps = sorted((d / "dumps").iterdir())
    assert len(dumps) == 3
    for dd in dumps[:1]:  # une image suffit ici ; les 3 sont rejouées par le golden C++
        qx = np.load(dd / "input.npy")
        outs = IntNetwork(qm).forward(qx, all_outputs=True)
        assert len(outs) == len(m["layers"])
        for i, o in enumerate(outs):
            ref = np.load(dd / f"L{i:02d}.npy")
            assert ref.dtype == np.int8 and np.array_equal(o, ref), i


def test_pack4_roundtrip():
    from yolo.io.export import pack4, unpack4, weight_bytes

    q = np.random.default_rng(0).integers(-8, 8, 37)
    b = pack4(q)
    assert b.dtype == np.int8 and len(b) == weight_bytes(37, 4) == 19
    assert np.array_equal(unpack4(b, 37), q)
    # quartet bas = poids d'indice pair
    assert pack4(np.array([1, -1])).view(np.uint8)[0] == 0xF1
    with pytest.raises(ValueError):
        pack4(np.array([8]))


def test_export_wbits4_roundtrip(tmp_path):
    qm, qx = _qm("tiny-yolov2-voc", np.random.default_rng(0))
    i = sorted(qm.convs)[1]
    c = qm.convs[i]
    c.qW = np.clip(c.qW, -7, 7).astype(np.int8)
    c.wbits = 4
    m = export_model(qm, tmp_path)
    e = next(e for e in m["layers"] if e["id"] == i)
    assert e["wbits"] == 4
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    back, _ = load_model(tmp_path)
    assert back.convs[i].wbits == 4 and np.array_equal(back.convs[i].qW, c.qW)
    a, b = IntNetwork(qm).forward(qx), IntNetwork(back).forward(qx)
    assert all(np.array_equal(a[k], b[k]) for k in a)
