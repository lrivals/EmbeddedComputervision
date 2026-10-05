"""T4.3 : modèle entier complet — moteurs bit-exacts, aucune valeur flottante, écart à la
couche flottante ≤ 2 % de la dynamique, accumulateur sous 2³⁰ (couche 13).
"""

from pathlib import Path

import numpy as np
import pytest

from yolo.layers.activations import leaky_forward
from yolo.layers.conv import conv_forward
from yolo.models.tiny_yolo import build
from yolo.quant.calibrate import ActStats, choose_scales, load_scales
from yolo.quant.fuse_bn import fuse_network
from yolo.quant.int_model import IntNetwork, QuantModel, fake_quant_forward
from yolo.quant.quantize import INPUT_SCALE, quantize_input

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights" / "yolov2-tiny-voc.weights"
IMAGE = ROOT / "data" / "VOCdevkit" / "VOC2007" / "JPEGImages" / "000004.jpg"


def _random_bn(net, rng):
    for i, layer in enumerate(net.layers):
        if layer["type"] == "conv" and layer["bn"]:
            c = layer["cout"]
            net.params[i]["gamma"][:] = rng.uniform(0.5, 2.0, c)
            net.params[i]["beta"][:] = rng.normal(0, 0.2, c)
            net.state[i]["var"][:] = rng.uniform(0.5, 2.0, c)


def _quant_model(fused, x):
    stats = ActStats(fused, per_image=4096, rng=0)
    stats.update(fused.forward(x, train=False, all_outputs=True))
    scales, _, _ = choose_scales(fused, stats, head_scale=None)
    return QuantModel.from_fused(fused, INPUT_SCALE, scales)


def _layer_errors(fused, qm, qx, outs):
    """Écart max de chaque conv entière à la conv flottante sur la **même** entrée
    déquantifiée, en fraction de la dynamique 254·s_y (sortie flottante saturée)."""
    errs = {}
    for i, c in qm.convs.items():
        j = fused.inputs[i][0]
        xin = (qx if j < 0 else outs[j]).astype(np.float64) * c.sx
        p = fused.params[i]
        y, _ = conv_forward(xin, p["W"].astype(np.float64), p["b"].astype(np.float64))
        if c.act == "leaky":
            y, _ = leaky_forward(y)
        y = np.clip(y, -127 * c.sy, 127 * c.sy)
        errs[i] = np.abs(outs[i] * c.sy - y).max() / (254 * c.sy)
    return errs


@pytest.fixture(scope="module", params=["tiny-yolov2-voc", "tiny-yolov3-voc"])
def setup(request):
    rng = np.random.default_rng(0)
    net = build(request.param, dtype=np.float64, rng=1)
    _random_bn(net, rng)
    fused = fuse_network(net)
    x = rng.uniform(0, 1, (2, 3, 64, 64))
    qm = _quant_model(fused, x)
    return fused, qm, quantize_input(x)


def test_engines_bit_exact(setup):
    _, qm, qx = setup
    a = IntNetwork(qm, "int64").forward(qx, all_outputs=True)
    b = IntNetwork(qm, "f64").forward(qx, all_outputs=True)
    assert len(a) == len(qm.layers)
    for i, (u, v) in enumerate(zip(a, b)):
        assert u.dtype == np.int8 and np.array_equal(u, v), i
        assert u.min() >= -127


def test_heads_and_types(setup):
    _, qm, qx = setup
    out = IntNetwork(qm).forward(qx)
    assert sorted(out) == qm.outputs
    for c in qm.convs.values():
        assert c.qW.dtype == np.int8 and c.qb.dtype == np.int32 and c.M0.dtype == np.int32
        assert 1 <= c.n <= 31
    with pytest.raises(TypeError):
        IntNetwork(qm).forward(qx.astype(np.float32))


def test_layer_error_within_2_percent(setup):
    fused, qm, qx = setup
    outs = IntNetwork(qm).forward(qx, all_outputs=True)
    errs = _layer_errors(fused, qm, qx, outs)
    assert max(errs.values()) <= 0.02, errs


def test_route_sources_share_scale(setup):
    _, qm, _ = setup
    for i, layer in enumerate(qm.layers):
        if layer["type"] == "route":
            assert len({qm.scale_of(j) for j in layer["from"]}) == 1


def test_fake_quant_simulation_close_to_int(setup):
    """La simulation flottante (sensibilité, T4.5) suit le modèle entier à quelques pas près :
    seuls diffèrent les arrondis (biais, M0, leaky avant ou après l'arrondi), qui se cumulent.
    """
    fused, qm, qx = setup
    sim = fake_quant_forward(fused, qm, qx.astype(np.float64) * INPUT_SCALE, all_outputs=True)
    real = IntNetwork(qm).forward(qx, all_outputs=True)
    assert np.abs(sim[0] / qm.convs[0].sy - real[0]).max() <= 1 + 1e-9
    for i, c in qm.convs.items():
        assert np.abs(sim[i] / c.sy - real[i]).mean() <= 3, i


CALIB = ROOT / "build" / "quant" / "tiny-yolov2-voc" / "calib.json"


@pytest.mark.skipif(not (WEIGHTS.exists() and IMAGE.exists() and CALIB.exists()),
                    reason="poids Darknet, VOC ou calibration absents (make calibrate)")
def test_darknet_weights_real_image():
    """Tiny-YOLOv2 VOC réel, échelles calibrées (T4.2) : couche 13 sous 2³⁰ (borne et
    observé), écarts ≤ 2 %."""
    from PIL import Image

    from yolo.infer.pipeline import preprocess
    from yolo.io.darknet_weights import load_darknet_weights

    net = build("tiny-yolov2-voc", dtype=np.float64)
    load_darknet_weights(net, WEIGHTS)
    fused = fuse_network(net)
    with Image.open(IMAGE) as img:
        x = preprocess(img, 416)[0][None].astype(np.float64)
    qm = QuantModel.from_fused(fused, *load_scales(CALIB))
    qx = quantize_input(x)
    inet = IntNetwork(qm)
    outs = inet.forward(qx, all_outputs=True)
    assert qm.acc_bound(13) < 2**30
    assert inet.max_acc[13] <= qm.acc_bound(13)
    errs = _layer_errors(fused, qm, qx, outs)
    assert max(errs.values()) <= 0.02, errs
