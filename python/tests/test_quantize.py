"""T4.2 : poids int8 symétriques par canal, biais int32, M0/n, calibration et routes."""

import numpy as np
import pytest

from yolo.models.tiny_yolo import build
from yolo.quant.calibrate import (HEAD_SCALE, ActStats, choose_scales, clip_candidates, quant_mse,
                                  scale_owners, unify_routes)
from yolo.quant.fuse_bn import fuse_network
from yolo.quant.quantize import (INPUT_SCALE, quantize, quantize_bias, quantize_input,
                                 quantize_weights_per_channel, requant_params, round_half_up)


def test_round_half_up_on_negatives():
    v = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -2.4, -2.6])
    assert round_half_up(v).tolist() == [-2, -1, 0, 1, 2, 3, -2, -3]


def test_weights_per_output_channel():
    rng = np.random.default_rng(0)
    # Canaux d'amplitudes très différentes, comme après la fusion BN (§9.2).
    W = rng.standard_normal((6, 4, 3, 3)) * np.array([1e-3, 0.1, 1, 10, 100, 0.0])[:, None,
                                                                                    None, None]
    qW, sw = quantize_weights_per_channel(W)
    assert qW.dtype == np.int8 and sw.shape == (6,)
    for f in range(5):
        assert np.abs(qW[f]).max() == 127  # chaque filtre utilise toute la plage
        assert np.abs(qW[f] * sw[f] - W[f]).max() <= sw[f] / 2 + 1e-15
    assert np.all(qW[5] == 0)
    assert np.all(qW >= -127)  # symétrique : −128 jamais utilisé


def test_quantize_bias_and_overflow():
    sw = np.array([0.01, 0.02])
    qb = quantize_bias(np.array([0.5, -0.3]), 0.1, sw)
    assert qb.dtype == np.int32 and qb.tolist() == [500, -150]
    with pytest.raises(OverflowError):
        quantize_bias(np.array([1e9]), 1e-3, np.array([1e-3]))


def test_requant_params():
    rng = np.random.default_rng(1)
    sx, sy = 0.05, 0.08
    sw = rng.uniform(1e-4, 1e-2, 16)
    M0, n = requant_params(sx, sw, sy)
    assert M0.dtype == np.int32 and n == 31
    assert np.all(np.abs(M0 / 2.0**n - sx * sw / sy) <= 2.0**-(n + 1))
    # Rapport > 1 : n diminue pour garder M0 < 2³¹.
    M0, n = requant_params(1.0, np.array([3.0]), 1.0)
    assert n == 29 and M0[0] == 3 << 29


def test_quantize_input():
    q = quantize_input(np.array([0.0, 0.5, 1.0, 1.2, -0.1]))
    assert q.dtype == np.int8 and q.tolist() == [0, 64, 127, 127, -13]
    assert INPUT_SCALE == 1 / 127


def test_quantize_clips_symmetric():
    assert quantize(np.array([-1000.0, 1000.0]), 1.0).tolist() == [-127, 127]


def test_mse_criterion_picks_best_candidate():
    # Queue lourde (Laplace) : écrêter un peu bat le max, écrêter trop (p90) est pire.
    v = np.random.default_rng(2).laplace(0, 1, 200_000)
    c = clip_candidates(v, np.abs(v).max())
    assert set(c) == {"p90", "p95", "p99", "p99.9", "p99.99", "max"}
    mses = {k: quant_mse(v, x) for k, x in c.items()}
    assert min(mses, key=mses.get) == "p99.99"
    assert mses["p90"] > mses["p95"] > mses["p99"] > mses["max"]


def test_scale_owners_v3():
    layers = build("tiny-yolov3-voc", rng=0).layers
    owner = scale_owners(layers)
    assert owner[1] == 0 and owner[11] == 10  # maxpool : échelle de la conv
    assert owner[17] == 13                     # route simple
    assert owner[19] == 18                     # upsample
    assert owner[20] == 18                     # route [19, 8] : première source


def test_unify_routes_v3():
    layers = build("tiny-yolov3-voc", rng=0).layers
    scales = {i: 0.1 for i, layer in enumerate(layers) if layer["type"] == "conv"}
    scales[8], scales[18] = 0.05, 0.07
    out, groups = unify_routes(layers, scales)
    assert groups == [[8, 18]]
    assert out[8] == out[18] == 0.07
    assert all(out[i] == 0.1 for i in scales if i not in (8, 18))


def test_choose_scales_route_constraint():
    rng = np.random.default_rng(3)
    fused = fuse_network(build("tiny-yolov3-voc", rng=4), dtype=np.float32)
    stats = ActStats(fused, per_image=256, rng=0)
    stats.update(fused.forward(rng.uniform(0, 1, (2, 3, 64, 64)).astype(np.float32),
                               train=False, all_outputs=True))
    scales, rows, groups = choose_scales(fused, stats)
    assert groups == [[8, 18]] and scales[8] == scales[18]
    assert scales[15] == scales[22] == HEAD_SCALE  # têtes : échelle fixe
    assert len(rows) == 13
