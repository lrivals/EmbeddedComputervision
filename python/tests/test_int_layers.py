"""T4.3 : couches entières (§9.3) — arrondi, leaky, conv int32, maxpool, upsample."""

import itertools

import numpy as np
import pytest

from yolo.layers.pool import maxpool_forward
from yolo.layers.upsample import upsample_forward
from yolo.quant import int_layers as il


def test_rshift_round_table():
    # (x + 2^(n−1)) >> n, décalage arithmétique : demis vers +∞, y compris négatifs.
    x = np.array([-12, -11, -10, -9, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    assert il.rshift_round(x, 2).tolist() == [-3, -3, -2, -2, -1, -1, -1, -1, 0, 0,
                                             0, 0, 1, 1, 1, 1, 2]
    # Égal à ⌊x/2ⁿ + ½⌋ sur une plage large.
    v = np.arange(-5000, 5000)
    for n in (1, 3, 7, 12):
        assert np.array_equal(il.rshift_round(v, n), np.floor(v / 2**n + 0.5).astype(int))


def test_rshift_round_int64_range():
    # acc · M0 < 2⁶² : pas de débordement de la somme avec 2³⁰.
    big = np.array([(1 << 62) - 1, -(1 << 62)], dtype=np.int64)
    assert il.rshift_round(big, 31).tolist() == [1 << 31, -(1 << 31)]


def test_leaky_int():
    y = np.array([-1000, -128, -100, -10, -1, 0, 1, 50])
    # y ≤ 0 : ⌊13 y / 128 + ½⌋ (−1,015 → −1 ; −10,156 → −10 ; −0,1 → 0)
    assert il.leaky_int(y).tolist() == [-102, -13, -10, -1, 0, 0, 1, 50]
    # Sans biais : sur une plage symétrique de négatifs, l'erreur moyenne à 13y/128 est ~0.
    v = np.arange(-20000, 1)
    assert abs(np.mean(il.leaky_int(v) - v * 13 / 128)) < 0.01
    assert il.LEAKY_MUL / 2**il.LEAKY_SHIFT == pytest.approx(0.1016, abs=1e-4)


def _naive_conv_int(qx, qW, qb):
    n_, c_, h, w = qx.shape
    f_, _, k, _ = qW.shape
    p = k // 2
    xp = np.pad(qx.astype(np.int64), ((0, 0), (0, 0), (p, p), (p, p)))
    acc = np.zeros((n_, f_, h, w), dtype=np.int64)
    for n, f, i, j in itertools.product(range(n_), range(f_), range(h), range(w)):
        acc[n, f, i, j] = int(np.sum(qW[f].astype(np.int64) * xp[n, :, i:i + k, j:j + k])) \
            + int(qb[f])
    return acc


@pytest.mark.parametrize("k", [1, 3])
def test_conv_acc_matches_naive_both_engines(k):
    rng = np.random.default_rng(k)
    qx = rng.integers(-127, 128, (2, 5, 7, 6))
    qW = rng.integers(-127, 128, (4, 5, k, k)).astype(np.int8)
    qb = rng.integers(-100000, 100000, 4).astype(np.int32)
    ref = _naive_conv_int(qx, qW, qb)
    for engine in il.ENGINES:
        acc = il.conv_acc(qx, qW, qb, engine)
        assert acc.dtype == np.int64 and np.array_equal(acc, ref), engine


def test_conv_int_engines_bit_exact_worst_case():
    # Pire cas de la couche 13 de Tiny-YOLOv2 : 9 × 1024 termes ±127·±127.
    rng = np.random.default_rng(5)
    qx = np.full((1, 1024, 3, 3), 127)
    qx[:, ::2] = -127
    qW = np.where(rng.random((3, 1024, 3, 3)) < 0.5, -127, 127).astype(np.int8)
    qW[0] = np.where(qx[0, :, :, :] > 0, 127, -127)[None]  # tous les produits = +127²
    qb = np.zeros(3, np.int32)
    a64 = il.conv_acc(qx, qW, qb, "int64")
    af = il.conv_acc(qx, qW, qb, "f64")
    assert np.array_equal(a64, af)
    assert a64[0, 0, 1, 1] == 9 * 1024 * 127 * 127
    # §9.3 : 16 + ⌈log2 9216⌉ = 30 bits au pire ; la valeur exacte tient sous 2³⁰.
    assert 9 * 1024 * 127 * 127 < 2**30
    M0 = rng.integers(1, 2**31, 3)
    y64, _ = il.conv_int(qx, qW, qb, M0, 31, "leaky", "int64")
    yf, _ = il.conv_int(qx, qW, qb, M0, 31, "leaky", "f64")
    assert np.array_equal(y64, yf)


def test_conv_acc_overflow_detected():
    qx = np.full((1, 1, 1, 1), 127)
    qW = np.full((1, 1, 1, 1), 127, np.int8)
    with pytest.raises(OverflowError):
        il.conv_acc(qx, qW, np.array([2**31 - 100], np.int64))


def test_conv_int_requant_leaky_clip():
    # 1×1, un canal : y = clip(leaky((acc·M0 + 2ⁿ⁻¹) >> n))
    qx = np.array([-3000, -40, -1, 0, 7, 3000]).reshape(1, 1, 1, 6)
    qW = np.ones((1, 1, 1, 1), np.int8)
    y, acc = il.conv_int(qx, qW, np.zeros(1, np.int32), np.array([1 << 29]), 31, "leaky")
    # acc / 4 arrondi (−0,25 → 0), puis leaky 13/128 arrondie sur les négatifs, saturation
    assert y.ravel().tolist() == [-76, -1, 0, 0, 2, 127]
    y, _ = il.conv_int(qx, qW, np.zeros(1, np.int32), np.array([1 << 29]), 31, "linear")
    assert y.ravel().tolist() == [-127, -10, 0, 0, 2, 127]


@pytest.mark.parametrize("s", [1, 2])
def test_maxpool_int_matches_float(s):
    rng = np.random.default_rng(s)
    q = rng.integers(-127, 128, (2, 3, 13 if s == 1 else 12, 13 if s == 1 else 12))
    y = il.maxpool_int(q, k=2, s=s)
    ref, _ = maxpool_forward(q.astype(np.float64), k=2, s=s)
    assert y.dtype == np.int64 and np.array_equal(y, ref)
    if s == 1:
        assert y.shape == q.shape


def test_upsample_and_route_int():
    q = np.arange(-8, 10).reshape(1, 2, 3, 3)
    assert np.array_equal(il.upsample_int(q), upsample_forward(q.astype(float))[0])
    r = il.route_int([q, q[:, :1]])
    assert r.shape == (1, 3, 3, 3) and np.array_equal(r[:, 2], q[:, 0])
