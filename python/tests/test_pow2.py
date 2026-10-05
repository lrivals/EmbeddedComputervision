"""T9.2.1 : niveaux REQ-YOLO (équidistants, mixed powers-of-two), projection, décalages."""

import numpy as np
import pytest

from yolo.quant.pow2 import (levels, project, project_int, shift_add_codes, shift_add_mul)


def test_mixed6_levels():
    m = levels("mixed6")
    assert m[0] == 0 and m.max() == 96 and len(m) <= 32  # 1 signe + 5 bits de magnitude
    for v in m[1:]:
        bits = [b for b in range(7) if v >> b & 1]
        assert len(bits) in (1, 2) and (len(bits) == 1 or bits[1] - bits[0] <= 3)
    assert list(levels("pot5")) == [0, 1, 2, 4, 8, 16, 32, 64]
    assert levels("uniform6").max() == 31


@pytest.mark.parametrize("kind", ["mixed6", "pot5", "uniform6", "uniform4"])
def test_projection_is_nearest_level(kind):
    m = levels(kind)
    v = np.random.default_rng(0).uniform(-110, 110, 5000)
    q = project_int(v, m)
    allowed = np.concatenate([-m[::-1], m])
    dist = np.abs(v[:, None] - allowed[None, :]).min(axis=1)
    assert np.allclose(np.abs(v - q), dist)
    assert set(np.abs(q)) <= set(m)


def test_project_per_channel_scale_beats_max_scale():
    W = np.random.default_rng(1).standard_normal((8, 4, 3, 3))
    qW, sw = project(W, "mixed6")
    assert set(np.abs(qW).ravel()) <= set(levels("mixed6"))
    err = ((W - qW * sw[:, None, None, None]) ** 2).sum()
    s0 = np.abs(W).reshape(8, -1).max(axis=1) / 96
    q0 = project_int(W / s0[:, None, None, None], levels("mixed6"))
    assert err <= ((W - q0 * s0[:, None, None, None]) ** 2).sum()


def test_two_shifts_and_one_add_equal_product():
    m = levels("mixed6")
    q = np.concatenate([-m, m])
    x = np.arange(-128, 128)
    got = shift_add_mul(x[None, :], q[:, None])
    assert np.array_equal(got, q[:, None] * x[None, :])
    sign, a, k = shift_add_codes(q)
    assert a.max() <= 6 and k.max() <= 3  # 3 bits primaires, 2 secondaires
