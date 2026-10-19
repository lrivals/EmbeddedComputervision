"""T23.0: two int8 products per DSP48E2 multiplier (packing, sign borrow)."""

import numpy as np
import pytest

from yolo.quant.pack import (A_BITS, P_BITS, QMAX, S, depth, groups, lanes, max_shift,
                             pack2, product_range, signed_fits, table, unpack2)

I8 = np.arange(-128, 128, dtype=np.int64)


def test_all_int8_triples():
    # 2^24 triples (w_a, w_b, x), one value of w_a at a time.
    wb, x = np.meshgrid(I8, I8, indexing="ij")
    for wa in I8:
        a = pack2(wa, wb)
        assert signed_fits(a, A_BITS).all()
        p = a * x
        assert signed_fits(p, A_BITS + 18).all()
        hi, lo = unpack2(p)
        assert (hi == wa * x).all() and (lo == wb * x).all()


def test_shift_is_the_largest_for_int8():
    assert max_shift(2, -128, 127) == S == 18
    assert not signed_fits(pack2(-128, -1, 19), A_BITS)
    assert max_shift(2, -QMAX, QMAX) == 19  # contract: −128 never produced


@pytest.mark.parametrize("q, n", [(128, 7), (QMAX, 8)])
def test_depth_is_tight(q, n):
    assert depth(S, *product_range(-q, 127, -q, 127)) == n

    def exact(k, wb, x):
        wb_, x_ = np.full(k, wb, dtype=np.int64), np.full(k, x, dtype=np.int64)
        hi, lo = unpack2((pack2(127, wb_) * x_).sum())
        return lo == (wb_ * x_).sum() and hi == 127 * x_.sum()

    # Low lane: N extreme products pass on both sides, N + 1 positive ones (−q·−q) overflow.
    assert exact(n, -q, -q) and exact(n, -q, 127)
    assert not exact(n + 1, -q, -q)


def test_random_partial_sums():
    rng = np.random.default_rng(0)
    for n, q in ((7, 128), (8, QMAX)):
        wa = rng.integers(-q, q, (10000, n), endpoint=q == QMAX)
        wb = rng.integers(-q, q, (10000, n), endpoint=q == QMAX)
        x = rng.integers(-q, q, (10000, n), endpoint=q == QMAX)
        p = (pack2(wa, wb) * x).sum(axis=1)
        assert signed_fits(p, P_BITS).all()
        hi, lo = unpack2(p)
        assert (hi == (wa * x).sum(axis=1)).all() and (lo == (wb * x).sum(axis=1)).all()


def test_budget_table():
    rows = {r["case"]: r for r in table()}
    assert rows["int8 × int8, contract ±127"]["groups"] == [8, 8, 8]
    assert rows["int8 × int8, any value"]["groups"] == [7, 7, 7, 3]
    assert lanes(2, -8, 7, -QMAX, QMAX, S)["n"] == 129  # 4 bits: the whole Tn tree at once
    assert lanes(3, -8, 7, -QMAX, QMAX)["n"] == 1
    assert lanes(4, -8, 7, -QMAX, QMAX)["n"] == 0
    assert lanes(2, -QMAX, QMAX, -QMAX, QMAX, a_bits=25)["n"] == 4  # DSP48E1, s = 17
    assert groups(24, 8) == [8, 8, 8] and groups(5, 8) == [5]
