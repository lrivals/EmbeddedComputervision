"""T23.5: integer-exact Winograd F(2×2, 3×3) (tools/winograd.py) == direct convolution."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("winograd", ROOT / "tools" / "winograd.py")
wg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wg)


def test_exact_on_int8_tiles():
    wg.self_check(np.random.default_rng(1), n=300)


def test_widths():
    u, v, p = wg.bounds()
    assert (wg.bits(u), wg.bits(v), wg.bits(p)) == (12, 10, 21)
    assert wg.bits(u) + wg.bits(p) > 27  # no packing of two Winograd products in A
