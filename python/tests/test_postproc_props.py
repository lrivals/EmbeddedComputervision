"""T28.6: properties of the hardware post-processing (no-sort NMS, 256 slots)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from yolo.infer import hw_postproc as hp

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("equivalence", ROOT / "tools" / "equivalence.py")
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)


def box(x, score, cls=0, w=100):
    return (x, 0, x + w, w, score, cls)


def test_capacity_and_sorted_order_on_random_streams():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 300))
        x = rng.integers(0, 3000, n)
        cands = [box(int(x[k]), int(rng.integers(1, 1000)), int(rng.integers(0, 3)))
                 for k in range(n)]
        eq.postproc_props(cands, perms=3)


def test_overflow_counted_when_slots_run_out():
    cands = [box(1000 * k, 10) for k in range(300)]  # no overlap: 300 slots needed
    s = eq.postproc_props(cands)
    assert s["slots"] == 300
    out, ov = hp.nms_nosort(cands, 9, 20)
    assert len(out) == 256 and ov == 44


def test_raw_order_matters_sorted_does_not():
    """The chain a < b < c of test_hw_postproc: the raw stream depends on the order, the
    stream sorted by score gives the classic NMS."""
    a, b, c = box(0, 100), box(30, 200), box(60, 300)
    assert hp.nms_nosort([a, b, c], 9, 20)[0] == [c]
    assert sorted(hp.nms_nosort([c, b, a], 9, 20)[0]) == [a, c]
    assert sorted(eq.greedy_nms([a, b, c], 9, 20)) == [a, c]
    s = eq.postproc_props([a, b, c], perms=0)
    assert s["differs_from_greedy"]


@pytest.mark.parametrize("net", eq.POST_NETS)
def test_properties_on_dumps(net):
    if not (ROOT / "model" / net / "manifest.json").exists():
        pytest.skip("export absent (make export)")
    for img in eq.post_images(net)[:3]:
        heads, _ = eq.head_data(net, img)
        eq.postproc_props(eq.candidates(heads, 0.005), perms=2)
