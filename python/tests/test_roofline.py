"""T5.6 : explorateur roofline — ordre de grandeur du §10.2 et cohérence du modèle."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("roofline", ROOT / "tools" / "roofline.py")
roofline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roofline)

V2 = roofline.conv_layers(roofline.NETWORKS["tiny-yolov2-voc"])


def test_order_of_magnitude_spec():
    # §10.2 : Tm = Tn = 16 → 3,49 GMAC / 256 ≈ 13,6 M cycles ≈ 68 ms à 200 MHz.
    assert roofline.total_macs(V2) / 1e9 == pytest.approx(3.49, abs=0.01)
    assert roofline.ideal_time_s(V2, 16, 16, 200e6) * 1e3 == pytest.approx(68, abs=0.5)


def test_tiled_cycles_bounded_by_ideal():
    # Le tuilage ajoute des cycles (bords, cin = 3), jamais moins que l'idéal ; une bande
    # passante infinie rend le temps égal aux cycles.
    e = roofline.evaluate(V2, 16, 16, 13, 13, 200e6, float("inf"))
    ideal = roofline.total_macs(V2) / 256
    assert ideal <= e["cycles"] < 1.15 * ideal
    assert e["time_s"] == pytest.approx(e["cycles"] / 200e6)
    assert e["attainable"] == pytest.approx(e["comp_roof"])


def test_buffer_formulas():
    assert roofline.buffer_sizes(16, 16, 13, 13) == (16 * 15 * 15, 16 * 16 * 9, 16 * 13 * 13)
    assert roofline.buffer_sizes(4, 4, 3, 3, k=3, s=2) == (4 * 7 * 7, 4 * 4 * 9, 4 * 3 * 3)


def test_bandwidth_bound():
    # Bande passante très faible : la performance suit CTC × BW.
    bw = 1e6
    e = roofline.evaluate(V2, 16, 16, 13, 13, 200e6, bw)
    assert e["attainable"] == pytest.approx(e["ctc"] * bw, rel=1e-6)


@pytest.mark.parametrize("path", sorted((ROOT / "hw" / "boards").glob("*.yaml")))
def test_boards_have_feasible_points(path):
    pytest.importorskip("yaml")
    board = roofline.load_board(path)
    points = roofline.explore(V2, board)
    assert points, f"aucune tuile ne tient sur {board['name']}"
    best = points[0]
    assert best["dsp"] <= roofline.UTIL * board["dsp"]
    assert best["bram18"] <= roofline.UTIL * board["bram18"]
