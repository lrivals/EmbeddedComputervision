"""T8.1/T8.3 : modèle de cycles Python == compteurs C-sim du noyau HLS (count_cycles)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
_spec = importlib.util.spec_from_file_location("perf_model", ROOT / "tools" / "perf_model.py")
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)

CYCLES_CSV = ROOT / "build" / "hls" / "cycles_conv.csv"


def _layers(net):
    path = ROOT / "model" / net / "manifest.json"
    if not path.exists():
        pytest.skip(f"export absent : {path} (make export)")
    return pm.conv_layers(path)


@pytest.mark.parametrize("net, overlapped, compute", [
    # Totaux de results/hls_report.md (tb_conv, carte avant pooling écrite partout).
    ("tiny-yolov2-voc", 42_093_912, 8_225_737),
    ("tiny-yolov3-coco", 41_432_106, 7_305_025),
])
def test_totals_match_hls_report(net, overlapped, compute):
    layers = _layers(net)
    cyc = [pm.layer_cycles(d, prepool_all=True) for d in layers]
    assert sum(c["overlapped"] for c in cyc) == overlapped
    assert sum(c["compute"] for c in cyc) == compute


@pytest.mark.skipif(not CYCLES_CSV.exists(), reason="make hls-cycles")
def test_every_layer_matches_csim():
    assert pm.check(CYCLES_CSV) == 0


def test_macs_spec():
    # §3 : Tiny-YOLOv2 = 3,49 GMAC.
    assert sum(pm.macs(d) for d in _layers("tiny-yolov2-voc")) / 1e9 == pytest.approx(3.49, abs=0.01)


def test_scenarios_ordered():
    # Ports plus larges ou canaux valides seulement : jamais plus de cycles ; borne calcul en bas.
    layers = _layers("tiny-yolov2-voc")
    base = pm.net_cycles(layers)
    assert pm.net_cycles(layers, trim=True) <= base
    assert pm.net_cycles(layers, width=16, trim=True) <= pm.net_cycles(layers, width=8) < base
    comp = sum(pm.layer_cycles(d)["compute"] for d in layers)
    assert comp <= pm.net_cycles(layers, width=1 << 20, trim=True)
    # Seule L08 de Tiny-YOLOv3 écrit sa carte avant pooling dans le programme réel.
    v3 = _layers("tiny-yolov3-coco")
    assert [d["layer"] for d in v3 if d["prepool"]] == [8]


def test_m10_kernel():
    # T10.1-T10.4 : chaque piste du noyau actuel réduit les cycles ; avec width = 1 et sans les
    # pistes, le modèle redonne le noyau M6 (test_totals_match_hls_report).
    assert pm.KERNEL["width"] == 8 and pm.KERNEL["trim"] and pm.KERNEL["requant"] == 8
    layers = _layers("tiny-yolov2-voc")
    m6 = pm.net_cycles(layers, **pm.M6)
    p64 = pm.net_cycles(layers, **pm.P64)
    rq = pm.net_cycles(layers, **dict(pm.P64, trim=True, requant=8))
    fold = pm.net_cycles(layers, **dict(pm.P64, trim=True, requant=8, fold=True))
    kern = sum(pm.kernel_cycles(d)["overlapped"] for d in layers)
    assert kern < fold < rq < p64 < m6
    # Pliage : seule L00 (cin·k = 9 ≤ Tn) est pliée ; tuile 14 : convs poolées en stride 2.
    tiling = [pm.layer_tiling(d, fold=True, tile_pool=14) for d in layers]
    assert [d["layer"] for d, t in zip(layers, tiling) if t[2]] == [0]
    assert [d["layer"] for d, t in zip(layers, tiling) if t[0] == 14] == [0, 2, 4, 6, 8]
