"""Cross-dataset analysis of the statistics (tools/notebooks/jeux.py, tools/figures/jeux.py):
parsing of the `_stats` notebook outputs, derived indicators, figures."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.notebooks import jeux as J  # noqa: E402

SYNTH = """\
| partie | images | objets | difficult | images vides | objets/image (méd. / max) | petits < 32² à 416×416 | plus petits qu'une cellule | cibles perdues | images > 256 / > 1 024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 1 000 | 12 000 | 0 | 2 | 10 / 300 | 60,0 % | 13x13 40,0 % · 26x26 10,0 % | 20,00 % | 1 / 0 |
| ensemble | 1 500 | 18 000 | 0 | 3 | 10 / 300 | 61,5 % | 13x13 41,0 % · 26x26 11,5 % | 21,50 % | 1 / 0 |
"""
CLASSES = """\
| classe | train : objets (images) | test : objets (images) | côté médian (px, letterbox) | petits |
|---|---|---|---|---|
| car | 11 000 (900) | 5 000 (450) | 20 | 70 % |
| bus | 1 000 (300) | 1 000 (200) | 40 | 30 % |

Rapport classe la plus / la moins fréquente : 8.
"""
GEOM = """\
| repère | petits | moyens | grands | côté médian (px) | plus petits qu'une cellule | hauteur < 8 px |
|---|---|---|---|---|---|---|
| image d'origine | 20,0 % | 50,0 % | 30,0 % | 60 | — | — |
| letterbox {s} | 61,5 % | 30,0 % | 8,5 % | 22 | 13x13 41,0 % · 26x26 11,5 % | 5,0 % |
| stretch {s} | 40,0 % | 45,0 % | 15,0 % | 35 | 13x13 20,0 % · 26x26 2,0 % | 0,5 % |

18 000 objets non-difficult ; pas des têtes : 13x13 → 32 px, 26x26 → 16 px.
"""
DENS = """\
| partie | objets/image moy. | méd. | max | > 256 / > 1 024 | cibles perdues | par tête (perdues / objets) | images touchées |
|---|---|---|---|---|---|---|---|
| ensemble | 12,0 | 10 | 300 | 1 / 0 | 3 870 / 18 000 (21,50 %) | 13x13 : 20 / 2 000 · 26x26 : 250 / 12 400 | 900 |
"""
IMG = """\
| partie | résolutions (3 premières) | distinctes | pixels sur | canaux | moyenne par canal | écart type par canal | luminance moy. | niveaux 8 bits → INT8 |
|---|---|---|---|---|---|---|---|---|
| ensemble | 1280×720 (1 400), 640×480 (100) | 2 | 200 images | 3 | 90,0 / 95,0 / 100,0 | 60,0 / 61,0 / 62,0 | 94,5 | 256 → 128 |
"""
ANCH = """\
| ancres | w,h (px à 416×416) | IoU moyenne |
|---|---|---|
| VOC (tiny-yolov2-voc) | `35,38  109,141  212,364  301,164  532,337` | 0,2500 |
| COCO (tiny-yolov3-coco) | `10,14  23,27  37,58  81,82  135,169  344,319` | 0,5000 |
| cfg du jeu (tiny-yolov3-coco) | `10,14  23,27  37,58  81,82  135,169  344,319` | 0,5000 |
| k-means k = 6 (graine 0) | `5,6  10,12  20,22  40,30  60,70  100,90` | 0,7000 |
"""
QUAL = """\
| partie | retirées (make_sample) | rognées | côté < 2 px (orig. / entrée) | doublons (IoU > 0,95) | régions retirées par le chargeur |
|---|---|---|---|---|---|
| ensemble | — | 180 | 1 / 36 | 2 | DontCare 1 200 |
"""
ECART = """\
| grandeur (train ↔ test) | variation totale |
|---|---|
| classes | 0,050 |
| per_image | 0,120 |

| split | classes du modèle | objets couverts | part | évalués (non-difficult) |
|---|---|---|---|---|
| train | VOC | 9 000 | 75,0 % | 9 000 |
| test | VOC | 4 500 | 75,0 % | 4 500 |
| train | COCO | 12 000 | 100,0 % | 12 000 |
"""


def _md(text):
    return {"output_type": "display_data", "metadata": {},
            "data": {"text/markdown": [text], "text/plain": ["<Markdown>"]}}


def _stream(text):
    return {"output_type": "stream", "name": "stdout", "text": [text]}


def _notebook(path, outputs_per_cell, error=None):
    cells = []
    for k, outs in enumerate(outputs_per_cell):
        cells.append({"cell_type": "code", "execution_count": k + 1, "id": f"c{k:02d}",
                      "metadata": {}, "source": ["pass"], "outputs": outs})
    if error:
        cells.append({"cell_type": "code", "execution_count": len(cells) + 1, "id": "cerr",
                      "metadata": {}, "source": ["pass"],
                      "outputs": [{"output_type": "error", "ename": "CalledProcessError",
                                   "evalue": error, "traceback": []}]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4,
                                "nbformat_minor": 5}))
    return path


def _toy(nb_dir, ds="visdrone", native=False):
    geo = [_md(GEOM.format(s="416×416"))]
    if native:  # KITTI: a second geometry table at its own input size
        geo.append(_md(GEOM.format(s="640×192").replace("13x13", "6x20")
                       .replace("26x26", "12x40")))
    return _notebook(nb_dir / ds / f"{ds}_stats.ipynb", [
        [_stream("cfg build/m11/cfg/x.cfg absente (tools/m11.sh x-prep) : ancres\n"),
         _stream("palier : annotations M, pixels M ; entrée 416 (letterbox) ; cfg tiny-yolov3-coco\n")],
        [_md(SYNTH), _md(CLASSES)],  # two tables back to back in one cell
        geo, [_md(DENS)], [_md(IMG)], [_md(ANCH)], [_md(QUAL)], [_md(ECART)],
        [_stream("sorties : build/notebooks/x/stats/ ; révision abc1234\n"), _md(SYNTH)],
    ])


def test_french_numbers():
    assert J.num("40 570") == 40570
    assert J.num("70,9 %") == 70.9
    assert J.num("0,6029") == pytest.approx(0.6029)
    assert J.num("—") is None
    assert J._ints("2 146 / 56 612") == [2146, 56612]
    assert J._heads("13x13 57,7 % · 26x26 25,5 %") == {"13x13": 57.7, "26x26": 25.5}


def test_parse_stats_tables(tmp_path):
    d = J.parse_stats(_toy(tmp_path, native=True))
    assert d["rev"] == "abc1234" and d["net"] == "tiny-yolov3-coco" and d["fallback"]
    assert d["family"] == "aerial"
    s = d["synthese"]
    assert (s["images"], s["objects"], s["median"], s["max"]) == (1500, 18000, 10, 300)
    assert s["cell"] == {"13x13": 41.0, "26x26": 11.5} and s["over256"] == 1
    assert s["splits"] == ["train"]
    c = d["classes"]
    assert c["names"] == ["car", "bus"] and c["counts"] == [16000, 2000] and c["ratio"] == 8
    assert 0 < c["entropy"] < 1
    assert d["geometrie"]["letterbox"]["cell"]["26x26"] == 11.5
    assert d["geometrie_native"]["size"] == "640×192"
    dn = d["densite"]
    # 18 000 objects, 14 400 assigned to a head, 270 collisions → 3 600 orphans.
    assert (dn["orphan"], dn["collide"]) == (3600, 270)
    assert dn["orphan"] + dn["collide"] == dn["lost"]
    im = d["images"]
    assert im["main"] == (1280, 720) and im["scale"] == pytest.approx(416 / 1280)
    assert im["mean"] == [90.0, 95.0, 100.0] and im["luminance"] == 94.5
    a = d["ancres"]
    assert a["iou"]["cfg"] == 0.5 and a["best_k"] == 0.7 and a["gain"] == pytest.approx(0.2)
    assert a["wh"]["coco"][0] == (10, 14)
    q = d["qualite"]
    assert q["removed"] is None and q["removed_pct"] is None
    assert q["clipped_pct"] == pytest.approx(1.0) and q["regions"] == {"DontCare": 1200}
    assert d["ecart"] == {"classes": 0.05, "per_image": 0.12}
    assert d["couverture"]["VOC"] == pytest.approx(75.0) and d["couverture"]["COCO"] == 100


def test_load_all_keeps_partial_and_reports_errors(tmp_path):
    _toy(tmp_path, "visdrone")
    _toy(tmp_path, "voc")
    _notebook(tmp_path / "coco" / "coco_stats.ipynb", [[]], error="data/coco absent")
    _notebook(tmp_path / "kitti" / "kitti_stats.ipynb", [[], []])  # never run
    data, pending = J.load_all(tmp_path)
    assert list(data) == ["voc", "visdrone"]  # ORDER: ground views, then drone views
    assert set(pending) == {"coco", "kitti"} and "data/coco absent" in pending["coco"]
    assert "not executed" in pending["kitti"]
    obs = J.observations(data)
    assert set(obs) >= {"taille", "densite", "difficulte"}
    assert "VisDrone" in obs["difficulte"]
    text = J.conclusions(data, pending)
    assert "Not analysed" in text and "coco" in text
    assert "| VisDrone |" in J.summary_table(data)
    sc = J.difficulty(data)
    assert set(sc["score"]) == {"voc", "visdrone"}


def test_figures_on_toy_data(tmp_path):
    pytest.importorskip("matplotlib")
    from tools.figures import jeux as FJ

    _toy(tmp_path / "nb", "visdrone")
    _toy(tmp_path / "nb", "voc")
    data, _ = J.load_all(tmp_path / "nb")
    for key, fn in FJ.PLOTS.items():
        paths = fn(data, tmp_path / "out")
        assert paths and all(p.exists() and p.stat().st_size > 1000 for p in paths), key


@pytest.mark.skipif(not (ROOT / "notebooks" / "kitti" / "kitti_stats.ipynb").exists(),
                    reason="notebook absent")
def test_versioned_kitti_notebook():
    p = ROOT / "notebooks" / "kitti" / "kitti_stats.ipynb"
    if not J._executed(p):
        pytest.skip("kitti_stats not executed")
    d = J.parse_stats(p)
    assert d["synthese"]["images"] > 7000 and len(d["classes"]["names"]) == 8
    assert d["densite"]["orphan"] + d["densite"]["collide"] == d["densite"]["lost"]


def test_english_tables_parse_as_french(tmp_path):
    """T19.6: tables written in English (decimal point, comma thousands) give the same
    values as the French tables of the notebooks executed before."""
    en = (SYNTH.replace("| partie | images | objets | difficult | images vides | "
                        "objets/image (méd. / max) | petits < 32² à 416×416 | "
                        "plus petits qu'une cellule | cibles perdues | images > 256 / > 1 024 |",
                        "| part | images | objects | difficult | empty images | "
                        "objects/image (med. / max) | small < 32² at 416×416 | "
                        "smaller than a cell | lost targets | images > 256 / > 1,024 |")
          .replace("1 000", "1,000").replace("12 000", "12,000").replace("1 500", "1,500")
          .replace("18 000", "18,000").replace(",0 %", ".0 %").replace(",5 %", ".5 %")
          .replace("20,00 %", "20.00 %").replace("21,50 %", "21.50 %")
          .replace("| ensemble |", "| all |"))
    dens = ("| part | objects/image mean | med. | max | > 256 / > 1,024 | lost targets | "
            "per head (lost / objects) | images hit |\n|---|---|---|---|---|---|---|---|\n"
            "| all | 12.0 | 10 | 300 | 1 / 0 | 3,870 / 18,000 (21.50 %) | "
            "13x13: 20 / 2,000 · 26x26: 250 / 12,400 | 900 |\n")
    path = _notebook(tmp_path / "visdrone" / "visdrone_stats.ipynb", [
        [_stream("outputs: build/notebooks/x/stats/; revision abc1234\n"), _md(en)],
        [_md(dens)]])
    d = J.parse_stats(path)
    fr = J.parse_stats(_toy(tmp_path / "fr"))
    assert d["rev"] == "abc1234" and d["synthese"] == fr["synthese"]
    assert d["densite"] == fr["densite"]
