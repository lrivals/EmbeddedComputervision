"""Statistiques des jeux (M16) : tools/data_stats.py et tools/figures/donnees.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import data_stats as DS  # noqa: E402
from yolo.data import datasets as D  # noqa: E402
from yolo.data.letterbox import boxes_to_letterbox  # noqa: E402
from yolo.data.targets import anchor_ref, build_targets  # noqa: E402

CLASSES = ("a", "b", "c")


def _sample(i, corners, labels, w=640, h=320, difficult=None, image="x.jpg", **kw):
    return D.make_sample(i, image, w, h, corners, labels,
                         difficult if difficult is not None else [False] * len(labels), **kw)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Jeu `fake` à trois classes, deux splits, images réelles (pixels lisibles)."""
    from PIL import Image

    rng = np.random.default_rng(0)
    paths = []
    for k in range(6):
        p = tmp_path / f"{k}.png"
        Image.fromarray(rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    train = [_sample("t0", [[10, 10, 100, 100], [10, 10, 100, 100]], [0, 0], image=paths[0]),
             _sample("t1", [[0, 0, 20, 8], [300, 100, 600, 300]], [1, 2], image=paths[1]),
             _sample("t2", [], [], image=paths[2]),
             _sample("t3", [[50, 50, 60, 60]], [0], difficult=[True], image=paths[3])]
    test = [_sample("v0", [[5, 5, 400, 300]], [2], image=paths[4]),
            _sample("v1", [[30, 30, 31, 200], [0, 0, 640, 320]], [1, 0], image=paths[5])]
    parts = {"tr": train, "te": test}
    ds = D.Dataset(CLASSES, lambda root, split: list(parts[split]), "fake", "te", "tr", "tr")
    monkeypatch.setitem(D.DATASETS, "fake", ds)
    return parts


def _ctx(**kw):
    return DS.Ctx("fake", **{"net": "tiny-yolov3-voc", **kw})


# ---------------------------------------------------------------------------- T16.5

def test_comptes(fake):
    c = DS.comptes(fake["tr"], _ctx())
    assert (c["images"], c["objects"], c["objects_easy"]) == (4, 5, 4)
    assert c["empty_images"] == 1 and c["difficult"] == 1
    assert c["per_class"].tolist() == [3, 1, 1]
    assert c["images_with"].tolist() == [2, 1, 1]
    assert c["cooc"][1, 2] == 1 and c["cooc"][0, 1] == 0 and c["cooc"][0, 0] == 2
    assert c["imbalance"] == 3.0


def test_couverture_egale_a_l_evaluation_hors_domaine():
    """T16.11 : objets évalués = non-difficult après `remap` par `eval_view` (eval_voc.py)."""
    s = [_sample("k", [[0, 0, 50, 50], [0, 0, 60, 60], [1, 1, 9, 9]], [0, 5, 7],
                 difficult=[False, False, True])]
    cov = DS.coverage(s, "kitti")
    assert cov["voc"]["objects"] == 1 and cov["voc"]["evaluated"] == 1   # Car seule
    view = D.eval_view("kitti", 20)
    kept = D.remap(s, view.gt_lut)
    assert cov["voc"]["evaluated"] == sum(int((~x["difficult"]).sum()) for x in kept)
    assert cov["coco"]["objects"] == 1


# ---------------------------------------------------------------------------- T16.6

@pytest.mark.parametrize("size", [416, (192, 640)])
def test_pixels_d_entree_par_boxes_to_letterbox(fake, size):
    s = fake["tr"][1]
    got = DS.wh_pixels([s], size, "letterbox")
    sh, sw = (size, size) if isinstance(size, int) else size
    ref = boxes_to_letterbox(s["boxes"], s["width"], s["height"], size)[:, 2:] * (sw, sh)
    np.testing.assert_allclose(got, ref)
    np.testing.assert_allclose(DS.wh_pixels([s], size, "stretch"), s["boxes"][:, 2:] * (sw, sh))


def test_geometrie(fake):
    g = DS.geometrie(fake["tr"], _ctx())
    assert g["n"] == 4                      # difficult exclu
    assert sum(g["orig"]["coco"]) == 4
    # 640×320 → letterbox 416 : facteur 0,65 ; boîte 20×8 → 13×5,2 px, sous les deux cellules
    assert g["letterbox"]["fit_cell"]["26x26"] == pytest.approx(0.25)
    assert g["letterbox"]["h_under"]["8"] == pytest.approx(0.25)
    assert g["strides"] == {"13x13": 32.0, "26x26": 16.0}
    assert np.asarray(g["centers"]).sum() == 4


# ---------------------------------------------------------------------------- T16.7

def test_collisions_egales_a_build_targets(fake):
    ctx = _ctx()
    col = DS.collisions(fake["tr"], ctx)
    assert col["lost"] == 1 and col["per_class"]["lost"].tolist() == [1, 0, 0]
    sp = ctx.spec()
    bl = [DS.input_boxes(s, ctx.size, ctx.resize)[~s["difficult"]] for s in fake["tr"]]
    ll = [s["labels"][~s["difficult"]] for s in fake["tr"]]
    t = build_targets(bl, ll, sp["cfg"]["anchors"], sp["heads"], sp["grids"],
                      anchor_ref(sp["cfg"]))
    pos = sum(int(v["obj"].sum()) for v in t.values())
    assert col["lost"] == sum(len(b) for b in bl) - pos


def test_densite(fake):
    d = DS.densite(fake["tr"], _ctx())
    assert d["per_image"].tolist() == [1, 1, 2]   # 1 image vide, 1 à un objet, 2 à deux
    assert d["max"] == 2 and d["over"]["256"] == 0


# ---------------------------------------------------------------------------- T16.8

def test_pixels_echantillonnes_reproductibles(fake):
    a = DS.pixel_stats(fake["tr"], 3, seed=1)
    b = DS.pixel_stats(fake["tr"], 3, seed=1)
    assert a["ids"] == b["ids"] and a["images"] == 3 and a["channels"] == 3
    assert np.asarray(a["hist"]).sum() == 3 * 32 * 64 * 3
    assert a["levels"]["8bit"] <= 256 and a["levels"]["int8"] <= 128


# ---------------------------------------------------------------------------- T16.9

def test_ancres_egales_a_kmeans_anchors(fake):
    from tools.kmeans_anchors import dataset_wh
    from yolo.data.anchors import kmeans_anchors

    a = DS.ancres(fake["tr"], _ctx(seed=3))
    wh = dataset_wh(fake["tr"], 416)
    k = min(len(wh), 3)
    np.testing.assert_allclose(a["kmeans"][str(k)], np.round(kmeans_anchors(wh, k, rng=3), 2))
    assert a["boxes"] == 4


# ---------------------------------------------------------------------------- T16.10

def test_qualite(fake):
    q = DS.qualite(fake["tr"], _ctx())
    assert q["duplicates"] == 1 and not q["raw_checked"]
    assert q["suspects"][0]["id"] == "t0"
    q = DS.qualite(fake["te"], _ctx())
    assert q["degenerate_orig"] == 1            # boîte de 1 px de large


def test_regions_retirees_kitti(tmp_path):
    lab = tmp_path / "training" / "label_2"
    lab.mkdir(parents=True)
    (lab / "000001.txt").write_text(
        "Car 0 0 0 -10 10 100 100 0 0 0 0 0 0 0\nDontCare -1 -1 -10 1 1 5 5 0 0 0 0 0 0 0\n")
    boxes, labels = D.parse_kitti((lab / "000001.txt").read_text())
    s = D.make_sample("000001", tmp_path / "training" / "image_2" / "000001.png", 200, 200,
                      boxes, labels, [False])
    ctx = DS.Ctx("kitti", net="tiny-yolov3-voc", root=tmp_path)
    q = DS.qualite([s], ctx)
    assert q["regions"] == {"DontCare": 1} and q["clipped"] == 1 and q["removed"] == 0


# ---------------------------------------------------------------------------- T16.11

def test_ecart():
    a = {"comptes": {"per_class": [10, 0]}}
    b = {"comptes": {"per_class": [0, 5]}}
    assert DS.ecart(a, b)["classes"] == 1.0
    assert DS.ecart(a, a)["classes"] == 0.0


# ---------------------------------------------------------------------------- T16.2

def test_fiches():
    for name, ds in D.DATASETS.items():
        f = DS.FICHES[name]
        assert tuple(f["classes"]) == ds.classes, name
        for k in ("source", "version", "license", "sensor", "resolution", "official"):
            assert f[k], (name, k)


def test_split_images_releves():
    from tools.notebooks.matrice import SPLIT_IMAGES

    assert all(SPLIT_IMAGES[d] for d in D.DATASETS)
    for d, n in SPLIT_IMAGES.items():
        off = DS.FICHES[d]["official"].get(D.DATASETS[d].test)
        assert off is None or off[0] in (None, n), d


# ---------------------------------------------------------------------------- bout en bout

def test_compute_json_markdown_figures(fake, tmp_path):
    stats, parts = DS.compute("fake", sample=2, net="tiny-yolov3-voc", log=lambda _: None)
    assert set(stats["splits"]) == {"tr", "te"} and stats["groupes"] == {"train": "tr",
                                                                         "test": "te"}
    assert stats["ensemble"]["comptes"]["images"] == 6
    assert "ecart" in stats and "ancres" in stats
    stats["galerie"] = DS.gallery(stats, parts, tmp_path, 1, 0)
    assert len(stats["galerie"]["tr"]) == 1 and len(stats["galerie"]["classes"]) == 3
    path = DS.write(stats, tmp_path)
    data = json.loads(path.read_text())
    md = (tmp_path / "stats.md").read_text()
    assert "## Synthèse" in md and "## Densité et collisions" in md
    from tools.figures import donnees

    pngs = donnees.plot_all(data, tmp_path / "figures")
    assert len(pngs) >= 10 and all(p.stat().st_size > 0 for p in pngs)


def test_report_garde_le_texte_hors_marqueurs(fake, tmp_path):
    stats, _ = DS.compute("fake", net="tiny-yolov3-voc", log=lambda _: None)
    DS.write(stats, tmp_path / "s")
    doc = tmp_path / "doc.md"
    doc.write_text("# Titre\n\nRéponse écrite à la main.\n")
    DS.report(doc, [tmp_path / "s" / "stats.json"])
    DS.report(doc, [tmp_path / "s" / "stats.json"])
    text = doc.read_text()
    assert "Réponse écrite à la main." in text
    assert text.count("<!-- data_stats:fake -->") == 1
    assert text.count("<!-- data_stats:comparatif -->") == 1


# ---------------------------------------------------------------------------- VOC réel

DEVKIT = ROOT / "data" / "VOCdevkit"


@pytest.mark.skipif(not (DEVKIT / "VOC2012" / "ImageSets" / "Main" / "trainval.txt").exists(),
                    reason="VOCdevkit absent")
def test_comptes_voc_egaux_au_devkit():
    stats, _ = DS.compute("voc", only=("comptes",), log=lambda _: None)
    chk = DS.check_official(DS._py(stats))
    assert len(chk) == 3 and all(ok for *_, ok in chk)


def test_voc_stats_view(fake):
    """La vue de la figure voc_stats se lit depuis stats.json (groupes train et test)."""
    from tools.figures import donnees

    stats, _ = DS.compute("fake", only=("comptes", "geometrie", "densite"),
                          net="tiny-yolov3-voc", log=lambda _: None)
    view = donnees.voc_stats_view(DS._py(stats))
    assert view["trainval 07+12"]["per_class"].sum() == 4
    assert view["test 2007"]["images"] == 2
