"""Analyse des balayages (tools/notebooks/balayages.py, tools/figures/balayages.py) :
lecture des sorties de notebooks `_sweep` / `_infer`, statistiques, figures."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.notebooks import balayages as B  # noqa: E402


def _log(n_images, iters=20, start=100.0):
    lines = [f"{n_images} images, {max(1, n_images // 8)} lots par époque"]
    for it in range(1, iters + 1):
        v = start / it
        lines.append(f"it {it:6d}  lr 1.00e-03  perte {v:8.3f}  coord {0.1 * v:.3f} "
                     f"obj {0.5 * v:.3f} noobj {0.2 * v:.3f} cls {0.2 * v:.3f}")
    return "\n".join(lines)


def _train(name, batch, subset, n_images, seconds, out="build/notebooks/x/m/runs"):
    sub = f" --subset {subset}" if subset else ""
    return (f"$ python tools/train.py --net m.cfg --dataset x --init coco --iters 20 "
            f"--batch {batch} --lr 0.001 --burn-in 10 --multiscale{sub} --workers 4 "
            f"--device gpu --out {out}/{name}\n"
            f"poids yolov3-tiny.weights copiés ; couches réinitialisées : [15, 22]\n"
            f"{_log(n_images)}\npoids : {out}/{name}/final.weights\n  ({seconds} s)\n")


def _eval_voc(name, aps, out="build/notebooks/x/m/runs"):
    rows = "\n".join(f"| {c} | {v} |" for c, v in aps.items())
    m = np.mean(list(aps.values()))
    return (f"$ python tools/eval_voc.py --net m.cfg --weights {out}/{name}/final.weights "
            f"--dataset x --subset 50\n50/50 images      13 s  (268 ms/image)\n"
            f"### m.cfg, x test, 50 images, AP 11 points\n\n| Classe | AP |\n|---|---|\n{rows}\n"
            f"| **mAP** | **{m:.2f}** |\n\n  (17 s)\n")


def _eval_coco(name, ap, ap50, out="build/notebooks/x/m/runs"):
    return (f"$ python tools/eval_voc.py --net m.cfg --weights {out}/{name}/final.weights "
            f"--dataset x --subset 50\n### m.cfg, x val, métrique COCO (101 points)\n\n"
            f"| AP | AP50 | AP75 | APs | APm | APl |\n|---|---|---|---|---|---|\n"
            f"| {ap} | {ap50} | 0.1 | 0.1 | 1.0 | 5.0 |\n\n"
            f"| Classe | AP | AP50 |\n|---|---|---|\n| person | {ap} | {ap50} |\n"
            f"| dog | -100.0 | -100.0 |\n\n  (15 s)\n")


def _notebook(path, outputs):
    cells = [{"cell_type": "code", "execution_count": 1, "id": "c00", "metadata": {},
              "source": ["ITERS = 20\n", "BURN_IN = 10\n", "NET = 'm.cfg'\n"], "outputs": []}]
    for k, text in enumerate(outputs, 1):
        cells.append({"cell_type": "code", "execution_count": k + 1, "id": f"c{k:02d}",
                      "metadata": {}, "source": ["pass"],
                      "outputs": [{"name": "stdout", "output_type": "stream", "text": [text]}]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4,
                                "nbformat_minor": 5}))
    return path


def _sweep_dir(tmp_path):
    """Deux jeux jouets : `voc` (métrique VOC, un run sauté) et `flir` (métrique COCO)."""
    train = (_train("b8-s500", 8, 500, 500, 100) + "b8-sall : déjà entraîné, sauté\n"
             + _train("b32-sall", 32, 0, 1000, 400))
    evals = (_eval_voc("b8-s500", {"cat": 10.0, "dog": 20.0})
             + _eval_voc("b32-sall", {"cat": 50.0, "dog": 30.0}))
    table = ("| run | lot | images | itérations | backend | perte finale | mAP | s/image | éval. |\n"
             "|---|---|---|---|---|---|---|---|---|\n"
             "| b32-sall | 32 | tout | 20 | gpu | 9.00 | 40.00 | 0.03 | 50 |\n"
             "| b8-sall | 8 | tout | 20 | gpu | 12.00 | 25.00 | 0.03 | 50 |\n"
             "| b8-s500 | 8 | 500 | 20 | gpu | 8.00 | 15.00 | 0.03 | 50 |\n")
    _notebook(tmp_path / "voc" / "m_sweep.ipynb", [train, evals + table])
    _notebook(tmp_path / "voc" / "tiny-yolov3-coco_infer.ipynb",
              [_eval_voc("x", {"cat": 70.0, "dog": 60.0})])
    train = (_train("b8-s500", 8, 500, 500, 100) + _train("b8-sall", 8, 0, 2000, 110)
             + _train("b32-sall", 32, 0, 2000, 380))
    evals = (_eval_coco("b8-s500", 0.3, 1.3) + _eval_coco("b8-sall", 0.7, 2.9)
             + _eval_coco("b32-sall", 0.4, 1.7))
    _notebook(tmp_path / "flir" / "m_sweep.ipynb", [train, evals])
    _notebook(tmp_path / "kitti" / "m_sweep.ipynb", [])  # pas encore exécuté
    return tmp_path


def test_parse_eval_voc_et_coco():
    ev = B.parse_eval(_eval_voc("r", {"cat": 10.0, "dog": 30.0}).partition("\n")[2])
    assert ev["metric"] == "voc" and ev["map"] == 20.0 and ev["ap50"] == 20.0
    assert ev["classes"] == {"cat": 10.0, "dog": 30.0} and ev["ms_image"] == 268
    ev = B.parse_eval(_eval_coco("r", 0.7, 2.9).partition("\n")[2])
    assert ev["metric"] == "coco" and ev["ap"] == 0.7 and ev["map"] == ev["ap50"] == 2.9
    assert ev["classes"] == {"person": 0.7, "dog": None}  # -100 : classe absente
    assert ev["classes50"]["person"] == 2.9 and ev["coco"]["APl"] == 5.0


def test_parse_sweep_runs_journal_table(tmp_path):
    root = _sweep_dir(tmp_path)
    d = B.parse_sweep(root / "voc" / "m_sweep.ipynb")
    assert [r["run"] for r in d["runs"]] == ["b8-s500", "b8-sall", "b32-sall"]
    assert d["n_train"] == 1000 and d["iters"] == 20 and d["burn_in"] == 10
    r = {r["run"]: r for r in d["runs"]}
    # journal : courbe, durée, époques ; perte finale de la table (loss.csv complet)
    assert len(r["b32-sall"]["curve"]["loss"]) == 20 and r["b32-sall"]["duration_s"] == 400
    assert r["b32-sall"]["final_loss"] == 9.0 and r["b32-sall"]["map"] == 40.0
    assert r["b32-sall"]["epochs"] == pytest.approx(20 * 32 / 1000)
    assert r["b8-s500"]["epochs"] == pytest.approx(20 * 8 / 500)
    parts = r["b32-sall"]["final_parts"]
    assert sum(parts.values()) == pytest.approx(np.mean(100 / np.arange(1, 21)), rel=1e-3)
    # run sauté : ni courbe ni durée, mAP et perte de la table
    assert r["b8-sall"]["curve"] is None and r["b8-sall"]["duration_s"] is None
    assert r["b8-sall"]["map"] == 25.0 and r["b8-sall"]["final_loss"] == 12.0
    assert r["b8-sall"]["images"] == 1000


def test_load_all_scores_rangs_et_attente(tmp_path):
    data, pending = B.load_all(_sweep_dir(tmp_path))
    assert list(data) == ["voc", "flir"] and list(pending) == ["kitti"]
    voc, flir = data["voc"], data["flir"]
    assert voc["best"] == "b32-sall" and voc["top"] == 40.0
    assert [r["rank"] for r in voc["runs"]] == [3.0, 2.0, 1.0]
    assert flir["metric"] == "coco" and flir["best"] == "b8-sall"  # score : AP50
    assert {r["run"]: r["rel"] for r in flir["runs"]}["b32-sall"] == pytest.approx(1.7 / 2.9)
    assert voc["baselines"]["tiny-yolov3-coco"]["map"] == 65.0
    assert voc["baselines"]["tiny-yolov3-coco"]["n_classes"] == 2
    assert flir["baselines"] == {}
    sets, rho = B.rank_agreement(data)
    assert rho[0, 0] == pytest.approx(1.0) and rho[0, 1] == pytest.approx(0.5)
    eff = B.effects(data)
    assert eff["sall_minus_s500"]["voc"][8] == pytest.approx((25 - 15) / 40)
    for text in (B.summary_table(data), B.runs_table(data), B.conclusions(data, pending),
                 *B.observations(data).values()):
        assert isinstance(text, str) and text
    assert "kitti" in B.conclusions(data, pending)
    json.loads(B.to_json(data))


def test_spearman_ex_aequo():
    assert B.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert B.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert list(B._ranks([5, 1, 5])) == [2.5, 1.0, 2.5]


def test_figures_sur_donnees_jouets(tmp_path):
    pytest.importorskip("matplotlib")
    from tools.figures import balayages as FB

    data, _ = B.load_all(_sweep_dir(tmp_path / "nb"))
    names = []
    for key, fn in FB.PLOTS.items():
        paths = fn(data, tmp_path / "out")
        assert all(p.suffix == ".png" and p.stat().st_size > 1000 for p in paths), key
        names += [p.name for p in paths]
    assert len(set(names)) == len(FB.PLOTS)
    assert all(n.startswith("balayages_") for n in names)


SWEEPS = ROOT / "notebooks"


@pytest.mark.parametrize("ds, run, score", [
    ("voc", "b32-sall", 36.27), ("visdrone", "b32-sall", 2.24), ("kitti", "b32-sall", 2.38),
    ("flir", "b8-sall", 2.9), ("exdark", "b32-sall", 13.29)])
def test_notebooks_versionnes(ds, run, score):
    """Valeurs relues == tables de docs/tasks/resultats-balayages.md (et sortie ExDark)."""
    p = SWEEPS / ds / f"tiny-yolov3-{ds}_sweep.ipynb"
    if not p.exists() or not B._executed(p):
        pytest.skip(f"{p.name} non exécuté")
    d = B._finish(B.parse_sweep(p))
    assert d["best"] == run and d["top"] == pytest.approx(score)
    assert len(d["runs"]) == 6 and all(r["map"] is not None for r in d["runs"])
    if ds == "flir":
        r = next(r for r in d["runs"] if r["run"] == run)
        assert r["ap"] == 0.7 and r["final_loss"] == pytest.approx(67.27)


# ------------------------------------------------- collecte en fin de commande (T14.8)

def _run_dir(root, name="b8-sall", iters=60):
    from tools.notebooks import runs

    d = root / "build" / "notebooks" / "x" / "m" / "runs" / name
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"batch": 8, "subset": 0, "iters": iters}))
    rows = ["it,lr,size,loss,coord,obj,noobj,cls,seconds"]
    rows += [f"{i},0.001,416,{100 / i},{10 / i},{50 / i},{20 / i},{20 / i},2.0"
             for i in range(1, iters + 1)]
    (d / "loss.csv").write_text("\n".join(rows) + "\n")
    (d / "map_float_s50.md").write_text(_eval_voc("r", {"cat": 10.0, "dog": 30.0})
                                        .partition("\n")[2])
    return runs, d


def test_summarize_journal_et_evaluation(tmp_path):
    runs, d = _run_dir(tmp_path)
    s = runs.summarize(d)
    assert s["meta"]["batch"] == 8
    assert s["train"]["iters"] == 60 and s["train"]["seconds"] == 120.0
    assert s["train"]["final_loss"] == pytest.approx(np.mean(100 / np.arange(11, 61)))
    assert set(s["train"]["final_parts"]) == {"coord", "obj", "noobj", "cls"}
    assert s["eval"]["map_float_s50"]["map"] == 20.0
    assert s["eval"]["map_float_s50"]["classes"] == {"cat": 10.0, "dog": 30.0}
    p = runs.write_summary(d)
    assert json.loads(p.read_text())["train"]["iters"] == 60
    assert runs.write_summary(tmp_path / "vide") is None
    agg = runs.write_summaries(d.parent.parent)
    assert list(json.loads(agg.read_text())) == [str(d)]


def test_autosync_colab_seulement(tmp_path, monkeypatch):
    from tools.notebooks import colab

    runs, d = _run_dir(tmp_path)
    monkeypatch.setattr(colab, "ROOT", tmp_path)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    rel = d.relative_to(tmp_path)
    cmd = ["python", "tools/train.py", "--net", "m.cfg", "--out", str(rel)]
    drive = tmp_path / "drive"
    assert colab.autosync(cmd, drive) == []  # hors Colab : rien
    monkeypatch.setattr(colab, "in_colab", lambda: True)
    assert colab.autosync(cmd, drive) == []  # Drive non monté : rien
    drive.mkdir()
    monkeypatch.setenv("EMBEDDEDCV_AUTOSYNC", "0")
    assert colab.autosync(cmd, drive) == []
    monkeypatch.delenv("EMBEDDEDCV_AUTOSYNC")
    assert colab.autosync(["python", "tools/detect.py", "--out", str(rel)], drive) == []
    assert colab.autosync(cmd, drive) == [rel]
    copied = drive / "runs" / rel
    assert (copied / "loss.csv").is_file() and (copied / "summary.json").is_file()
    # évaluation : dossier du --markdown, résumé avec la mAP
    ev = ["python", "tools/eval_voc.py", "--weights", "w", "--markdown",
          str(rel / "map_float_s50.md")]
    assert colab.autosync(ev, drive) == [rel]
    assert json.loads((copied / "summary.json").read_text())["eval"]["map_float_s50"]["map"] == 20.0


def test_complete_from_build(tmp_path):
    root = _sweep_dir(tmp_path / "nb")
    _, d = _run_dir(tmp_path)  # build/notebooks/x/m/runs/b8-sall
    sweep = B.parse_sweep(root / "voc" / "m_sweep.ipynb")
    sweep["dataset"] = "x"
    assert B.complete_from_build(sweep, tmp_path / "build" / "notebooks") == ["b8-sall"]
    r = next(r for r in sweep["runs"] if r["run"] == "b8-sall")
    assert len(r["curve"]["loss"]) == 60 and r["duration_s"] == 120.0
    assert r["final_loss"] == 12.0  # perte de la table gardée
    assert r["full_log"] and r["from_build"]
