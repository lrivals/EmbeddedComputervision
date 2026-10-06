"""M13 : figures (tools/figures/) — registre, tracés sur données jouets, valeurs == source,
mode automatique qui n'échoue jamais."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.figures as F  # noqa: E402
from tools.figures import auto, modeles, projet, resultats  # noqa: E402
from tools.figures.__main__ import gallery  # noqa: E402

RNG = np.random.default_rng(0)


def _pngs(paths):
    pngs = [Path(p) for p in paths if str(p).endswith(".png")]
    assert pngs and all(p.stat().st_size > 1000 for p in pngs)
    assert all(Path(p).with_suffix(".svg").exists() for p in pngs)
    return pngs


# --- infrastructure -----------------------------------------------------------------------

def test_registry_is_consistent():
    assert len(F.FIGURES) >= 20
    for f in F.FIGURES.values():
        assert f.family in F.FAMILIES and f.task.startswith("T13.") and f.caption
    assert F.select(["resultats"]) == [f for f in F.FIGURES.values() if f.family == "resultats"]
    with pytest.raises(SystemExit):
        F.select(["inconnue"])


def test_all_skips_cleanly_without_data(tmp_path, monkeypatch):
    """`make figures` sans données : chaque figure est sautée, aucune exception."""
    for mod in (resultats, modeles, projet):
        monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(resultats, "DEVKIT", tmp_path / "VOCdevkit")
    monkeypatch.setattr(modeles, "DEVKIT", tmp_path / "VOCdevkit")
    monkeypatch.setattr(projet, "TASKS", tmp_path / "docs" / "tasks")
    status = F.run(F.select(["all"]), tmp_path / "res", tmp_path / "build", verbose=False)
    assert set(status) == set(F.FIGURES)
    assert all(s == "sautée" for s, _ in status.values()), status
    assert not (tmp_path / "res").exists()


def test_gallery_links_existing_images_only(tmp_path):
    res = tmp_path / "results" / "figures"
    fig = F.FIGURES["cascade_m10"]
    (res / fig.family).mkdir(parents=True)
    (res / fig.family / "cascade_m10.png").write_bytes(b"png")
    path = gallery({"etat_art": ("sautée", "results/benchmarks.csv absent")}, res,
                   tmp_path / "build")
    text = path.read_text()
    assert path == tmp_path / "results" / "figures.md"
    assert "](figures/resultats/cascade_m10.png)" in text
    assert "Non générée : results/benchmarks.csv absent" in text
    for f in F.FIGURES.values():
        assert f"`{f.name}`" in text


def test_style_save_png_svg(tmp_path):
    from tools.figures import style as st

    fig, ax = st.plt().subplots()
    ax.plot([0, 1], [0, 1])
    _pngs(st.save(fig, tmp_path, "x"))
    assert np.allclose(st.smooth([1, 1, 1, 1], 3), 1)


# --- résultats : tracés sur données jouets -------------------------------------------------

def _aps(n=20):
    return list(RNG.uniform(0.2, 0.8, n))


def test_map_stades_plot_and_pending_stage(tmp_path):
    from yolo.data.voc import VOC_CLASSES

    d = {"net": "n", "images": 4952, "classes": list(VOC_CLASSES), "stages": [
        {"key": "flottant", "name": "Flottant", "map": 0.56, "aps": _aps()},
        {"key": "entier", "name": "Entier", "map": 0.55, "aps": _aps()},
        {"key": "csim", "name": "C-sim", "map": 0.55, "aps": _aps()},
        {"key": "carte", "name": "KV260", "map": None, "aps": None}]}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(d))
    _pngs(resultats.plot_map_stades(resultats.load_map_stades(p), tmp_path))
    d["images"] = 500
    p.write_text(json.dumps(d))
    with pytest.raises(F.MissingSource):
        resultats.load_map_stades(p)


def test_map_formats_reads_full_runs_only(tmp_path):
    def write(rel, results, images=4952):
        p = tmp_path / rel.format(net="n")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"images": images, "results": results}))

    write("build/m8/{net}/eval_float_int.json", {"float": {"map": 0.56}, "int": {"map": 0.55}})
    write("build/m9/map_{net}-w4a4-qat.json", {"int": {"map": 0.36}})
    write("build/m9/map_{net}-pow2-ptq.json", {"int": {"map": 0.5}}, images=500)
    rows = resultats.load_map_formats("n", tmp_path)
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("flottant", 0.56, 32), ("INT8 PTQ", 0.55, 8), ("w4a4 QAT", 0.36, 4)]
    _pngs(resultats.plot_map_formats(rows, tmp_path))


def test_pareto():
    pts = [(32, 56.3), (8, 55.7), (6, 54.4), (6, 52.5), (4, 36.4), (4, 17.1)]
    assert resultats.pareto(pts) == [0, 1, 2, 4]


def test_pr_curves_area_is_class_ap(tmp_path):
    from yolo.infer.metrics import voc_ap

    samples = [{"id": f"{i}", "xyxy": np.array([[10.0, 10, 50, 50]]),
                "labels": np.array([0]), "difficult": np.array([False])} for i in range(6)]
    dets = {0: ([f"{i}" for i in range(6)] + ["0"], np.linspace(1, 0.3, 7),
                np.array([[10.0, 10, 50, 50]] * 6 + [[100.0, 100, 120, 120]]))}
    curves = resultats.pr_curves(dets, samples, ["a", "b"])
    rec, prec, ap = curves["a"]
    assert ap == pytest.approx(voc_ap(rec, prec, True)) and ap == pytest.approx(1.0)
    r, p = resultats.voc07_points(rec, prec)
    assert len(r) == 11 and np.mean(p) == pytest.approx(ap)
    _pngs(resultats.plot_pr({"v": curves}, tmp_path, "pr", "t"))


def test_sensitivity_sources(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("layer,kind,map,loss\n-,int8,57.7,0\n0,uniform4,53.5,4.2\n2,uniform4,55,2.7\n"
                 "0,mixed6,54.4,3.3\n")
    ref, per = resultats.load_sensitivity(p)
    assert ref == 57.7 and per == {"uniform4": {0: 4.2, 2: 2.7}, "mixed6": {0: 3.3}}
    j = tmp_path / "fq.json"
    j.write_text(json.dumps({"images": 1000, "results": {
        "float": {"map": 0.56}, "fq:all": {"map": 0.55}, "fq:0": {"map": 0.555}}}))
    f, fall, n, fq = resultats.load_fq_sensitivity(j)
    assert (f, fall, n) == (56.00000000000001, 55.00000000000001, 1000)
    assert fq[0] == pytest.approx(0.5)
    _pngs(resultats.plot_sensitivity([("a", {"INT8": fq}), ("b", per)], tmp_path))


def test_calibration_plot(tmp_path):
    layers = [{"id": i, "act": "leaky", "candidates": {"p99": 3.0, "max": 6.0}, "pick": "p99",
               "scale": 3.0 / 127, "clip_rate": 0.01} for i in (0, 2)]
    samples = {0: RNG.normal(0, 1, 5000).astype(np.float32)}
    _pngs(resultats.plot_calibration("n", layers, samples, tmp_path, "c"))
    _pngs(resultats.plot_weight_scales("n", {0: RNG.uniform(1e-3, 1e-2, 16)}, tmp_path, "w"))


def test_layer_scales_and_snr():
    m = {"layers": [{"type": "conv", "out_scale": 0.5}, {"type": "maxpool"},
                    {"type": "route", "from": [0]}, {"type": "conv", "out_scale": 0.1}]}
    assert resultats.layer_scales(m) == {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.1}
    x = np.ones(10)
    assert resultats.snr_db(x, x) == np.inf
    assert resultats.snr_db(x, x * 0.9) == pytest.approx(20.0)


def test_layer_errors_plot(tmp_path):
    errors = [("000001", [(0, 30.0, 1.0), (1, 25.0, 2.0)])]
    _pngs(resultats.plot_layer_errors("n", errors, [("000001", [(0, 0), (1, 0)])], tmp_path,
                                      "e"))
    _pngs(resultats.plot_layer_errors("n", errors, None, tmp_path, "e2"))


def test_cycles_load_equals_csv(tmp_path):
    p = tmp_path / "c.csv"
    cols = ["net", "layer", "load_in", "load_w", "compute", "store", "sequential",
            "overlapped"]
    with p.open("w") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(["a", 0, 10, 20, 30, 5, 65, 40])
        w.writerow(["a", 2, 1, 2, 3, 4, 10, 6])
    d = resultats.load_cycles(p)
    assert d["a"][1] == dict(net="a", layer=2, load_in=1, load_w=2, compute=3, store=4,
                             sequential=10, overlapped=6)
    _pngs(resultats.plot_cycles(d, tmp_path))


def test_cascade_steps_follow_scenarios():
    pm = pytest.importorskip("tools.perf_model")
    if not (ROOT / "model" / "tiny-yolov2-voc" / "manifest.json").exists():
        pytest.skip("export absent : model/tiny-yolov2-voc (make export)")
    steps, bound, _ = resultats.load_cascade()
    _, rows = pm.scenario_table("tiny-yolov2-voc")
    assert [s[0] for s in steps] == [n for n, _ in pm.SCENARIOS]
    assert [s[1] for s in steps] == [r[2] for r in rows[:len(steps)]]
    assert bound == next(r[2] for r in rows if r[0].startswith("borne calcul M6"))
    assert steps[0][3]  # le noyau M6 est fait


def test_scenario_done():
    k = dict(width=8, trim=True, requant=8, fold=True, tile_pool=14)
    base = dict(width=1, trim=False, requant=1, fold=False, tile_pool=None)
    assert resultats.scenario_done(base, k)
    assert resultats.scenario_done(dict(base, width=8, fold=True), k)
    assert not resultats.scenario_done(dict(base, width=16), k)
    assert not resultats.scenario_done(dict(base, tile_pool=12), k)


def test_cascade_and_roofline_plots(tmp_path):
    steps = [("M6", 200.0, 5.0, True), ("x", 40.0, 25.0, False)]
    _pngs(resultats.plot_cascade(steps, 41.1, 31.9, tmp_path))
    board = {"name": "B", "freq_hz": 200e6, "bw_bytes": 13.4e9}
    _pngs(resultats.plot_roofline_layers(board, [(0, 8.0, 3.0, 30.0), (2, 90.0, 30.0, 140.0)],
                                         "n", tmp_path, "r"))


def test_benchmarks_load_equals_csv(tmp_path):
    src = ROOT / "results" / "benchmarks.csv"
    if not src.exists():
        pytest.skip("results/benchmarks.csv absent")
    rows = resultats.load_benchmarks(src)
    raw = list(csv.DictReader(src.open()))
    assert len(rows) == len(raw)
    for r, s in zip(rows, raw):
        assert r["travail"] == s["travail"]
        assert r["gops"] == (float(s["gops"]) if s["gops"] else None)
        assert r["projection"] == ("projection" in s["travail"])
    _pngs(resultats.plot_benchmarks(rows, tmp_path))


def test_resources_hwpp_plots(tmp_path):
    board = {"name": "B"}
    _pngs(resultats.plot_resources(board, [("a", 896, 258048), ("b", 968, 2.0e6)],
                                   (1248, 3.0e6), tmp_path))
    _pngs(resultats.plot_hwpp(0.5566, 0.5556, 0.5556, RNG.integers(0, 200, 500), tmp_path))


def test_training_load_and_plots(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    lines = ["it,lr,size,loss,coord,obj,noobj,cls,seconds"]
    lines += [f"{i},{1e-4 * min(1, i / 10)},416,{9 - i * 0.01},1,4,2,0.5,1" for i in range(120)]
    (run / "loss.csv").write_text("\n".join(lines) + "\n")
    (run / "admm.csv").write_text("it,rho,res_0,res_2\n50,0.001,0.05,0.06\n100,0.0013,0.07,0.08\n")
    r = resultats.load_training(run)
    assert r["loss"]["loss"][0] == 9 and r["admm"]["res_2"].tolist() == [0.06, 0.08]
    _pngs(resultats.plot_training([r], tmp_path, "t1"))
    _pngs(resultats.plot_training([r, dict(r, name="b", admm=None)], tmp_path, "t2"))


def test_eval_json_and_summary_plots(tmp_path):
    d = {"net": "n", "images": 10, "classes": ["a", "b", "c"],
         "results": {"float": {"map": 0.5, "aps": [0.4, 0.5, 0.6]},
                     "int": {"map": 0.4, "aps": [0.3, 0.4, 0.5]}}}
    _pngs(resultats.plot_eval_json(d, tmp_path, "e"))
    _pngs(resultats.plot_map_summary([("a", 0.5, 4952), ("b", 0.3, 500)], tmp_path))


# --- modèles : valeurs == source ----------------------------------------------------------

def _manifest(net):
    if not (ROOT / "model" / net / "manifest.json").exists():
        pytest.skip(f"export absent : model/{net} (make export)")


@pytest.mark.parametrize("net, gmac", [("tiny-yolov2-voc", 3.486), ("tiny-yolov3-coco", 2.782)])
def test_profile_totals_equal_spec(net, gmac, tmp_path):
    """§3 : 3,486 GMAC (v2) et 2,782 GMAC (v3), comme tools/count_macs.py."""
    _manifest(net)
    from tools.count_macs import net_from_manifest, table

    rows = modeles.load_profile(net)
    assert round(sum(r[2] for r in rows) / 1e9, 3) == gmac
    ref = table(net_from_manifest(ROOT / "model" / net / "manifest.json"))
    assert [(r[2], r[3]) for r in rows] == [(t[6], t[5]) for t in ref]
    _pngs(modeles.plot_profile({net: rows}, tmp_path))


def test_memory_arena_matches_driver_rule(tmp_path):
    _manifest("tiny-yolov2-voc")
    m = json.loads((ROOT / "model" / "tiny-yolov2-voc" / "manifest.json").read_text())
    want = sum(-(-n // 64) * 64 for n in m["buffers"].values()) + 64
    assert modeles.arena_size(m) == want
    rows, arena, bram, nbytes = modeles.load_memory("tiny-yolov2-voc")
    assert arena == want and nbytes == bram * 2304
    # poids int8 des convs == blob exporté, au rembourrage d'alignement près (< 64 octets)
    assert 0 <= m["blobs"]["weights.bin"] - sum(r[1] for r in rows) < 64
    _pngs(modeles.plot_memory("tiny-yolov2-voc", rows, arena, bram, nbytes, tmp_path, "m"))


def test_graph_dot_one_node_per_layer():
    _manifest("tiny-yolov3-coco")
    src = modeles.graph_dot("tiny-yolov3-coco")
    m = json.loads((ROOT / "model" / "tiny-yolov3-coco" / "manifest.json").read_text())
    assert all(f"  L{i} [label=" in src for i in range(len(m["layers"])))
    assert "L19 -> L20;" in src and "L8 -> L20;" in src  # upsample + route vers la 2e tête


def test_anchors_md_parsed_as_published(tmp_path):
    src = ROOT / "results" / "anchors.md"
    if not src.exists():
        pytest.skip("results/anchors.md absent")
    a = modeles.load_anchors_md(src)
    assert a[5][1] == 0.6113 and a[6][1] == 0.6307
    assert a[5][0].shape == (5, 2) and a[6][3].shape == (6, 2)
    wh = RNG.uniform(10, 400, (500, 2))
    _pngs(modeles.plot_anchors(wh, a, tmp_path))


def test_voc_stats_plot(tmp_path):
    stats = {k: {"per_class": RNG.integers(100, 1000, 20), "areas": RNG.uniform(50, 1e5, 500),
                 "per_image": RNG.integers(1, 8, 100), "images": 100}
             for k in ("trainval 07+12", "test 2007")}
    _pngs(modeles.plot_voc_stats(stats, tmp_path))


# --- projet -------------------------------------------------------------------------------

def test_expand_task_ranges():
    assert projet._expand("T6.0-T6.3 en C-sim") == {"T6.0", "T6.1", "T6.2", "T6.3"}
    assert projet._expand("T9.1.3 vérifié en sim") == {"T9.1.3"}


def test_progress_and_tracking(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "M6-x.md").write_text("### [x] T6.0 — a\n### [ ] T6.1 — b\n### [ ] T6.2 — c\n"
                                   "### [~] T6.3 — d\n")
    (tasks / "README.md").write_text("## Suivi\n\n| Jalon | Tâches | Faites |\n|---|---|---|\n"
                                     "| M6 | 4 | 2 (T6.1 en C-sim ; carte en attente) |\n")
    t = projet.load_tasks(tasks)
    tr = projet.load_tracking(tasks / "README.md")
    assert t == {"M6": [("T6.0", "faite"), ("T6.1", "ouverte"), ("T6.2", "ouverte"),
                        ("T6.3", "partielle")]}
    prog = projet.progress(t, tr)["M6"]
    assert prog == {"faite": 1, "partielle": 1, "vérifiée sur PC, carte en attente": 1,
                    "ouverte": 1}
    assert projet.tracking_mismatches(t, tr) == [("M6", (4, 1), (4, 2))]
    _pngs(projet.plot_progress({"M6": prog}, projet.tracking_mismatches(t, tr), tmp_path))


def test_repo_tasks_vs_readme_tracking():
    """Signale (sans échouer) les écarts entre les titres [x] et le suivi du README."""
    tasks = projet.load_tasks()
    if not tasks:
        pytest.skip("docs/tasks absent")
    bad = projet.tracking_mismatches(tasks, projet.load_tracking())
    if bad:
        print("écarts suivi README / titres :", bad)


def test_milestone_spans_and_chronology(tmp_path):
    from datetime import datetime, timedelta

    t0 = datetime(2026, 10, 4, 19)
    commits = [(t0, "Initial commit"), (t0 + timedelta(hours=1), "M0 : a"),
               (t0 + timedelta(hours=2), "docs : b"), (t0 + timedelta(hours=4), "M1 : c")]
    spans = projet.milestone_spans(commits)
    assert spans == {"M0": (t0, t0 + timedelta(hours=1)),
                     "M1": (t0 + timedelta(hours=2), t0 + timedelta(hours=4))}
    _pngs(projet.plot_chronology(commits, spans, ["M2"], tmp_path))


def test_code_history_last_point_equals_tree():
    """Dernier point de `git log --numstat` == lignes des fichiers texte suivis."""
    try:
        _, _, series = projet.load_code_history()
    except F.MissingSource:
        pytest.skip("dépôt git absent")
    import subprocess

    files = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", "HEAD",
                            "python/"], capture_output=True, text=True).stdout.split()
    n = 0
    for f in files:
        data = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{f}"],
                              capture_output=True).stdout
        if b"\0" not in data:
            n += data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    assert series["python"][-1] == n


def test_code_and_tests_plots(tmp_path):
    series = {"python": np.array([10, 20]), "docs": np.array([5, 6]), "autres": np.array([0, 0])}
    _pngs(projet.plot_code(["M0 : a", "docs : b"], series, tmp_path))
    _pngs(projet.plot_tests({"test_a": 3, "test_b": 10}, {"build/golden": 4}, tmp_path))


# --- mode automatique ---------------------------------------------------------------------

def _run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    lines = ["it,lr,size,loss,coord,obj,noobj,cls,seconds"]
    lines += [f"{i},1e-4,416,{5 + np.sin(i)},1,2,1,1,1" for i in range(30)]
    (run / "loss.csv").write_text("\n".join(lines) + "\n")
    return run


def test_after_run_train_writes_next_to_run(tmp_path, monkeypatch):
    monkeypatch.delenv("YOLO_FIGURES", raising=False)
    run = _run_dir(tmp_path)
    paths = auto.after_run("train", run)
    assert run / "figures" / "entrainement.png" in paths


def test_after_run_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_FIGURES", "0")
    run = _run_dir(tmp_path)
    assert auto.after_run("train", run) == []
    assert not (run / "figures").exists()


def test_after_run_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("YOLO_FIGURES", raising=False)
    assert auto.after_run("train", tmp_path / "absent") == []
    assert auto.after_run("type-inconnu", tmp_path) == []

    def boom(*a, **k):
        raise RuntimeError("panne")

    monkeypatch.setitem(auto.HANDLERS, "train", boom)
    assert auto.after_run("train", tmp_path) == []
    assert "panne" in capsys.readouterr().out


def test_after_run_without_matplotlib(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("YOLO_FIGURES", raising=False)
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # import → ImportError
    assert auto.after_run("train", _run_dir(tmp_path)) == []
    assert "matplotlib absent" in capsys.readouterr().out


def test_after_run_scan(tmp_path, monkeypatch):
    monkeypatch.delenv("YOLO_FIGURES", raising=False)
    _run_dir(tmp_path)
    for name, m in (("a.json", 0.5), ("b.json", 0.4)):
        (tmp_path / name).write_text(json.dumps({"net": "n", "images": 20, "classes": ["x"],
                                                 "results": {"int": {"map": m, "aps": [m]}}}))
    names = {Path(p).name for p in auto.after_run("m12", tmp_path)}
    assert {"entrainement.png", "map_a.png", "map_b.png", "map_resume.png"} <= names


def test_yolo_package_never_imports_matplotlib():
    """ADR 0001 : le paquet yolo reste en NumPy pur."""
    for p in (ROOT / "python" / "yolo").rglob("*.py"):
        assert "matplotlib" not in p.read_text(), p
