"""Notebooks M14 : générateur, registre, commandes face à tools/m11.sh et m12.sh, fumée."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.notebooks import commandes as C  # noqa: E402
from tools.notebooks import runs  # noqa: E402
from tools.notebooks.env import Prerequis  # noqa: E402
from tools.notebooks.__main__ import generate, main  # noqa: E402
from tools.notebooks.gabarits import M12_ADMM, M12_QAT, dumps, render  # noqa: E402
from tools.notebooks.matrice import DATA_MARKERS, NOTEBOOKS, TRAINABLE, prerequis  # noqa: E402
from yolo.data.datasets import DATASETS, MAPPINGS  # noqa: E402


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _params(nb, **over):
    """Valeurs de la cellule `parameters` du notebook rendu, avec `over` en plus."""
    cell = next(c for c in nb["cells"] if "parameters" in c["metadata"].get("tags", ()))
    p = {}
    exec("".join(cell["source"]), {}, p)
    return {**p, **over}


# ------------------------------------------------------------------ T14.0 générateur

def test_generation_idempotente_sans_sorties(tmp_path):
    files = generate(NOTEBOOKS.values(), tmp_path)
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    again = generate(NOTEBOOKS.values(), tmp_path)
    assert all(again[p] == p.read_text() for p in files)
    nbformat = None
    try:
        import nbformat
    except ImportError:
        pass
    for p in files:
        if p.suffix != ".ipynb":
            continue
        text = p.read_text()
        assert text.endswith("}\n")
        nb = json.loads(text)
        assert dumps(nb) == text  # format d'enregistrement de Jupyter
        if nbformat:
            nbformat.validate(nbformat.reads(text, as_version=4))
        assert nb["nbformat"] == 4
        for c in nb["cells"]:
            assert isinstance(c["source"], list)
            if c["cell_type"] == "code":
                assert c["outputs"] == [] and c["execution_count"] is None
        for src in _code(nb):
            kept = [line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))]
            compile("\n".join(kept), str(p), "exec")


def test_structure_fixe():
    for nb in NOTEBOOKS.values():
        cells = render(nb)["cells"]
        assert cells[0]["source"][0].startswith(f"# {nb.model} sur {nb.dataset}")
        assert cells[2]["metadata"] == {"tags": ["parameters"]}
        assert "env.prepare(" in "".join(cells[4]["source"])
        assert cells[-2]["source"] == ["## Résumé"]
        p = _params(render(nb))
        assert p["OUT"].startswith("build/notebooks/") and p["SUBSET"] <= 100
        for k in ("NET", "DATASET", "SPLIT", "WEIGHTS", "DEVICE", "SUBSET", "SIZE", "RESIZE"):
            assert k in p
        if nb.role == "train":
            for k in ("ITERS", "BATCH", "LR", "INIT", "QAT", "ADMM", "DRIVE_DIR"):
                assert k in p
        if nb.role == "sweep":
            for k in ("BATCHES", "TRAIN_SUBSETS", "ITERS", "LR", "INIT", "SKIP_DONE"):
                assert k in p
            assert "BATCH" not in p and "QAT" not in p
        if nb.role == "infer" and nb.finetuned:
            assert p["WEIGHTS"] is None and p["RUN"] is None and p["COMPARE"] is False
            assert p["RUNS_DIR"] == nb.train_dir and p["OUT"] == f"{nb.train_dir}/eval"


def test_check_detecte_une_retouche(tmp_path, capsys):
    assert main(["all", "--out", str(tmp_path)]) == 0
    assert main(["--check", "--out", str(tmp_path)]) == 0
    p = tmp_path / "voc" / "tiny-yolov2-voc_infer.ipynb"
    nb = json.loads(p.read_text())
    nb["cells"][2]["source"][0] = "NET = 'autre'\n"
    p.write_text(json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    assert main(["--check", "--out", str(tmp_path)]) == 1
    assert "voc/tiny-yolov2-voc_infer.ipynb" in capsys.readouterr().out


def test_visionneuse_generee_et_affichage_seul(tmp_path):
    from tools.notebooks.gabarits import VIEWER, viewer

    files = generate([], tmp_path)
    assert tmp_path / VIEWER in files and VIEWER in files[tmp_path / "README.md"]
    nb = viewer()
    p = _params(nb)
    assert "docs/tasks/figures" in p["DIRS"] and p["PATTERN"] == "*.png"
    src = "\n".join(_code(nb))
    assert "clear_output" in src and "KeyboardInterrupt" in src
    assert "tools.notebooks.commandes" not in src and "C.run" not in src  # aucun calcul


def test_notebooks_versionnes_a_jour():
    assert main(["--check"]) == 0


# ------------------------------------------------------------------- T14.2 registre

def test_matrice():
    keys = set(NOTEBOOKS)
    assert {"voc/tiny-yolov2-voc_infer", "voc/tiny-yolov3-coco_infer",
            "voc/tiny-yolov3-voc_infer", "voc/tiny-yolov3-voc_train",
            "coco/tiny-yolov3-coco_infer", "kitti/tiny-yolov3-kitti_train",
            "visdrone/tiny-yolov3-visdrone_infer", "flir/tiny-yolov3-flir_train",
            "exdark/tiny-yolov2-voc_infer", "crowdhuman/tiny-yolov3-coco_infer"} <= keys
    assert "coco/tiny-yolov2-voc_infer" not in keys
    assert not any(k.startswith(("coco/", "exdark/", "crowdhuman/")) and k.endswith("_train")
                   for k in keys)
    for ds in DATASETS:
        assert any(nb.dataset == ds and nb.role == "infer" for nb in NOTEBOOKS.values()), ds
    for nb in NOTEBOOKS.values():
        if nb.out_of_domain:
            assert (nb.dataset, nb.family) in MAPPINGS
        if nb.role == "infer" and nb.net.endswith(".cfg"):
            assert nb.finetuned and "cfg" in prerequis(nb)
    assert NOTEBOOKS["flir/tiny-yolov3-flir_train"].net == "build/m11/cfg/tiny-yolov3-flir-c1.cfg"
    for ds in TRAINABLE:  # T14.10 : un balayage par jeu affiné, mêmes cfg et dossier
        train, sweep = (next(nb for nb in NOTEBOOKS.values() if nb.dataset == ds
                             and nb.role == role) for role in ("train", "sweep"))
        assert (sweep.net, sweep.train_dir) == (train.net, train.train_dir)


def test_temoins_de_get_datasets():
    script = (ROOT / "tools" / "get_datasets.sh").read_text()
    for ds, marker in DATA_MARKERS.items():
        if ds != "voc":
            assert marker.removeprefix("data/") in script, ds


# ---------------------------------------------- T14.6, T14.7 commandes face aux scripts

def _block(script, start, end):
    text = (ROOT / "tools" / script).read_text()
    i = text.index(start)
    return text[i:text.index(end, i)].replace("\\\n", " ")


def _expand(line, subs):
    for k in sorted(subs, key=len, reverse=True):
        line = line.replace(k, subs[k])
    assert "$" not in line, line
    return shlex.split(line)


def _steps(block, subs):
    return [_expand(line.strip().removeprefix("step "), subs) for line in block.splitlines()
            if line.strip().startswith("step python")]


def _evalq(script, block, subs):
    """Appels `evalq sortie options…` du bloc, développés avec la fonction du script."""
    tpl = _block(script, "evalq() {", "\n}\n")
    tpl = next(line.strip().removeprefix("step ") for line in tpl.splitlines()
               if "step python" in line)
    out = []
    for line in block.splitlines():
        if line.strip().startswith("evalq "):
            args = line.strip().removeprefix("evalq ").split(" ", 1)
            out.append(_expand(tpl.replace('"$out"', args[0]).replace('"$@"', args[1]), subs))
    return out


def _opts(cmd):
    """Outil et {option: valeur} (nombres comparés comme flottants)."""
    def val(v):
        try:
            return float(v)
        except ValueError:
            return v
    tool, opts, i = cmd[1], {}, 2
    while i < len(cmd):
        if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            opts[cmd[i]] = val(cmd[i + 1])
            i += 2
        else:
            opts[cmd[i]] = True
            i += 1
    return tool, opts


def _same(got, want):
    assert [_opts(c) for c in got] == [_opts(c) for c in want]


def _train(p, run, resume=False):
    """Appel de `cmd_train` de la cellule d'affinage du gabarit `train`."""
    return C.cmd_train(p["NET"], run, p["DATASET"], p["INIT"], p["INIT_NET"], resume, p["QAT"],
                       p["QAT_STEPS"], p["ADMM"], p["ADMM_RHO"], p["ADMM_GROWTH"],
                       p["ADMM_EVERY"], p["ITERS"], p["BATCH"], p["LR"], p["BURN_IN"],
                       p["MULTISCALE"], None, 0, p["SAVE_EVERY"], p["WORKERS"], p["DEVICE"],
                       p["DATA_ROOT"])


def test_gabarit_appelle_cmd_train_comme_le_test():
    src = "\n".join(_code(render(NOTEBOOKS["kitti/tiny-yolov3-kitti_train"])))
    call = re.sub(r"\s+", " ", src[src.index("C.cmd_train("):])
    assert call.startswith("C.cmd_train(NET, RUN, DATASET, INIT, INIT_NET, resume, QAT, "
                           "QAT_STEPS, ADMM, ADMM_RHO, ADMM_GROWTH, ADMM_EVERY, ITERS, BATCH, "
                           "LR, BURN_IN, MULTISCALE, None, 0, SAVE_EVERY, WORKERS, DEVICE, "
                           "DATA_ROOT)")


def test_kitti_identique_a_m11():
    nb = NOTEBOOKS["kitti/tiny-yolov3-kitti_train"]
    p = _params(render(nb))
    d, anchors = nb.out_dir, "ANCHORS.md"
    subs = {'"${INIT[@]}"': "--init coco", '"${resume[@]}"': "", '"${CFG_OPTS[@]}"': "",
            '"$(anchors_k $K "$anchors")"': "A", '"$anchors"': anchors, '"$ds"': "kitti",
            '"$cfg"': p["NET"], '"$d/final.weights"': f"{d}/final.weights", '"$d"': d,
            '"$NET-voc"': "tiny-yolov3-voc", '"${SIZE:-416}"': "416", '"$ITERS"': "4000",
            '"$BATCH"': "16", '"$DEVICE"': p["DEVICE"], '"$SUBSET"': "0",
            '"$M11/cfg/$NET-$ds$VAR.cfg"': "build/m11/cfg/tiny-yolov3-kitti.cfg"}
    want = (_steps(_block("m11.sh", "\nprep() {", "\n}\n"), subs)
            + _steps(_block("m11.sh", "\ntrain() {", "\n}\n"), subs))
    assert [c[1] for c in want] == ["tools/kmeans_anchors.py", "tools/make_cfg.py",
                                    "tools/train.py", "tools/eval_voc.py"]
    got = [C.cmd_anchors(p["DATASET"], p["SIZE"], anchors),
           C.cmd_make_cfg(p["BASE"], p["DATASET"], "A", p["NET"], p["SIZE"], p["CHANNELS"]),
           _train(p, d),
           C.cmd_eval_voc(p["NET"], f"{d}/final.weights", p["DATASET"], p["RESIZE"], 0)]
    _same(got, want)
    assert p["NET"] == C.cfg_path("tiny-yolov3", "kitti")


def test_sweep_ne_change_que_lot_et_sous_ensemble():
    """Une case (b, s) du balayage = la commande du notebook _train à --batch et --subset
    près ; dossier OUT/runs/b<b>-s<s>."""
    nb = NOTEBOOKS["kitti/tiny-yolov3-kitti_sweep"]
    src = re.sub(r"\s+", " ", "\n".join(_code(render(nb))))
    assert ("C.cmd_train(NET, run_dir, DATASET, INIT, INIT_NET, resume, iters=ITERS, batch=b, "
            "lr=LR, burn_in=BURN_IN, multiscale=MULTISCALE, subset=s, save_every=SAVE_EVERY, "
            "workers=WORKERS, device=DEVICE, data_root=DATA_ROOT)") in src
    p = _params(render(nb))
    t = _params(render(NOTEBOOKS["kitti/tiny-yolov3-kitti_train"]), ITERS=p["ITERS"], BATCH=8)
    d = f"{nb.train_dir}/runs/{runs.run_name(8, 500)}"
    got = C.cmd_train(p["NET"], d, p["DATASET"], p["INIT"], p["INIT_NET"], False,
                      iters=p["ITERS"], batch=8, lr=p["LR"], burn_in=p["BURN_IN"],
                      multiscale=p["MULTISCALE"], subset=500, save_every=p["SAVE_EVERY"],
                      workers=p["WORKERS"], device=p["DEVICE"], data_root=p["DATA_ROOT"])
    want = _train(t, d)
    assert _opts(got)[1] == {**_opts(want)[1], "--subset": 500.0}
    assert d.endswith("/runs/b8-s500") and runs.run_name(16, 0) == "b16-sall"
    assert p["BATCHES"] and 0 in p["TRAIN_SUBSETS"]


def _fake_run(root, rel, mtime, **meta):
    d = root / rel
    d.mkdir(parents=True)
    (d / "final.weights").write_bytes(b"w")
    (d / "loss.csv").write_text("it,lr,size,loss,coord,obj,noobj,cls,seconds\n"
                                "1,0.001,416,4.0,1,1,1,1,8.0\n2,0.001,416,2.0,1,1,1,1,8.0\n")
    if meta:
        runs.write_meta(d, **meta)
    os.utime(d / "final.weights", (mtime, mtime))


def test_runs_trouves_et_choisis(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    _fake_run(tmp_path, "m", 100)
    _fake_run(tmp_path, "m/runs/b8-s500", 300, batch=8, subset=500, iters=600)
    _fake_run(tmp_path, "m/runs/b16-sall", 200, batch=16, subset=0, iters=600)
    (tmp_path / "m" / "runs" / "b32-sall").mkdir()  # sans final.weights : ignoré
    found = runs.find_runs("m")
    assert [r.name for r in found] == ["train", "b16-sall", "b8-s500"]
    assert found[1].weights == "m/runs/b16-sall/final.weights" and found[1].meta["batch"] == 16
    assert runs.pick("m").name == "b8-s500"  # le plus récent
    assert runs.pick("m", "train").weights == "m/final.weights"
    with pytest.raises(Prerequis, match="b16-sall"):
        runs.pick("m", "b64-sall")
    with pytest.raises(Prerequis, match="aucun run"):
        runs.pick("vide")
    row = runs.row(found[2], 41.5, 50)
    assert row["perte finale"] == 3.0 and row["s/image"] == 1.0 and row["images"] == 500
    text = runs.table([runs.row(found[1], 30.0), row, runs.row(found[0], None)])
    assert text.splitlines()[2].startswith("| b8-s500 | 8 | 500 |")
    assert text.splitlines()[-1].startswith("| train |")


def test_read_map_des_tables_eval_voc(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_voc", ROOT / "tools" / "eval_voc.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    (tmp_path / "voc.md").write_text(ev.table([0.5, 0.25], 0.375, "t", ["a", "b"]))
    assert runs.read_map("voc.md") == 37.5
    stats = {k: 0.1 * (i + 1) for i, k in enumerate(ev.coco_eval.STATS)}
    res = {**stats, "ap_class": [0.2, 0.4], "ap50_class": [0.5, 0.6]}
    (tmp_path / "coco.md").write_text(ev.coco_table(res, "t", ["a", "b"]))
    assert runs.read_map("coco.md") == 10.0
    assert runs.read_map("absent.md") is None


@pytest.mark.parametrize("profile,over,lowbit", [("qat", M12_QAT, "w4a4"),
                                                 ("admm", M12_ADMM, "")])
def test_qat_admm_identiques_a_m12(profile, over, lowbit):
    nb = NOTEBOOKS["voc/tiny-yolov3-voc_train"]
    p = _params(render(nb), **over)
    d = nb.out_dir
    subs = {'"$NET"': p["NET"], '"$INIT"': p["INIT"], '"$DEVICE"': p["DEVICE"], '"$d"': d,
            '"$d/checkpoint.npz"': f"{d}/checkpoint.npz", '"$d/model"': f"{d}/model",
            '"$d/map.json"': f"{d}/map.json", '"$d/plan_mixed6.json"': f"{d}/plan_mixed6.json",
            '"$JOBS"': str(p["JOBS"]), '"$SUBSET"': str(p["SUBSET"]),
            '"${steps[@]}"': f"--qat-steps {p['QAT_STEPS']}", '"$lr"': "1e-4",
            '"$batch"': "8", '"$iters"': "600", '"$1"': "1e-3", '"$2"': "1.3", '"$3"': "50",
            '"$4"': "600"}
    block = _block("m12.sh", f"\n{profile})", "\n  ;;")
    want = _steps(block, subs) + _evalq("m12.sh", block, subs)
    assert [c[1] for c in want] == ["tools/train.py", "tools/quant_lowbit.py",
                                    "tools/eval_quant.py"]
    assert f"{d}/plan_mixed6.json" == M12_ADMM["ADMM"]
    got = [_train(p, d),
           C.cmd_quant_lowbit(p["NET"], f"{d}/checkpoint.npz", f"{d}/model", p["QAT"],
                              bool(p["ADMM"])),
           C.cmd_eval_quant(p["NET"], f"{d}/map.json", "int", p["SUBSET"], p["JOBS"],
                            p["DATASET"], model_dir=f"{d}/model")]
    assert ("--scheme" in got[1]) == bool(lowbit)
    _same(got, want)
    assert "W4A4_STEPS=build/m9/models/$NET-w4a4-ptq/steps.json" in (
        ROOT / "tools" / "m12.sh").read_text()
    if lowbit:
        assert p["QAT_STEPS"] == f"build/m9/models/{p['NET']}-w4a4-ptq/steps.json"


def test_run_relaie_et_echoue(capsys):
    assert C.run(["python", "-c", "print('bonjour')"]) == 0
    out = capsys.readouterr().out
    assert "$ python -c" in out and "bonjour" in out
    with pytest.raises(subprocess.CalledProcessError):
        C.run(["python", "-c", "raise SystemExit(3)"])


def test_anchors_k(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("| k | ancres | IoU |\n|---|---|---|\n| 6 | `10,14  23,27` | 0.6 |\n")
    assert C.anchors_k(p, 6) == "10,14  23,27"


# ---------------------------------------------------------------------- T14.9 fumée

SMOKE = {"SUBSET": 4, "ITERS": 2, "BATCH": 2, "SHOW": 2, "CALIB_IMAGES": 8, "JOBS": 2,
         "WORKERS": 2, "INT8": True, "BATCHES": [2], "TRAIN_SUBSETS": [4],
         "DEVICE": "cpu"}  # PC sans GPU


def _missing(nb):
    req = prerequis(nb)
    if nb.trains:
        req.pop("cfg", None)
    if nb.finetuned:  # poids : un run de _train ou _sweep (runs.find_runs)
        w = req.pop("poids")
        if not runs.find_runs(nb.train_dir):
            return [w]
    return [v for v in req.values() if not (ROOT / v).exists()]


@pytest.mark.slow
@pytest.mark.parametrize("key", list(NOTEBOOKS))
def test_fumee(key):
    """Exécution de bout en bout par `jupyter nbconvert --execute`, paramètres minimaux
    injectés dans la cellule `parameters` d'une copie (sans papermill)."""
    nb = NOTEBOOKS[key]
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter absent (pip install -e 'python[notebooks]')")
    if missing := _missing(nb):
        pytest.skip(f"{key} : {', '.join(missing)} absent(s)")
    out = f"build/notebooks/smoke/{nb.dataset}/{nb.name}"
    shutil.rmtree(ROOT / out, ignore_errors=True)
    data = render(nb)
    cell = next(c for c in data["cells"] if "parameters" in c["metadata"].get("tags", ()))
    cell["source"][-1] += "\n"
    cell["source"] += [f"{k} = {v!r}\n" for k, v in {**SMOKE, "OUT": out}.items()]
    copy = ROOT / out / nb.path.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(dumps(data))
    r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--ExecutePreprocessor.timeout=3600", "--output", copy.name,
                        str(copy)], cwd=copy.parent, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr[-4000:]
