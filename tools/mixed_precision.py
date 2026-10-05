"""Précision mixte par couche (T10.10) : sensibilité, front de Pareto, configuration retenue.

    python tools/mixed_precision.py sens     # une couche en bas bit, les autres INT8
    python tools/mixed_precision.py front    # glouton perte / bits gagnés, préfixes évalués
    python tools/mixed_precision.py full     # mAP complète (4 952 images) de la config retenue

Chaque configuration est un plan {id conv: int8 | uniform6 | mixed6 | uniform4} exporté par
`tools/quant_lowbit.py --weights pow2 --weights-plan` (activations INT8 calibrées, poids
uniform4 paquetés deux par octet) puis évalué par `tools/eval_quant.py --model-dir
--variants int` sur `--subset` images de VOC2007 test (classement, palier M de M12). Les
résultats sont gardés dans build/m10/mixed/ : une relance saute ce qui est déjà mesuré.

Coûts d'une configuration :
- octets de poids : stockage du moteur unique (4 bits paquetés, 6 et 8 bits sur un octet) et
  stockage au plus juste (6 bits = 0,75 octet, pour le streaming) ;
- DSP : plan `stream_model` à poids par couche (deux MAC par DSP à 4 bits), seule
  architecture où le DSP dépend de la couche ; le moteur unique garde ses Tm·Tn MAC.

La configuration retenue n'utilise que les niveaux uniformes (int8, uniform6, uniform4) :
mixed6 demande le multiplieur à décalages `ACC_WMODE_POW2`, global au noyau.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from perf_model import conv_layers  # noqa: E402
from stream_model import load_board, plan as stream_plan  # noqa: E402

NET = "tiny-yolov2-voc"
OUT = ROOT / "build" / "m10" / "mixed"
KIND_BITS = {"int8": 8, "uniform6": 6, "mixed6": 6, "uniform4": 4}
SENS_KINDS = ("uniform4", "uniform6", "mixed6")
UNIFORM = ("int8", "uniform6", "uniform4")  # ordre décroissant de bits
TOLERANCE = 1.0  # perte de mAP admise face à l'INT8 projeté (points, sous-ensemble)


def layers():
    return conv_layers(ROOT / "model" / NET / "manifest.json")


def convs():
    return [d["layer"] for d in layers()]


def all_int8():
    return {i: "int8" for i in convs()}


def plan_name(plan):
    low = [f"L{i}-{k}" for i, k in sorted(plan.items()) if k != "int8"]
    return "_".join(low) or "int8"


def costs(plan, board):
    ls = layers()
    n = {d["layer"]: d["k"] ** 2 * d["cin"] * d["cout"] for d in ls}
    engine = sum(n[i] // 2 if k == "uniform4" else n[i] for i, k in plan.items())
    exact = sum(n[i] * KIND_BITS[k] / 8 for i, k in plan.items())
    sp = stream_plan(ls, board, {i: KIND_BITS[k] for i, k in plan.items()})
    return {"engine_bytes": engine, "bits_bytes": int(exact), "stream_dsp": sp["dsp"],
            "stream_fps": round(sp["fps"], 1)}


def evaluate(plan, subset, jobs, keep=False):
    """mAP (sous-ensemble ou test complet si subset = 0) du plan ; résultat mis en cache."""
    name = plan_name(plan)
    tag = f"s{subset}" if subset else "full"
    res = OUT / f"map_{name}_{tag}.json"
    if not res.exists():
        model = OUT / "models" / f"{NET}-{name}"
        pf = OUT / "plans" / f"{name}.json"
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(json.dumps({str(i): k for i, k in plan.items()}, indent=1) + "\n")
        if not (model / "manifest.json").exists():
            subprocess.run([sys.executable, str(ROOT / "tools" / "quant_lowbit.py"),
                            "--net", NET, "--weights", "pow2", "--weights-plan", str(pf),
                            "--out", str(model)], check=True, stdout=subprocess.DEVNULL)
        cmd = [sys.executable, str(ROOT / "tools" / "eval_quant.py"), "--model-dir",
               str(model), "--variants", "int", "--resize", "stretch", "--jobs", str(jobs),
               "--blas-threads", "1", "--out", str(res)]
        if subset:
            cmd += ["--subset", str(subset)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        if not keep:
            subprocess.run(["rm", "-rf", str(model)], check=True)
    m = 100 * json.loads(res.read_text())["results"]["int"]["map"]
    print(f"{name:40s} {tag:6s} mAP {m:.2f}", flush=True)
    return m


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def cmd_sens(args):
    board = load_board(ROOT / "hw" / "boards" / "kv260.yaml")
    ref = evaluate(all_int8(), args.subset, args.jobs)
    rows = [dict(layer="-", kind="int8", map=ref, loss=0.0, **costs(all_int8(), board))]
    for i in convs():
        for kind in SENS_KINDS:
            p = {**all_int8(), i: kind}
            m = evaluate(p, args.subset, args.jobs)
            rows.append(dict(layer=i, kind=kind, map=m, loss=round(ref - m, 2),
                             **costs(p, board)))
    write_csv(OUT / "sensitivity.csv", rows)
    print(f"→ {OUT / 'sensitivity.csv'}")


def greedy_order(sens, n_weights):
    """Passages (couche, niveau) triés par perte mesurée / bits gagnés, niveaux uniformes."""
    moves = []
    for r in sens:
        if r["layer"] == "-" or r["kind"] not in UNIFORM:
            continue
        i, kind = int(r["layer"]), r["kind"]
        saved = n_weights[i] * (8 - KIND_BITS[kind])
        moves.append((max(float(r["loss"]), 0.0) / saved, i, kind))
    return sorted(moves)


def cmd_front(args):
    board = load_board(ROOT / "hw" / "boards" / "kv260.yaml")
    sens = read_csv(OUT / "sensitivity.csv")
    n = {d["layer"]: d["k"] ** 2 * d["cin"] * d["cout"] for d in layers()}
    ref = float(sens[0]["map"])
    plan = all_int8()
    rows = [dict(step=0, change="INT8 (référence)", plan=plan_name(plan), map=ref, loss=0.0,
                 **costs(plan, board))]
    predicted = 0.0
    for _, i, kind in greedy_order(sens, n):
        if UNIFORM.index(kind) <= UNIFORM.index(plan[i]):
            continue  # couche déjà à ce niveau ou plus bas
        loss_alone = next(float(r["loss"]) for r in sens
                          if r["layer"] == str(i) and r["kind"] == kind)
        predicted += max(loss_alone, 0.0)
        if predicted > args.max_loss:
            break
        plan = {**plan, i: kind}
        m = evaluate(plan, args.subset, args.jobs)
        rows.append(dict(step=len(rows), change=f"L{i:02d} → {kind}", plan=plan_name(plan),
                         map=m, loss=round(ref - m, 2), **costs(plan, board)))
    tout4 = {i: "uniform4" for i in convs()}
    m = evaluate(tout4, args.subset, args.jobs)
    rows.append(dict(step="-", change="tout uniform4", plan=plan_name(tout4), map=m,
                     loss=round(ref - m, 2), **costs(tout4, board)))
    write_csv(OUT / "front.csv", rows)
    print(f"→ {OUT / 'front.csv'}")


def retained(front):
    """Plus petite empreinte (moteur unique) dont la perte reste sous la tolérance."""
    ok = [r for r in front if r["step"] != "-" and float(r["loss"]) <= TOLERANCE]
    best = min(ok, key=lambda r: int(r["engine_bytes"]))
    plan = all_int8()
    if best["plan"] != "int8":
        for part in best["plan"].split("_"):
            layer, kind = part.split("-")
            plan[int(layer[1:])] = kind
    return best, plan


def cmd_full(args):
    best, plan = retained(read_csv(OUT / "front.csv"))
    print(f"retenue : {best['plan']}")
    evaluate(all_int8(), 0, args.jobs)
    evaluate(plan, 0, args.jobs, keep=True)
    (OUT / "retained.json").write_text(json.dumps({str(i): k for i, k in plan.items()},
                                                  indent=1) + "\n")
    # Export de référence du ctest tb_net_mixed (hls/CMakeLists.txt) et de golden-check.
    dst = ROOT / "build" / "m10" / "models" / f"{NET}-mixed"
    subprocess.run(["rm", "-rf", str(dst)], check=True)
    subprocess.run(["cp", "-r", str(OUT / "models" / f"{NET}-{plan_name(plan)}"), str(dst)],
                   check=True)
    print(f"→ {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("sens", "front", "full"))
    ap.add_argument("--subset", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--max-loss", type=float, default=8.0,
                    help="front : arrêt quand la somme des pertes isolées dépasse ce seuil")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"sens": cmd_sens, "front": cmd_front, "full": cmd_full}[args.cmd](args)


if __name__ == "__main__":
    main()
