"""Tables de synthèse des profils de M12 (docs/tasks/M12-profils-pc.md), en Markdown.

    # mAP et détections par image d'une ou plusieurs évaluations (eval_quant.py --out)
    python tools/m12_report.py map build/m12/map/*.json
    # perte d'un entraînement : moyenne glissante sur 50 itérations, début contre fin (T12.5)
    python tools/m12_report.py loss build/m12/qat/lr1e-4/loss.csv --window 50
    # résidus de l'ADMM : dernière ligne et pente entre deux pas Z / U (T12.6)
    python tools/m12_report.py admm build/m12/admm/A/admm.csv
    # détections identiques entre deux stades, sur les images de GOT (T12.9)
    python tools/m12_report.py dets build/m12/sim/int.jsonl build/m12/sim/c100/dets_*.jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def map_rows(paths):
    """Une ligne par (fichier, variante) : mAP en %, détections par image, réglages."""
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        for v, r in d["results"].items():
            rows.append({
                "file": Path(p).stem, "variant": v, "images": d["images"],
                "resize": d.get("resize"), "conf": d.get("conf"), "iou": d.get("iou"),
                "map": r["map"] * 100, "dets": r.get("dets_per_image"),
                "overflow": r.get("overflow"),
            })
    return rows


def map_table(rows):
    def fmt(v, spec):
        return "—" if v is None else format(v, spec)

    lines = ["| Fichier | Variante | Images | resize | conf | iou | mAP | dét./image | perdues |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['variant']} | {r['images']} | {r['resize'] or '—'} | "
                     f"{fmt(r['conf'], 'g')} | {fmt(r['iou'], 'g')} | {r['map']:.2f} | "
                     f"{fmt(r['dets'], '.1f')} | {fmt(r['overflow'], 'd')} |")
    return "\n".join(lines)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def moving_average(values, window):
    v = np.asarray(values, dtype=np.float64)
    if len(v) < window:
        return np.array([v.mean()]) if len(v) else v
    return np.convolve(v, np.ones(window) / window, mode="valid")


def loss_summary(path, window=50):
    """Moyenne glissante de la perte : première et dernière fenêtre, écart relatif."""
    rows = read_csv(path)
    loss = [float(r["loss"]) for r in rows]
    ma = moving_average(loss, window)
    sec = [float(r["seconds"]) for r in rows if r.get("seconds")]
    return {"iters": len(rows), "window": min(window, len(rows)),
            "start": float(ma[0]), "end": float(ma[-1]),
            "change": float(ma[-1] / ma[0] - 1) if len(ma) and ma[0] else 0.0,
            "s_per_iter": float(np.mean(sec)) if sec else None}


def admm_summary(path):
    """Dernière ligne de admm.csv, résidu max et pente moyenne des deux derniers pas."""
    rows = read_csv(path)
    keys = [k for k in rows[0] if k.startswith("res_")]
    res = np.array([[float(r[k]) for k in keys] for r in rows])
    last = res[-1]
    slope = (res[-1] - res[-2]) if len(res) > 1 else np.zeros_like(last)
    return {"updates": len(rows), "it": int(rows[-1]["it"]), "rho": float(rows[-1]["rho"]),
            "max_res": float(last.max()), "worst": keys[int(last.argmax())][4:],
            "mean_slope": float(slope.mean()), "converged": bool(last.max() < 0.01),
            "per_layer": dict(zip((k[4:] for k in keys), last.tolist()))}


def dets_compare(ref_path, got_paths):
    """(images comparées, différentes) : égalité exacte, sur les images présentes dans GOT."""
    from map_stades import compare, read_jsonl

    ref = read_jsonl([Path(ref_path)])
    got = read_jsonl([Path(p) for p in got_paths])
    ids = [i for i in got if i in ref]
    missing_ref = [i for i in got if i not in ref]
    _, diff = compare(ref, got, ids)
    return len(ids), diff, missing_ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("map")
    p.add_argument("files", nargs="+", type=Path)
    p = sub.add_parser("loss")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--window", type=int, default=50)
    p = sub.add_parser("admm")
    p.add_argument("files", nargs="+", type=Path)
    p = sub.add_parser("dets")
    p.add_argument("ref", type=Path)
    p.add_argument("got", nargs="+", type=Path)
    args = ap.parse_args()

    if args.cmd == "map":
        print(map_table(map_rows(args.files)))
    elif args.cmd == "loss":
        print(f"| Essai | Itérations | Perte début (moy. {args.window}) | Perte fin | Écart | "
              "s / itération |\n|---|---|---|---|---|---|")
        for f in args.files:
            s = loss_summary(f, args.window)
            spi = "—" if s["s_per_iter"] is None else f"{s['s_per_iter']:.2f}"
            print(f"| {f.parent.name} | {s['iters']} | {s['start']:.3f} | {s['end']:.3f} | "
                  f"{s['change'] * 100:+.1f} % | {spi} |")
    elif args.cmd == "admm":
        print("| Essai | Pas Z/U | Itération | ρ | Résidu max (couche) | Pente moyenne | "
              "< 0,01 |\n|---|---|---|---|---|---|---|")
        for f in args.files:
            s = admm_summary(f)
            print(f"| {f.parent.name} | {s['updates']} | {s['it']} | {s['rho']:.3g} | "
                  f"{s['max_res']:.4f} (L{s['worst']}) | {s['mean_slope']:+.4f} | "
                  f"{'oui' if s['converged'] else 'non'} |")
    else:
        n, diff, missing = dets_compare(args.ref, args.got)
        print(f"{n} images comparées, {len(diff)} différentes"
              + (f" (ex. {diff[:5]})" if diff else "")
              + (f", {len(missing)} absentes de la référence" if missing else ""))
        sys.exit(1 if diff or missing or n == 0 else 0)


if __name__ == "__main__":
    main()
