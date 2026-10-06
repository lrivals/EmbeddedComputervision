"""mAP VOC2007 test aux trois stades : flottant, entier Python, FPGA (T8.2, §11).

    python tools/map_stades.py --net tiny-yolov2-voc      # → results/map_stades.md
    # autre jeu (T11.1) : entrées et détections dans build/m11/<jeu>/<net>/
    python tools/map_stades.py --net tiny-yolov3-coco --dataset coco \
        --out results/map_stades_coco.md

Entrées (make m8-int, make bench-sim, mesures carte) dans build/m8/<net>/ :
- `eval_float_int.json` : mAP flottante et entière de `tools/eval_quant.py --resize stretch` ;
- `int.jsonl` : détections du modèle entier, une ligne par image (`--save-dets`) ;
- `sim/dets_*.jsonl` : `yolo_bench --backend sim` (noyau C-sim derrière le driver ARM) ;
- `board/dets_*.jsonl` : `yolo_bench` sur la KV260 (backend uio), si présent.
Les stades FPGA lisent les mêmes entrées int8 que le modèle entier (`tools/make_inputs.py`) :
leurs détections doivent être **identiques** à celles de `int.jsonl`, image par image.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from yolo.data import datasets  # noqa: E402
from yolo.infer import coco_eval  # noqa: E402
from yolo.infer.metrics import evaluate, to_voc_pixels  # noqa: E402
from yolo.models.tiny_yolo import load_cfg  # noqa: E402


def read_jsonl(paths):
    """{image: (boxes, scores, labels)} de un ou plusieurs fichiers JSONL."""
    out = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["image"]] = (d["boxes"], d["scores"], d["labels"])
    return out


def to_per_class(dets, samples, det_lut):
    """Format de `evaluate` : {c: (ids, scores, coins pixels VOC)}, c classe évaluée."""
    per = {c: ([], [], []) for c in range(len(det_lut))}
    for s in samples:
        boxes, scores, labels = dets[s["id"]]
        px = to_voc_pixels(np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                           s["width"], s["height"])
        for b, sc, c in zip(px, scores, labels):
            per[c][0].append(s["id"])
            per[c][1].append(sc)
            per[c][2].append(b)
    per = {c: (v[0], np.array(v[1]), np.array(v[2]).reshape(-1, 4)) for c, v in per.items()}
    return datasets.remap_detections(per, det_lut)


def compare(ref, got, ids):
    """(images absentes, images différentes) ; égalité exacte des flottants."""
    missing = [i for i in ids if i not in got]
    diff = [i for i in ids if i in got and got[i] != ref[i]]
    return missing, diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--dir", type=Path, default=None, help="défaut build/m8/<net>")
    datasets.add_args(ap)
    ap.add_argument("--metric", choices=("voc", "coco"), default=None,
                    help="défaut : celle du jeu (coco pour coco et flir)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "map_stades.md")
    ap.add_argument("--json", type=Path, default=None,
                    help="stades en JSON pour les figures (défaut <dir>/map_stades.json, T13.13)")
    ap.add_argument("--no-figures", action="store_true", help="pas de figure en fin de run")
    args = ap.parse_args()
    d = args.dir or (ROOT / "build" / "m8" / args.net if args.dataset == "voc"
                     else ROOT / "build" / "m11" / args.dataset / args.net)
    metric = args.metric or datasets.DATASETS[args.dataset].metric
    split = datasets.split_of(args)
    view = datasets.eval_view(args.dataset, load_cfg(args.net)["classes"])
    names = view.names

    def score(dets):
        """(AP par classe, valeur principale) : mAP VOC 11 points ou AP COCO."""
        per = to_per_class(dets, samples, view.det_lut)
        if metric == "coco":
            res = coco_eval.evaluate(per, samples, len(names))
            return res["ap_class"], res["AP"]
        return evaluate(per, samples, len(names), use_07=True)

    samples = datasets.remap(datasets.load_args(args), view.gt_lut)
    ids = [s["id"] for s in samples]
    ev = json.loads((d / "eval_float_int.json").read_text())
    if ev["images"] != len(samples) or ev["resize"] != "stretch":
        sys.exit(f"{d}/eval_float_int.json : {ev['images']} images en {ev['resize']}, "
                 f"attendu {len(samples)} en stretch")
    ref = read_jsonl([d / "int.jsonl"])
    missing, _ = compare(ref, ref, ids)
    if missing:
        sys.exit(f"int.jsonl : {len(missing)} images absentes")

    stages = [("Flottant (BN fusionnée, NumPy float32)", ev["results"]["float"]["map"],
               ev["results"]["float"]["aps"], "—")]
    aps_int, m_int = score(ref)
    if abs(m_int - ev["results"]["int"]["map"]) > 1e-12:
        sys.exit(f"mAP de int.jsonl ({m_int}) ≠ eval_quant ({ev['results']['int']['map']})")
    stages.append(("Entier Python (`IntNetwork`, bit-exact)", m_int, list(aps_int), "référence"))

    for name, sub in [("FPGA, C-sim (driver ARM, backend sim, PC)", "sim"),
                      ("FPGA, KV260 (backend uio)", "board")]:
        files = sorted((d / sub).glob("dets_*.jsonl"))
        got = read_jsonl(files)
        if not files:
            stages.append((name, None, None, "à mesurer"))
            continue
        missing, diff = compare(ref, got, ids)
        eq = f"{len(ids) - len(missing) - len(diff)} / {len(ids)} images identiques"
        if missing:
            stages.append((name, None, None, f"{eq}, {len(missing)} absentes (incomplet)"))
            continue
        aps, m = score(got)
        stages.append((name, m, list(aps), eq + ("" if not diff else " — **ÉCART**")))
        if diff:
            print(f"{name} : {len(diff)} images différentes, ex. {diff[:5]}")

    where = (f"VOC{split.replace(':', ' ')}" if args.dataset == "voc"
             else f"{args.dataset} {split}")
    how = "AP 11 points" if metric == "voc" else "AP COCO @[.5:.95] (101 points)"
    lines = [
        "# mAP aux trois stades (T8.2)" if args.dataset == "voc"
        else f"# AP aux trois stades, {args.dataset} (T11.1)", "",
        f"{args.net}, {where} ({len(samples)} images), redimensionnement direct 416×416 "
        f"(`stretch`, Pillow), seuil 0,005, NMS 0,45, {how}. Généré par "
        "`python tools/map_stades.py` (`make map-stades`).", "",
        "Les stades entier et FPGA reçoivent **les mêmes octets** : entrées int8 de "
        "`tools/make_inputs.py` (`preprocess` + `quantize_input`), lues par "
        "`sw/app/yolo_bench --inputs` ; le décodage JPEG de la carte (stb) n'intervient pas. "
        "Égalité exigée : boîtes, scores et classes identiques, image par image (flottants "
        "comparés exactement). Stade C-sim : `make bench-sim` (noyau compilé sans `ap_int`, "
        "`-DACC_NO_APINT`, même arithmétique ; les 1 832 premières images ont aussi été "
        "calculées avec `ap_int`, mêmes détections).", "",
        "| Stade | mAP | Égalité avec l'entier |", "|---|---|---|"]
    for name, m, _, eq in stages:
        lines.append(f"| {name} | {'—' if m is None else f'**{m * 100:.2f}**'} | {eq} |")
    done = [(n, aps) for n, m, aps, _ in stages if m is not None]
    lines += ["", "## AP par classe", "",
              "| Classe | " + " | ".join(n.split(" (")[0] for n, _ in done) + " |",
              "|---" * (len(done) + 1) + "|"]
    for c, cls in enumerate(names):
        lines.append(f"| {cls} | " + " | ".join(f"{aps[c] * 100:.1f}" for _, aps in done) + " |")
    lines += ["", "## Reproduire", "", "```",
              "make m8-inputs          # build/m8/<net>/inputs.bin (2,6 Go) + ids.txt",
              "make m8-int             # eval_quant float,int --resize stretch --save-dets",
              "make bench-sim          # stade FPGA en C-sim, lots parallèles",
              "# carte : yolo_bench --inputs ... --dets build/m8/<net>/board/dets_0.jsonl "
              "(results/protocole.md)",
              "make map-stades", "```", ""]
    if args.dataset == "voc":
        lines[-1:-1] = ["", "Figure : `results/figures/resultats/map_stades.png` "
                        "(`python -m tools.figures map_stades`, T13.13)."]
    args.out.write_text("\n".join(lines))
    keys = ("flottant", "entier", "csim", "carte")
    js = args.json or d / "map_stades.json"
    js.write_text(json.dumps({
        "net": args.net, "dataset": args.dataset, "split": split, "metric": metric,
        "images": len(samples), "classes": list(names),
        "stages": [{"key": k, "name": n.split(" (")[0], "label": n, "map": m,
                    "aps": None if aps is None else [float(a) for a in aps], "eq": eq}
                   for k, (n, m, aps, eq) in zip(keys, stages)]}, indent=1) + "\n")
    for name, m, _, eq in stages:
        print(f"{name:45s} {'—' if m is None else f'{m * 100:.2f}':>6s}  {eq}")
    print(f"→ {args.out}, {js}")
    if not args.no_figures:
        sys.path.insert(0, str(ROOT))
        from tools.figures.auto import after_run

        after_run("map-stades", js)


if __name__ == "__main__":
    main()
