"""mAP VOC2007 test aux trois stades : flottant, entier Python, FPGA (T8.2, §11).

    python tools/map_stades.py --net tiny-yolov2-voc      # → results/map_stades.md

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

from yolo.data.voc import VOC_CLASSES, load_split  # noqa: E402
from yolo.infer.metrics import evaluate, to_voc_pixels  # noqa: E402


def read_jsonl(paths):
    """{image: (boxes, scores, labels)} de un ou plusieurs fichiers JSONL."""
    out = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["image"]] = (d["boxes"], d["scores"], d["labels"])
    return out


def to_per_class(dets, samples):
    """Format de `evaluate` : {c: (ids, scores, coins pixels VOC)}."""
    per = {c: ([], [], []) for c in range(len(VOC_CLASSES))}
    for s in samples:
        boxes, scores, labels = dets[s["id"]]
        px = to_voc_pixels(np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                           s["width"], s["height"])
        for b, sc, c in zip(px, scores, labels):
            per[c][0].append(s["id"])
            per[c][1].append(sc)
            per[c][2].append(b)
    return {c: (v[0], np.array(v[1]), np.array(v[2]).reshape(-1, 4)) for c, v in per.items()}


def compare(ref, got, ids):
    """(images absentes, images différentes) ; égalité exacte des flottants."""
    missing = [i for i in ids if i not in got]
    diff = [i for i in ids if i in got and got[i] != ref[i]]
    return missing, diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--dir", type=Path, default=None, help="défaut build/m8/<net>")
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "map_stades.md")
    args = ap.parse_args()
    d = args.dir or ROOT / "build" / "m8" / args.net

    samples = load_split(args.devkit, 2007, "test")
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
    aps_int, m_int = evaluate(to_per_class(ref, samples), samples, len(VOC_CLASSES), use_07=True)
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
        aps, m = evaluate(to_per_class(got, samples), samples, len(VOC_CLASSES), use_07=True)
        stages.append((name, m, list(aps), eq + ("" if not diff else " — **ÉCART**")))
        if diff:
            print(f"{name} : {len(diff)} images différentes, ex. {diff[:5]}")

    lines = [
        "# mAP aux trois stades (T8.2)", "",
        f"{args.net}, VOC2007 test ({len(samples)} images), redimensionnement direct 416×416 "
        "(`stretch`, Pillow), seuil 0,005, NMS 0,45, AP 11 points. Généré par "
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
    for c, cls in enumerate(VOC_CLASSES):
        lines.append(f"| {cls} | " + " | ".join(f"{aps[c] * 100:.1f}" for _, aps in done) + " |")
    lines += ["", "## Reproduire", "", "```",
              "make m8-inputs          # build/m8/<net>/inputs.bin (2,6 Go) + ids.txt",
              "make m8-int             # eval_quant float,int --resize stretch --save-dets",
              "make bench-sim          # stade FPGA en C-sim, lots parallèles",
              "# carte : yolo_bench --inputs ... --dets build/m8/<net>/board/dets_0.jsonl "
              "(results/protocole.md)",
              "make map-stades", "```", ""]
    args.out.write_text("\n".join(lines))
    for name, m, _, eq in stages:
        print(f"{name:45s} {'—' if m is None else f'{m * 100:.2f}':>6s}  {eq}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
