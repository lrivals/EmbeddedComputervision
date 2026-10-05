"""Élagage structuré de Tiny-YOLOv2 par norme de filtre (T10.11) : .cfg et .weights réduits.

    python tools/prune.py --rate 0.5 --out build/m10/prune/r50
    # puis affinage, calibration, export, mAP et cycles :
    python tools/train.py --net build/m10/prune/r50/pruned.cfg \\
        --init build/m10/prune/r50/pruned.weights --iters 300 --out build/m10/prune/r50/train
    python tools/calibrate.py --net build/m10/prune/r50/pruned.cfg \\
        --weights build/m10/prune/r50/train/final.weights --out build/m10/prune/r50/calib.json
    python tools/export_model.py --net … --weights … --calib … --out build/m10/prune/r50/model
    python tools/perf_model.py --manifest build/m10/prune/r50/model/manifest.json

Taux uniforme sur les convs élagables (`yolo.prune.prunable`), ou sur celles de `--layers`
(par exemple la queue L10-L13, la moins sensible d'après T10.10), filtres gardés au multiple de
Tm = 32 le plus proche. Le rapport par couche (cout avant / après) est écrit dans
`<out>/prune.json`.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from yolo.io.darknet_weights import load_darknet_weights, save_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import CFG_DIR, CFG_FILES, PRETRAINED, build  # noqa: E402
from yolo.prune import prune, select, write_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc", choices=sorted(PRETRAINED))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--rate", type=float, required=True, help="part de filtres retirés")
    ap.add_argument("--multiple", type=int, default=32, help="filtres gardés : multiple de Tm")
    ap.add_argument("--layers", default="",
                    help="convs élaguées, ex. 10,12,13 (défaut : toutes les élagables)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    net = build(args.net, dtype=np.float32)
    load_darknet_weights(net, args.weights or ROOT / "weights" / PRETRAINED[args.net])
    layers = {int(t) for t in args.layers.split(",") if t} or None
    keep = select(net, args.rate, args.multiple, layers)
    small = prune(net, keep)

    args.out.mkdir(parents=True, exist_ok=True)
    text = (CFG_DIR / CFG_FILES[args.net]).read_text()
    cfg = args.out / "pruned.cfg"
    cfg.write_text(write_cfg(text, {i: len(idx) for i, idx in keep.items()}))
    save_darknet_weights(small, args.out / "pruned.weights")
    params = [sum(p.size for p in n.params[i].values())
              for n in (net, small) for i, layer in enumerate(n.layers) if layer["type"] == "conv"]
    half = len(params) // 2
    report = {"net": args.net, "rate": args.rate, "multiple": args.multiple,
              "only": sorted(layers) if layers else None,
              "params_before": int(sum(params[:half])), "params_after": int(sum(params[half:])),
              "layers": {str(i): {"cout": net.layers[i]["cout"], "kept": len(idx)}
                         for i, idx in keep.items()}}
    (args.out / "prune.json").write_text(json.dumps(report, indent=1) + "\n")
    for i, idx in keep.items():
        print(f"L{i:02d} : {net.layers[i]['cout']:5d} → {len(idx):5d} filtres")
    print(f"paramètres : {report['params_before']:_} → {report['params_after']:_} "
          f"({100 * report['params_after'] / report['params_before']:.1f} %)".replace("_", " "))
    print(f"→ {cfg}, {args.out / 'pruned.weights'}")


if __name__ == "__main__":
    main()
