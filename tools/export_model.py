"""Export du modèle entier vers model/<net>/ : manifest, blobs, LUT et dumps (T4.7).

    python tools/export_model.py --net tiny-yolov2-voc
    python tools/export_model.py --net tiny-yolov3-coco

Prérequis : `tools/calibrate.py` (build/quant/<net>/calib.json). Les dumps de 3 images de
VOC2007 test servent de référence au golden C++ (M5) : `dumps/<id>/input.npy` (int8,
(1, 3, 416, 416)), `L00.npy` … (sortie int8 de chaque couche, numéros du §3 ; une conv à
maxpool fusionné a sa carte avant pooling en `Lxx` et la carte poolée en `Lxx+1`) et
`detections.json` (décodage entier §9.4, seuil 0,25, NMS 0,45, boîtes `(cx, cy, w, h)`
normalisées dans l'image letterbox).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from calibrate import load_fused, net_tag  # noqa: E402
from yolo.data.voc import load_split  # noqa: E402
from yolo.infer.nms import CONF_THR, IOU_THR, postprocess_int  # noqa: E402
from yolo.infer.pipeline import preprocess  # noqa: E402
from yolo.io.export import export_model, load_model  # noqa: E402
from yolo.quant.calibrate import load_scales  # noqa: E402
from yolo.quant.int_model import IntNetwork, QuantModel, head_luts  # noqa: E402
from yolo.quant.quantize import quantize_input  # noqa: E402

DUMP_IMAGES = 3


def _preprocessed(s, size):
    """(image float (3, size, size), (w, h)) : `s["x"]` déjà prétraitée, sinon `s["image"]`."""
    if "x" in s:
        return s["x"], (size, size)
    from PIL import Image

    with Image.open(s["image"]) as img:
        return preprocess(img, size)


def write_dumps(qm, samples, out_dir, size):
    inet, luts = IntNetwork(qm, "int64"), head_luts(qm)
    for s in samples:
        d = out_dir / "dumps" / s["id"]
        d.mkdir(parents=True, exist_ok=True)
        x, (w, h) = _preprocessed(s, size)
        qx = quantize_input(x[None])
        np.save(d / "input.npy", qx)
        outs = inet.forward(qx, all_outputs=True)
        for i, o in enumerate(outs):
            np.save(d / f"L{i:02d}.npy", o)
        heads = {i: outs[i] for i in qm.outputs}
        boxes, scores, labels = postprocess_int(heads, qm.net, luts, CONF_THR, IOU_THR)[0]
        (d / "detections.json").write_text(json.dumps({
            "image": s["id"], "width": w, "height": h, "conf": CONF_THR, "iou": IOU_THR,
            "boxes": boxes.tolist(), "scores": scores.tolist(), "labels": labels.tolist(),
        }, indent=1) + "\n")
        print(f"dumps {d} : {len(outs)} couches, {len(labels)} détections")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc", help="réseau pré-entraîné, ou chemin d'un .cfg (élagué, T10.11 ; --weights requis)")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--size", type=int, default=416)
    args = ap.parse_args()
    calib = args.calib or ROOT / "build" / "quant" / net_tag(args.net) / "calib.json"
    out = (args.out or ROOT / "model" / net_tag(args.net)).resolve()

    qm = QuantModel.from_fused(load_fused(args.net, args.weights, np.float64),
                               *load_scales(calib))
    m = export_model(qm, out)
    print(f"export {out} : {len(m['layers'])} couches, blobs {m['blobs']}")

    # Contrôle : le modèle relu des blobs donne les mêmes têtes que l'original.
    samples = load_split(args.devkit, 2007, "test")[:DUMP_IMAGES]
    reloaded, _ = load_model(out)
    write_dumps(reloaded, samples, out, args.size)
    first = np.load(out / "dumps" / samples[0]["id"] / "input.npy")
    a, b = IntNetwork(qm).forward(first), IntNetwork(reloaded).forward(first)
    assert all(np.array_equal(a[k], b[k]) for k in a), "modèle relu != modèle exporté"
    print("relecture des blobs : sorties identiques")


if __name__ == "__main__":
    main()
