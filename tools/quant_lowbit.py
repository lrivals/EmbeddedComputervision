"""Modèle entier basse précision exporté (M9.2, M9.3) : PTQ, ou paramètres d'un checkpoint
QAT / ADMM, puis export (manifest, blobs, LUT, dumps) lisible par le golden et le HLS.

    # PTQ 4 bits (pas d'activation puissances de 2 calibrés) → build/m9/models/<net>-w4a4-ptq
    python tools/quant_lowbit.py --scheme w4a4
    # 1re et dernière conv gardées en INT8 (poids et sortie, T12.4)
    python tools/quant_lowbit.py --scheme w4a4 --int8-layers 0,14 --out build/m12/ptq/w4a4-int8ends
    # après QAT (tools/train.py --qat w4a4) : paramètres du checkpoint
    python tools/quant_lowbit.py --scheme w4a4 --checkpoint build/train/qat-w4a4/checkpoint.npz \\
        --out build/m9/models/tiny-yolov2-voc-w4a4-qat
    # poids 6 bits « mixed powers-of-two » de REQ-YOLO (T9.2) ; activations INT8 calibrées ;
    # plan par couche int8 / uniform6 / mixed6 / uniform4 (paqueté 4 bits, T10.10)
    # (build/quant/<net>/calib.json, comme le modèle INT8 de référence) ; --checkpoint : ADMM
    python tools/quant_lowbit.py --weights pow2 [--weights-plan plan.json] [--checkpoint …]

Puis : `tools/eval_quant.py --model-dir <out> --variants int`. Les statistiques de
calibration (500 images VOC2007 trainval, comme `tools/calibrate.py`) sont gardées dans
`build/m9/stats_<net>.npz`. `steps.json` (pas initiaux) sert à `tools/train.py --qat`.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from calibrate import calib_images, collect, load_fused  # noqa: E402
from export_model import write_dumps  # noqa: E402
from yolo.data.voc import load_split  # noqa: E402
from yolo.io.export import export_model  # noqa: E402
from yolo.quant.lowbit import build_qat, load_params, steps_table, to_quant_model  # noqa: E402


def load_stats(net, fused, devkit, n=500):
    """{id conv: valeurs de calibration} (cache npz)."""
    path = ROOT / "build" / "m9" / f"stats_{net}.npz"
    if path.exists():
        with np.load(path) as d:
            return {int(k[1:]): d[k] for k in d.files}
    stats = collect(fused, calib_images(devkit, n))
    values = {i: stats.values(i) for i in stats.convs}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"v{i}": v for i, v in values.items()})
    return values


def lowbit_model(net, scheme, devkit, checkpoint=None, int8_layers=()):
    """(QATNetwork, QuantModel) du schéma wXaY (T9.3) ; `int8_layers` : convs en INT8."""
    fused = load_fused(net)
    values = load_stats(net, fused, devkit)
    qat = build_qat(fused, scheme, values.get, int8_layers=int8_layers)
    if checkpoint:
        load_params(qat, checkpoint)
    return qat, to_quant_model(qat)


def pow2_model(net, calib, checkpoint=None, plan_file=None):
    """QuantModel à poids REQ-YOLO (T9.2) et activations INT8 de `calib`."""
    from yolo.quant.calibrate import load_scales
    from yolo.quant.int_model import QuantModel
    from yolo.quant.pow2 import WBITS, load_plan, quantize_network_pow2

    fused = load_fused(net, dtype=np.float64)
    if checkpoint:
        load_params(fused, checkpoint)
    plan = load_plan(plan_file, fused.net)
    qm = QuantModel.from_fused(fused, *load_scales(calib),
                               weights=quantize_network_pow2(fused, plan=plan))
    for i, kind in plan.items():  # uniform4 : paqueté deux par octet (T10.10)
        qm.convs[i].wbits = WBITS.get(kind, 8)
    return plan, qm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--scheme", default="w4a4", help="wXaY")
    ap.add_argument("--checkpoint", type=Path, default=None, help="checkpoint QAT / ADMM")
    ap.add_argument("--weights", choices=("uniform", "pow2"), default="uniform")
    ap.add_argument("--weights-plan", type=Path, default=None,
                    help="schéma de poids par couche (JSON, T9.2)")
    ap.add_argument("--calib", type=Path, default=None,
                    help="échelles INT8 (--weights pow2 ; défaut build/quant/<net>/calib.json)")
    ap.add_argument("--int8-layers", default="",
                    help="convs gardées en INT8 en wXaY, ex. 0,14 (T12.4)")
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    int8_layers = [int(i) for i in args.int8_layers.split(",") if i]
    if int8_layers and args.weights == "pow2":
        ap.error("--int8-layers : schémas wXaY seulement (en pow2, passer par --weights-plan)")
    tag = ("admm" if args.weights == "pow2" else "qat") if args.checkpoint else "ptq"
    name = f"{args.net}-pow2" if args.weights == "pow2" else f"{args.net}-{args.scheme}"
    out = args.out or ROOT / "build" / "m9" / "models" / f"{name}-{tag}"
    samples = load_split(args.devkit, 2007, "test")[:3]

    if args.weights == "pow2":
        calib = args.calib or ROOT / "build" / "quant" / args.net / "calib.json"
        plan, qm = pow2_model(args.net, calib, args.checkpoint, args.weights_plan)
        export_model(qm, out)
        write_dumps(qm, samples, out, 416)
        (out / "plan.json").write_text(json.dumps({str(k): v for k, v in plan.items()},
                                                  indent=1) + "\n")
        for i, kind in plan.items():
            print(f"conv {i:2d} {kind:9s} niveaux utilisés {len(np.unique(qm.convs[i].qW))}")
        print(f"modèle : {out}")
        return

    qat, qm = lowbit_model(args.net, args.scheme, args.devkit, args.checkpoint, int8_layers)
    export_model(qm, out)
    write_dumps(qm, samples, out, 416)
    steps = steps_table(qat)
    (out / "steps.json").write_text(json.dumps({
        "scheme": args.scheme,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "int8_layers": int8_layers,
        "log2_steps": {str(i): k for i, _, k, learned in steps if learned},
        "layers": [{"id": i, "qmax": q, "log2_step": k, "learned": learned,
                    "w_qmax": qm.convs[i].wqmax,
                    "w_levels": int(len(np.unique(qm.convs[i].qW)))}
                   for i, q, k, learned in steps],
    }, indent=1) + "\n")
    print(f"{'conv':>4} {'qmax':>4} {'log2 s':>6} {'niveaux de poids':>16}")
    for i, q, k, _ in steps:
        print(f"{i:4d} {q:4d} {k:6.0f} {len(np.unique(qm.convs[i].qW)):16d}")
    print(f"modèle : {out}")


if __name__ == "__main__":
    main()
