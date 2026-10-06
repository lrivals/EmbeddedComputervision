"""Entrées int8 du stade FPGA de la mAP (T8.2) : VOC2007 test → inputs.bin + ids.txt.

    python tools/make_inputs.py --net tiny-yolov2-voc        # → build/m8/tiny-yolov2-voc/
    python tools/make_inputs.py --net tiny-yolov3-coco --dataset coco
                                                    # → build/m11/coco/tiny-yolov3-coco/

Même prétraitement que le modèle entier de `tools/eval_quant.py --resize stretch` :
`preprocess(img, 416, "stretch", "pil")` puis `quantize_input`. La carte lit ces octets au
lieu de décoder les JPEG (stb ≠ PIL) : les détections sont alors comparables bit à bit.
Format : (3, S, S) int8 par image, concaténées dans l'ordre de `ids.txt`, sans en-tête
(lu par `sw/app/yolo_bench --inputs`).
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from yolo.data import datasets  # noqa: E402
from yolo.infer.pipeline import MODES, preprocess  # noqa: E402
from yolo.quant.quantize import quantize_input  # noqa: E402


def _load(task):
    from PIL import Image

    path, size, mode = task
    with Image.open(path) as img:
        x, _ = preprocess(img, size, mode)
    return quantize_input(x).tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    datasets.add_args(ap)
    ap.add_argument("--subset", type=int, default=0, help="n premières images seulement")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--resize", choices=MODES, default="stretch")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="défaut build/m8/<net> (VOC), build/m11/<jeu>/<net>")
    args = ap.parse_args()

    samples = datasets.load_args(args)
    if args.subset:
        samples = samples[:args.subset]
    out = args.out or (ROOT / "build" / "m8" / args.net if args.dataset == "voc"
                       else ROOT / "build" / "m11" / args.dataset / args.net)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tasks = [(s["image"], args.size, args.resize) for s in samples]
    with mp.Pool(args.workers) as pool, (out / "inputs.bin").open("wb") as f:
        for k, b in enumerate(pool.imap(_load, tasks, chunksize=8), 1):
            assert len(b) == 3 * args.size * args.size
            f.write(b)
            if k % 1000 == 0:
                print(f"{k}/{len(samples)} images  {time.time() - t0:.0f} s", flush=True)
    (out / "ids.txt").write_text("".join(s["id"] + "\n" for s in samples))
    print(f"{len(samples)} entrées ({np.dtype(np.int8).name}, 3×{args.size}×{args.size}, "
          f"{args.resize}) → {out}/inputs.bin, ids.txt")


if __name__ == "__main__":
    main()
