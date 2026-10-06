"""Cfg Darknet d'affinage à N classes, depuis une cfg de base (T11.4, T11.5, T11.7).

    # Tiny-YOLOv3 à 8 classes (KITTI), ancres de tools/kmeans_anchors.py --dataset kitti
    python tools/make_cfg.py --base tiny-yolov3-voc --dataset kitti \\
        --anchors "8,20 15,40 …" --out build/m11/cfg/tiny-yolov3-kitti.cfg
    # entrée non carrée 640 × 192 (T11.4) ; ancres de kmeans_anchors.py --size 640x192
    python tools/make_cfg.py --base tiny-yolov3-voc --dataset kitti --size 640x192 \\
        --anchors "…" --out build/m11/cfg/tiny-yolov3-kitti-640x192.cfg
    # entrée à un canal (T11.7, thermique)
    python tools/make_cfg.py --base tiny-yolov3-voc --dataset flir --channels 1 \\
        --out build/m11/cfg/tiny-yolov3-flir-c1.cfg

Seules changent les lignes `classes` des sections [yolo] / [region], `filters` de la
convolution qui précède chacune (`ancres de la tête × (5 + N)`) et, avec `--anchors`, les
ancres (pixels à l'entrée 416, ou à l'entrée `--size` si elle n'est pas carrée,
docs/conventions.md ; converties en cellules de 32 px pour [region]). `--size` et
`--channels` changent `width`, `height` et `channels` de [net]. Le réseau produit se charge
avec `--net <chemin du .cfg>` dans tools/train.py et tools/eval_*.py.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.datasets import DATASETS  # noqa: E402
from yolo.data.letterbox import as_hw, parse_size  # noqa: E402
from yolo.models.cfg import REGION_STRIDE  # noqa: E402
from yolo.models.tiny_yolo import CFG_DIR, CFG_FILES  # noqa: E402


def _sections(lines):
    """[(nom, indice de la ligne d'en-tête)] dans l'ordre du fichier."""
    return [(m.group(1), i) for i, line in enumerate(lines)
            if (m := re.match(r"\s*\[(\w+)\]", line))]


def _opt(lines, start, end, key):
    """Indice de la ligne `key=…` entre `start` et `end`, ou None."""
    for i in range(start, end):
        if re.match(rf"\s*{key}\s*=", lines[i]):
            return i
    return None


def make_cfg(text, classes, anchors=None, size=None, channels=None):
    """Texte de la cfg avec `classes` classes (et `anchors` [(w, h)] en pixels de l'entrée de
    référence) ; `size` (S ou (H, W)) et `channels` : entrée de [net], inchangée si None."""
    lines = text.splitlines()
    secs = _sections(lines) + [("", len(lines))]
    for k, (name, start) in enumerate(secs[:-1]):
        end = secs[k + 1][1]
        if name == "net":
            h, w = as_hw(size) if size is not None else (None, None)
            for key, val in (("width", w), ("height", h), ("channels", channels)):
                if val is not None:
                    lines[_opt(lines, start, end, key)] = f"{key}={val}"
        if name not in ("yolo", "region"):
            continue
        i = _opt(lines, start, end, "classes")
        lines[i] = f"classes={classes}"
        if anchors is not None:
            if name == "region":
                vals = [v / REGION_STRIDE for wh in anchors for v in wh]
                txt = ",  ".join(f"{a:.4g},{b:.4g}" for a, b in zip(vals[::2], vals[1::2]))
            else:
                txt = ",  ".join(f"{round(w)},{round(h)}" for w, h in anchors)
            lines[_opt(lines, start, end, "anchors")] = f"anchors = {txt}"
            j = _opt(lines, start, end, "num")
            if j is not None:
                lines[j] = f"num={len(anchors)}"
        if name == "yolo":
            n_anchors = len(lines[_opt(lines, start, end, "mask")].split("=")[1].split(","))
        else:
            n_anchors = int(lines[_opt(lines, start, end, "num")].split("=")[1])
        conv = [s for s in secs[:k] if s[0] == "convolutional"][-1][1]
        f = _opt(lines, conv, start, "filters")
        lines[f] = f"filters={n_anchors * (5 + classes)}"
    return "\n".join(lines) + "\n"


def parse_anchors(text):
    """« w,h  w,h … » → [(w, h)]."""
    vals = [float(v) for v in re.split(r"[,\s]+", text.strip()) if v]
    return list(zip(vals[::2], vals[1::2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="tiny-yolov3-voc", help="réseau de CFG_FILES ou chemin")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--classes", type=int)
    g.add_argument("--dataset", choices=sorted(DATASETS), help="nombre de classes du jeu")
    ap.add_argument("--anchors", default=None,
                    help="« w,h  w,h … » en pixels à 416 (ou à l'entrée --size non carrée)")
    ap.add_argument("--size", type=parse_size, default=None,
                    help="entrée S ou LxH (ex. 640x192) ; défaut : celle de la base")
    ap.add_argument("--channels", type=int, choices=(1, 3), default=None,
                    help="canaux d'entrée (1 : thermique) ; défaut : ceux de la base")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = CFG_DIR / CFG_FILES[args.base] if args.base in CFG_FILES else Path(args.base)
    n = args.classes or len(DATASETS[args.dataset].classes)
    anchors = parse_anchors(args.anchors) if args.anchors else None
    text = make_cfg(base.read_text(), n, anchors, args.size, args.channels)
    what = f"{args.dataset} ({n} classes)" if args.dataset else f"{n} classes"
    head = f"# Générée par tools/make_cfg.py depuis {base.name} : {what}.\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(head + text)
    print(f"cfg : {args.out}")


if __name__ == "__main__":
    main()
