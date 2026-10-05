"""Génère model/example_manifest.json : Tiny-YOLOv3 VOC complet, valeurs fictives (T0.4).

Les formes viennent de yolo.models.specs ; l'allocation des tampons et les offsets de blobs
sont ceux de l'export réel (`yolo.io.export`, T4.7). Les échelles sont fictives (puissances
de 2).

    python tools/make_example_manifest.py           # écrit le fichier
    python tools/make_example_manifest.py --check   # échoue si le fichier versionné diffère
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.io.export import build_manifest  # noqa: E402
from yolo.models.specs import TINY_YOLOV3_VOC  # noqa: E402
from yolo.quant.calibrate import scale_owners  # noqa: E402

OUTPUT = ROOT / "model" / "example_manifest.json"
SHIFT = 31
IN_SCALE = 2.0**-7    # pixels [0, 1] en int8 symétrique
LEAKY_SCALE = 2.0**-4
LINEAR_SCALE = 2.0**-3


def build():
    """Placement des tampons de `yolo.io.export.layout` (§10.3), échelles fictives."""
    net = TINY_YOLOV3_VOC
    owner = scale_owners(net["layers"])

    def scale_of(i):
        o = owner[i] if i >= 0 else -1
        if o < 0:
            return IN_SCALE
        return LEAKY_SCALE if net["layers"][o]["act"] == "leaky" else LINEAR_SCALE

    return build_manifest(net, IN_SCALE, scale_of, lambda i: SHIFT)


def dumps(manifest):
    return json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = dumps(build())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            sys.exit(f"{OUTPUT.relative_to(ROOT)} n'est pas à jour : relancer sans --check")
        print("à jour")
    else:
        OUTPUT.write_text(text)
        print(f"écrit {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
