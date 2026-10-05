"""Compare couche par couche deux jeux de dumps int8 `Lxx.npy` (T5.4, §10.5 étape 3).

    python tools/compare_dumps.py model/tiny-yolov2-voc/dumps/000001 build/golden/out/tiny-yolov2-voc/000001

Rapport par couche : forme, nombre de valeurs différentes, écart max. Si les deux dossiers
ont un `detections.json`, compare aussi les détections (égalité exacte). Code de retour 1 au
moindre écart (contrat « égalité exacte », docs/conventions.md).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compare_layers(ref_dir, test_dir):
    """[(nom, forme, nb de différences, écart max)] ; forme None si la couche manque."""
    rows = []
    for ref in sorted(Path(ref_dir).glob("L*.npy")):
        test = Path(test_dir) / ref.name
        a = np.load(ref)
        if not test.exists():
            rows.append((ref.stem, None, a.size, None))
            continue
        b = np.load(test)
        if a.shape != b.shape:
            rows.append((ref.stem, b.shape, a.size, None))
            continue
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        rows.append((ref.stem, a.shape, int((d != 0).sum()), int(d.max(initial=0))))
    return rows


def compare_detections(ref_dir, test_dir):
    """None si l'un des fichiers manque, sinon True si boîtes, scores et classes sont égaux."""
    paths = [Path(d) / "detections.json" for d in (ref_dir, test_dir)]
    if not all(p.exists() for p in paths):
        return None
    a, b = (json.loads(p.read_text()) for p in paths)
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))
               for k in ("boxes", "scores", "labels"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path, help="dumps Python (model/<net>/dumps/<id>)")
    ap.add_argument("test", type=Path, help="dumps du golden C++")
    args = ap.parse_args()

    rows = compare_layers(args.ref, args.test)
    if not rows:
        sys.exit(f"aucun dump Lxx.npy dans {args.ref}")
    print(f"{args.ref} ↔ {args.test}")
    print(f"{'couche':<7}{'forme':<22}{'différences':>12}{'écart max':>11}")
    bad = 0
    for name, shape, ndiff, dmax in rows:
        if shape is None or dmax is None:
            print(f"{name:<7}{'absente ou forme ≠ ' + str(shape):<22}{ndiff:>12}{'—':>11}")
            bad += 1
            continue
        print(f"{name:<7}{str(tuple(shape)):<22}{ndiff:>12}{dmax:>11}")
        bad += ndiff != 0
    det = compare_detections(args.ref, args.test)
    if det is not None:
        print(f"détections : {'identiques' if det else 'DIFFÉRENTES'}")
        bad += not det
    print("OK : 0 écart" if not bad else f"ÉCHEC : {bad} couche(s) en écart")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
