"""Export synthétique de model/<net>/ pour la CI (T10.13), sans poids Darknet ni VOC.

    python tools/make_ci_model.py                  # les deux réseaux → model/
    python tools/make_ci_model.py --out build/ci/model --net tiny-yolov2-voc

Même chaîne que `make calibrate export` (calibration, `QuantModel`, `export_model`, dumps),
mais :
- poids He tirés avec une graine fixe, BN aléatoire (γ, β, moyenne, variance) puis
  fusionnée, biais de tête aléatoires ;
- images synthétiques : bruit basse fréquence (grille 13 × 13 agrandie) plus un bruit fin,
  dans [0, 1] ;
- calibration sur 8 de ces images, dumps sur 3 autres, nommées 000001 à 000003 comme les
  dumps VOC attendus par les testbenches.

Les formes, donc les cycles, sont celles des vrais réseaux : `perf_model --check` et
l'égalité golden == C-sim à l'octet gardent tout leur sens. Les détections, elles, n'ont
pas de sens : le fichier `SYNTHETIC` de l'export le signale aux tests concernés (écart
post-traitement matériel / flottant, sensible aux scores quasi égaux). Le script échoue si
une couche ne sort que des zéros (test trop faible).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from export_model import write_dumps  # noqa: E402
from yolo.io.export import export_model, load_model  # noqa: E402
from yolo.models.tiny_yolo import PRETRAINED, build  # noqa: E402
from yolo.quant.calibrate import (HEAD_SCALE, ActStats, calib_dict,  # noqa: E402
                                  choose_scales, save_calib)
from yolo.quant.fuse_bn import fuse_network  # noqa: E402
from yolo.quant.int_model import IntNetwork, QuantModel  # noqa: E402

CALIB_IMAGES = 8
DUMP_IDS = ["000001", "000002", "000003"]
# Moyenne des biais de tête : softmax de 20 classes (v2) contre sigmoïdes de 80 (v3).
HEAD_BIAS = {"tiny-yolov2-voc": 1.0, "tiny-yolov3-coco": -1.5}


def synthetic_net(name, seed):
    """Réseau flottant à poids aléatoires reproductibles, BN fusionnée (float64)."""
    rng = np.random.default_rng(seed)
    net = build(name, dtype=np.float64, rng=rng)
    for i, layer in enumerate(net.layers):
        if layer["type"] != "conv":
            continue
        p, c = net.params[i], layer["cout"]
        if layer["bn"]:
            p["gamma"] = rng.uniform(0.5, 1.5, c)
            p["beta"] = rng.normal(0.0, 0.2, c)
            net.state[i] = {"mean": rng.normal(0.0, 0.1, c), "var": rng.uniform(0.5, 2.0, c)}
        else:
            # Tête : poids réduits et biais négatifs, pour quelques dizaines de boîtes.
            p["W"] = p["W"] * 0.3
            p["b"] = rng.normal(HEAD_BIAS[name], 1.0, c)
    return fuse_network(net, dtype=np.float64)


def synthetic_images(n, size, rng):
    """(n, 3, size, size) dans [0, 1] : grille 13 × 13 agrandie + bruit fin."""
    coarse = rng.uniform(0.0, 1.0, (n, 3, 13, 13))
    rep = -(-size // 13)
    x = np.repeat(np.repeat(coarse, rep, axis=2), rep, axis=3)[:, :, :size, :size]
    x = x + rng.normal(0.0, 0.08, x.shape)
    return np.clip(x, 0.0, 1.0)


def make(name, out_root, seed, size):
    fused = synthetic_net(name, seed)
    rng = np.random.default_rng(seed + 1)
    stats = ActStats(fused)
    stats.update(fused.forward(synthetic_images(CALIB_IMAGES, size, rng), train=False,
                               all_outputs=True))
    scales, rows, groups = choose_scales(fused, stats, "mse", HEAD_SCALE)
    calib = calib_dict(fused, scales, rows, groups, stats.images, "mse", HEAD_SCALE)
    save_calib(ROOT / "build" / "ci" / name / "calib.json", calib)

    qm = QuantModel.from_fused(fused, calib["input_scale"], scales)
    out = (out_root / name).resolve()
    m = export_model(qm, out)
    # Témoin lu par les tests qui n'ont de sens que sur de vraies détections.
    (out / "SYNTHETIC").write_text(f"tools/make_ci_model.py, graine {seed}\n")
    print(f"export {out} : {len(m['layers'])} couches (synthétique, graine {seed})")

    xs = synthetic_images(len(DUMP_IDS), size, rng)
    samples = [{"id": i, "x": x} for i, x in zip(DUMP_IDS, xs)]
    reloaded, _ = load_model(out)
    write_dumps(reloaded, samples, out, size)

    first = np.load(out / "dumps" / DUMP_IDS[0] / "input.npy")
    a, b = IntNetwork(qm).forward(first), IntNetwork(reloaded).forward(first)
    assert all(np.array_equal(a[k], b[k]) for k in a), "modèle relu != modèle exporté"
    dead = [f.name for f in sorted((out / "dumps" / DUMP_IDS[0]).glob("L*.npy"))
            if not np.load(f).any()]
    if dead:
        sys.exit(f"{name} : couches entièrement nulles {dead}, test trop faible")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", action="append", choices=sorted(PRETRAINED))
    ap.add_argument("--out", type=Path, default=ROOT / "model")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--size", type=int, default=416)
    args = ap.parse_args()
    nets = sorted(PRETRAINED)
    for name in args.net or nets:
        make(name, args.out, args.seed + 10 * nets.index(name), args.size)


if __name__ == "__main__":
    main()
