"""Calibration des échelles d'activation par couche (PTQ) — §9.2.

Chaque convolution a une échelle de sortie s_y (carte avant un éventuel maxpool) ; maxpool,
upsample et route conservent l'échelle de leur entrée (conventions.md).

- Statistiques : passe avant flottante du réseau fusionné (T4.1) sur des images de
  calibration ; on garde par couche un échantillon uniforme des valeurs et le max exact.
- Candidats d'écrêtage c : percentiles 90, 95, 99 (2026-fata), 99,9, 99,99 de |x| et max ;
  s_y = c / 127. On retient celui qui minimise l'erreur quadratique entre l'activation FP32
  et son fake-quant INT8 [2026-fata#015.1].
- Têtes linéaires : échelle fixe `HEAD_SCALE` = 1/8 (t ∈ [−15,9 ; 15,9]), sauf si
  `head_scale=None` (calibrées comme les autres). Le pas 1/16 du §9.4 écrête les logits de
  Tiny-YOLOv2 (p99,99 de |t| ≈ 19) : 2,5 % des sorties saturent, le softmax s'aplatit et la
  mAP entière perd ~7,5 points (500 images de test : 49,7 contre 57,2 en flottant) ; au pas
  1/8 elle est intacte (57,3). Voir `results/map_int8.md`.
- **Route** : toutes les sources d'une concaténation partagent l'échelle (la plus grande),
  sinon la concaténation par adressage du §10.3 est impossible.
"""

import json

import numpy as np

from yolo.quant.quantize import INPUT_SCALE, QMAX, fake_quant

HEAD_SCALE = 1.0 / 8
PERCENTILES = (90, 95, 99, 99.9, 99.99)
CHOICES = ("mse", "p90", "p95", "p99", "p99.9", "p99.99", "max")


def scale_owners(layers):
    """owner[i] : convolution dont l'échelle s'applique à la sortie de la couche i (−1 =
    entrée du réseau). Pour une route, celle de sa première source (toutes égales).
    """
    owner = []
    for i, layer in enumerate(layers):
        t = layer["type"]
        if t == "conv":
            owner.append(i)
        elif t == "route":
            j = layer["from"][0]
            owner.append(-1 if j < 0 else owner[j])
        else:
            owner.append(-1 if i == 0 else owner[i - 1])
    return owner


def unify_routes(layers, scales):
    """Impose à toutes les sources de chaque route la plus grande de leurs échelles.

    `scales` : {id conv: s} ; rend un nouveau dict et la liste des groupes unifiés.
    """
    owner = scale_owners(layers)
    scales = dict(scales)
    groups = []
    changed = True
    while changed:
        changed = False
        for layer in layers:
            if layer["type"] != "route" or len(layer["from"]) < 2:
                continue
            group = sorted({owner[j] for j in layer["from"]})
            if -1 in group:
                raise ValueError("route sur l'entrée du réseau : échelle non modifiable")
            s = max(scales[g] for g in group)
            for g in group:
                if scales[g] != s:
                    scales[g] = s
                    changed = True
            if group not in groups:
                groups.append(group)
    return scales, groups


class ActStats:
    """Échantillon uniforme des sorties de chaque convolution, et max exact de |x|."""

    def __init__(self, net, per_image=4096, rng=0):
        self.convs = [i for i, layer in enumerate(net.layers) if layer["type"] == "conv"]
        self.per_image = per_image
        self.rng = np.random.default_rng(rng)
        self.samples = {i: [] for i in self.convs}
        self.max = {i: 0.0 for i in self.convs}
        self.images = 0

    def update(self, outs):
        """`outs` : sorties de toutes les couches (`Network.forward(all_outputs=True)`)."""
        for i in self.convs:
            y = outs[i]
            n = y.shape[0]
            flat = y.reshape(n, -1)
            idx = self.rng.integers(0, flat.shape[1], (n, self.per_image))
            self.samples[i].append(np.take_along_axis(flat, idx, axis=1).astype(np.float32))
            self.max[i] = max(self.max[i], float(np.abs(y).max()))
        self.images += outs[0].shape[0]

    def values(self, i):
        return np.concatenate(self.samples[i], axis=0).reshape(-1)


def clip_candidates(values, vmax):
    """{nom: seuil d'écrêtage c} ; c = max(c, ε) pour éviter une échelle nulle."""
    a = np.abs(values)
    out = {f"p{p:g}": float(np.percentile(a, p)) for p in PERCENTILES}
    out["max"] = float(vmax)
    return {k: max(v, 1e-8) for k, v in out.items()}


def quant_mse(values, clip):
    """Erreur quadratique moyenne FP32 / fake-quant INT8 avec s = c/127."""
    v = values.astype(np.float64)
    return float(np.mean((v - fake_quant(v, clip / QMAX)) ** 2))


def clip_rate(values, scale):
    return float(np.mean(np.abs(values) > QMAX * scale))


def choose_scales(net, stats, choice="mse", head_scale=HEAD_SCALE):
    """Échelles de sortie {id conv: s_y} et tableau par couche (pour le rapport)."""
    if choice not in CHOICES:
        raise ValueError(f"choix inconnu : {choice!r}")
    scales, rows = {}, []
    for i in stats.convs:
        layer = net.layers[i]
        v = stats.values(i)
        cands = clip_candidates(v, stats.max[i])
        mses = {k: quant_mse(v, c) for k, c in cands.items()}
        head = layer["act"] == "linear"
        if head and head_scale is not None:
            pick, s = "head", head_scale
        else:
            pick = min(mses, key=mses.get) if choice == "mse" else choice
            s = cands[pick] / QMAX
        scales[i] = s
        rows.append({"id": i, "act": layer["act"], "candidates": cands, "mse": mses,
                     "pick": pick, "scale": s, "clip_rate": clip_rate(v, s)})
    scales, groups = unify_routes(net.layers, scales)
    for r in rows:
        r["final_scale"] = scales[r["id"]]
        r["final_clip_rate"] = clip_rate(stats.values(r["id"]), scales[r["id"]])
    return scales, rows, groups


def calib_dict(net, scales, rows, groups, images, choice, head_scale):
    return {
        "network": net.net["name"],
        "images": images,
        "choice": choice,
        "head_scale": head_scale,
        "input_scale": INPUT_SCALE,
        "act_scales": {str(i): s for i, s in scales.items()},
        "route_groups": groups,
        "layers": rows,
    }


def save_calib(path, calib):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=1) + "\n")


def load_scales(path):
    """(échelle d'entrée, {id conv: s_y}) depuis un `calib.json`."""
    calib = json.loads(path.read_text())
    return calib["input_scale"], {int(k): v for k, v in calib["act_scales"].items()}
