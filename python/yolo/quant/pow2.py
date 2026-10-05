"""Poids sur des niveaux équidistants ou « mixed powers-of-two » — REQ-YOLO, T9.2, §9.2.

2019-ding choisit, couche par couche, entre deux ensembles de niveaux [2019-ding#009.4] :

- **équidistants** : α·{−(M/2 − 1), …, M/2 − 1}, soit `uniform6` (M = 64 : entiers
  ±31) ;
- **puissances de 2** : `mixed6`, un poids de 6 bits = 1 bit de signe, 3 bits « primaires »
  et 2 « secondaires » [2019-ding#008.1]. Ici, en entiers : |q| ∈ {0} ∪ {2^a} ∪
  {2^a + 2^{a−k}}, a ∈ [0, 6] (code primaire 0 = poids nul, 1 à 7 = a + 1), k ∈ {1, 2, 3}
  (code secondaire 0 = absent) et a − k ≥ 0. Le produit q·x devient
  ±((x ≪ a) + (x ≪ (a − k))) : **deux décalages et une addition**, aucun DSP. |q| ≤ 96
  tient dans l'int8 du moteur : le golden et le noyau HLS restent bit-exacts sans
  modification.
- `pot5` : puissances de 2 seules, α·{0, ±2^0, …, ±2^6} (4 bits utiles + signe).
- `uniform4` : entiers ±7, poids de 4 bits paquetés par deux dans `weights.bin` (précision
  mixte par couche, T10.10).

α = s_w,f par **canal de sortie** : pris sur les poids fusionnés (BN comprise), la structure
survit à la fusion (α'_f = s_f·α_f, §9.1). Pour chaque canal, α est cherché sur une grille
max|W_f|/q_max · 2^{−t/16} (t = 0…31) et l'on garde l'erreur quadratique minimale.

`project(W, kind)` est aussi le pas Z de l'ADMM (`yolo.train.admm`).
"""

import json

import numpy as np

KINDS = ("uniform4", "uniform6", "mixed6", "pot5", "int8")
# Bits de stockage des poids dans weights.bin (8 par défaut) : uniform4 est paqueté.
WBITS = {"uniform4": 4}
SCALE_STEPS = 32


def levels(kind):
    """Magnitudes entières autorisées (croissantes, 0 compris)."""
    if kind == "uniform4":
        return np.arange(0, 8, dtype=np.int64)
    if kind == "uniform6":
        return np.arange(0, 32, dtype=np.int64)
    if kind == "int8":
        return np.arange(0, 128, dtype=np.int64)
    mags = {0}
    for a in range(7):
        mags.add(1 << a)
        if kind == "mixed6":
            for k in (1, 2, 3):
                if a - k >= 0:
                    mags.add((1 << a) + (1 << (a - k)))
    if kind not in ("mixed6", "pot5"):
        raise ValueError(f"niveaux inconnus : {kind!r}")
    return np.array(sorted(mags), dtype=np.int64)


def project_int(v, mags):
    """Entier signé de magnitude dans `mags` le plus proche de `v` (réel, en pas de α) ;
    à égale distance, la plus petite magnitude."""
    a = np.abs(np.asarray(v, dtype=np.float64))
    mid = (mags[1:] + mags[:-1]) / 2.0
    idx = np.searchsorted(mid, a, side="left")  # a == mid → magnitude inférieure
    return np.sign(v).astype(np.int64) * mags[idx]


def project(W, kind):
    """Poids réels (F, C, k, k) → (qW int64, s_w (F,)) : niveaux de `kind`, α par canal
    minimisant ‖W_f − α·q_f‖²."""
    W = np.asarray(W, dtype=np.float64)
    mags = levels(kind)
    top = float(mags[-1])
    flat = W.reshape(len(W), -1)
    m = np.abs(flat).max(axis=1)
    m = np.where(m > 0, m, 1.0)
    best_q = np.zeros(flat.shape, dtype=np.int64)
    best_s = m / top
    best_e = np.full(len(W), np.inf)
    for t in range(SCALE_STEPS):
        s = m / top * 2.0 ** (-t / 16)
        q = project_int(flat / s[:, None], mags)
        e = ((flat - q * s[:, None]) ** 2).sum(axis=1)
        better = e < best_e
        best_e = np.where(better, e, best_e)
        best_s = np.where(better, s, best_s)
        best_q[better] = q[better]
    return best_q.reshape(W.shape), best_s


def dequantize(qW, sw):
    return qW * np.asarray(sw)[:, None, None, None]


def shift_add_codes(q):
    """Codes matériels (signe, a, k) de magnitudes `mixed6` ; k = 0 : pas de terme
    secondaire, a = −1 : poids nul."""
    q = np.asarray(q, dtype=np.int64)
    m = np.abs(q)
    a = np.where(m > 0, np.floor(np.log2(np.maximum(m, 1))).astype(np.int64), -1)
    rest = m - np.where(a >= 0, np.left_shift(1, np.maximum(a, 0)), 0)
    k = np.where(rest > 0, a - np.floor(np.log2(np.maximum(rest, 1))).astype(np.int64), 0)
    return np.sign(q), a, k


def shift_add_mul(x, q):
    """q·x par décalages et une addition, à partir des codes (référence du matériel)."""
    x = np.asarray(x, dtype=np.int64)
    sign, a, k = shift_add_codes(q)
    prim = np.where(a >= 0, x << np.maximum(a, 0), 0)
    sec = np.where(k > 0, x << np.maximum(a - k, 0), 0)
    return sign * (prim + sec)


def default_plan(net, kind="mixed6"):
    return {str(i): kind for i, layer in enumerate(net["layers"]) if layer["type"] == "conv"}


def load_plan(path, net):
    """Plan {id conv: niveaux} (JSON) ; défaut : `mixed6` partout."""
    plan = default_plan(net)
    if path:
        plan.update({str(k): v for k, v in json.loads(open(path).read()).items()})
    for k, v in plan.items():
        if v not in KINDS:
            raise ValueError(f"conv {k} : niveaux inconnus {v!r}")
    return {int(k): v for k, v in plan.items()}


def quantize_network_pow2(fused, plan_file=None, plan=None):
    """{id conv: (qW int8, s_w)} des poids de `fused` (réseau à BN fusionnée) selon le plan."""
    plan = plan or load_plan(plan_file, fused.net)
    out = {}
    for i, kind in plan.items():
        qW, sw = project(fused.params[i]["W"], kind)
        out[i] = (qW.astype(np.int8), sw)
    return out
