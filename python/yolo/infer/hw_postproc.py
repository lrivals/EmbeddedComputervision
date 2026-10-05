"""Post-traitement tout entier, tel que le matériel le fera (T9.1.1, §8.1, §8.2, §9.4, §10.3).

Référence bit-exacte du noyau `yolo_post` (hls/kernels/postproc.cpp) et de son miroir C++
(cpp/golden/include/golden/hw_postproc.hpp). Aucun flottant entre la tête int8 et les
boîtes :

- **Seuil d'objectness** sur l'entier t_o (§9.4), puis tables Q16 de `lut.py`.
- **Coins en pixels Q4** (entrée 416 = `ANCHOR_REF`, pas de grille 416/S puissance de 2) :
  cx = ((σ_q + j·2¹⁶) · pas) ≫ₐ 12, w = (ancre_Q8 · e_q) ≫ₐ (8 + f − 4), écrêtée à
  `W_MAX` ; x₁ = cx − ⌊w/2⌋, x₂ = x₁ + w (≫ₐ : décalage arrondi, `rshift_round`).
- **Scores Q16** : v3 (σ_o · σ_c) ≫ₐ 16 ; v2 softmax en trois étages avec **une
  réciproque par cellule**, r = ⌊2³² / Σ e⌋, puis (σ_o · e_c · r) ≫ₐ 32. Une candidate
  (cellule, classe) passe si score > ⌊θ·2¹⁶⌋.
- **IoU sans division** : avec θ = p/q, IoU > θ ⟺ (p + q)·inter > p·(a₁ + a₂).
- **NMS sans tri** [2024-zhang#014.2] : les candidates arrivent dans l'ordre du flux
  (tête, ancre, ligne, colonne, classe). Une candidate recouverte (même classe, IoU > θ)
  par une sélectionnée de score ≥ au sien est rejetée ; sinon elle prend la place de la
  première sélectionnée qu'elle recouvre et invalide les autres, ou s'ajoute à la fin.
  `cap` emplacements au plus ; au-delà la candidate est perdue et comptée (`overflow`).

Différence avec la NMS triée (§8.2) : une boîte éliminée par une sélectionnée qui est
ensuite remplacée n'est pas reconsidérée, et l'ordre d'arrivée compte. L'écart de mAP est
mesuré par `tools/eval_quant.py --variants int,int-hwpp`.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from yolo.data.targets import heads
from yolo.quant.lut import FRAC_BITS, OFFSET, SOFTMAX_OFFSET, logit_threshold_q

ANCHOR_REF = 416  # ancres en pixels pour une entrée de 416
ANCHOR_FRAC = 8   # ancres en Q8
BOX_FRAC = 4      # coins en pixels Q4
W_MAX = (1 << 20) - 1  # largeur / hauteur max (Q4) : aires < 2⁴⁰, (p+q)·inter < 2⁶³
CAP = 256         # emplacements de sélection du noyau


def rshift_round(v, n):
    """(v + 2^{n−1}) ≫ n, décalage arithmétique (même convention que `int_layers`)."""
    v = np.asarray(v, dtype=np.int64)
    return (v + (np.int64(1) << (n - 1))) >> n if n > 0 else v


def iou_fraction(iou):
    """θ = p/q de plus petit dénominateur ≤ 1000 égal à θ à 1e-12 près (0,45 → 9/20)."""
    for q in range(1, 1001):
        p = int(np.floor(iou * q + 0.5))
        if abs(p / q - iou) < 1e-12:
            return p, q
    f = Fraction(iou).limit_denominator(1000)
    return f.numerator, f.denominator


def conf_q16(conf):
    return int(np.floor(conf * (1 << FRAC_BITS)))


def anchors_q8(anchors_px):
    a = np.asarray(anchors_px, dtype=np.float64).reshape(-1, 2)
    return np.floor(a * (1 << ANCHOR_FRAC) + 0.5).astype(np.int64)


@dataclass
class HwHead:
    """Une tête int8 (A·(5+C), S, S) et ce que le noyau reçoit pour elle."""

    data: np.ndarray
    anchors_q8: np.ndarray  # (A, 2)
    classes: int
    softmax: bool
    scale: float
    exp_frac: int
    sigmoid: np.ndarray
    exp: np.ndarray
    softmax_exp: np.ndarray

    @property
    def grid(self):
        return self.data.shape[-1]

    @property
    def stride_log2(self):
        stride = ANCHOR_REF // self.grid
        if stride * self.grid != ANCHOR_REF or stride & (stride - 1):
            raise ValueError("pas de grille 416/S non puissance de 2")
        return stride.bit_length() - 1


def make_heads(outputs, net, luts, image=0):
    """Têtes de l'image `image` de `outputs` ({id: (N, C, S, S)}) ; `luts` : {id: HeadLuts}."""
    out = []
    for hid, mask in heads(net):
        lut = luts[hid]
        out.append(HwHead(np.asarray(outputs[hid][image]), anchors_q8(np.asarray(
            net["anchors"], dtype=np.float64)[mask]), net["classes"],
            net["layers"][hid]["type"] == "region", lut.scale, lut.exp_frac,
            np.asarray(lut.sigmoid, np.int64), np.asarray(lut.exp, np.int64),
            np.asarray(lut.softmax_exp, np.int64)))
    return out


def decode_head(h, conf, out):
    """Candidates (cellule, classe) de la tête `h`, ajoutées à `out` dans l'ordre du flux.

    Chaque candidate : (x1, y1, x2, y2, score Q16, classe), entiers Python.
    """
    a_n = len(h.anchors_q8)
    s, nc = h.grid, h.classes
    p = np.asarray(h.data, dtype=np.int64).reshape(a_n, 5 + nc, s, s)
    thr_o = logit_threshold_q(conf, h.scale)
    thr_s = conf_q16(conf)
    sl = h.stride_log2
    wshift = ANCHOR_FRAC + h.exp_frac - BOX_FRAC
    a, i, j = np.nonzero(p[:, 4] >= thr_o)  # ordre (ancre, ligne, colonne)
    if len(a) == 0:
        return
    t = p[a, :, i, j]  # (K, 5 + C)
    sig, ex, sm = h.sigmoid, h.exp, h.softmax_exp
    cx = rshift_round((sig[t[:, 0] + OFFSET] + (j.astype(np.int64) << 16)) << sl, 16 - BOX_FRAC)
    cy = rshift_round((sig[t[:, 1] + OFFSET] + (i.astype(np.int64) << 16)) << sl, 16 - BOX_FRAC)
    w = np.minimum(rshift_round(h.anchors_q8[a, 0] * ex[t[:, 2] + OFFSET], wshift), W_MAX)
    hh = np.minimum(rshift_round(h.anchors_q8[a, 1] * ex[t[:, 3] + OFFSET], wshift), W_MAX)
    x1, y1 = cx - (w >> 1), cy - (hh >> 1)
    x2, y2 = x1 + w, y1 + hh
    obj = sig[t[:, 4] + OFFSET]
    tc = t[:, 5:]
    if h.softmax:
        e = sm[tc - tc.max(axis=1, keepdims=True) + SOFTMAX_OFFSET]
        recip = (np.int64(1) << 32) // e.sum(axis=1)
        score = rshift_round(obj[:, None] * e * recip[:, None], 32)
    else:
        score = rshift_round(obj[:, None] * sig[tc + OFFSET], FRAC_BITS)
    for k, c in zip(*np.nonzero(score > thr_s)):  # ordre (cellule, classe)
        out.append((int(x1[k]), int(y1[k]), int(x2[k]), int(y2[k]), int(score[k, c]), int(c)))


def overlaps(b, x1, y1, x2, y2, p, q):
    """IoU(b, sélection) > p/q, vectorisé sur la sélection ; tout entier."""
    iw = np.maximum(0, np.minimum(b[2], x2) - np.maximum(b[0], x1))
    ih = np.maximum(0, np.minimum(b[3], y2) - np.maximum(b[1], y1))
    inter = iw * ih
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return (p + q) * inter > p * (area_b + (x2 - x1) * (y2 - y1))


def nms_nosort(cands, p, q, cap=CAP):
    """NMS sans tri sur des candidates dans l'ordre du flux → (boîtes retenues, débordement).

    Boîtes retenues dans l'ordre des emplacements : (x1, y1, x2, y2, score, classe).
    """
    sel = np.zeros((cap, 6), dtype=np.int64)
    alive = np.zeros(cap, dtype=bool)
    n = overflow = 0
    for b in cands:
        m = alive[:n] & (sel[:n, 5] == b[5])
        idx = np.nonzero(m)[0]
        if len(idx):
            s = sel[idx]
            idx = idx[overlaps(b, s[:, 0], s[:, 1], s[:, 2], s[:, 3], p, q)]
        if len(idx):
            if np.any(sel[idx, 4] >= b[4]):
                continue  # recouverte par une meilleure (ou égale, arrivée avant)
            sel[idx[0]] = b
            alive[idx[1:]] = False
        elif n < cap:
            sel[n] = b
            alive[n] = True
            n += 1
        else:
            overflow += 1
    return [tuple(int(v) for v in r) for r in sel[:n][alive[:n]]], overflow


def run(heads_, conf, iou, cap=CAP):
    """Têtes d'une image → (boîtes retenues, débordement), comme le noyau `yolo_post`."""
    cands = []
    for h in heads_:
        decode_head(h, conf, cands)
    p, q = iou_fraction(iou)
    return nms_nosort(cands, p, q, cap)


def to_detections(boxes):
    """Boîtes du noyau → (boxes (m, 4) (cx, cy, w, h) normalisées, scores, labels), triées
    par score décroissant (stable) comme `filter_and_nms`. Seule étape flottante (hôte)."""
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    r = np.asarray(boxes, dtype=np.int64)
    unit = float(ANCHOR_REF << BOX_FRAC)
    b = np.stack([(r[:, 0] + r[:, 2]) / 2, (r[:, 1] + r[:, 3]) / 2,
                  r[:, 2] - r[:, 0], r[:, 3] - r[:, 1]], axis=1) / unit
    s = r[:, 4] / float(1 << FRAC_BITS)
    order = np.argsort(-r[:, 4], kind="stable")
    return b[order], s[order], r[order, 5]


def postprocess_hw_counted(outputs, net, luts, conf_thr, iou_thr, cap=CAP, overflow=None):
    """Comme `postprocess_int` : têtes int8 → par image `(boxes, scores, labels)` ; les
    débordements de chaque image sont ajoutés à la liste `overflow`."""
    n = len(next(iter(outputs.values())))
    out = []
    for b in range(n):
        boxes, ov = run(make_heads(outputs, net, luts, b), conf_thr, iou_thr, cap)
        out.append(to_detections(boxes))
        if overflow is not None:
            overflow.append(ov)
    return out


def postprocess_hw(outputs, net, luts, conf_thr, iou_thr, cap=CAP):
    return postprocess_hw_counted(outputs, net, luts, conf_thr, iou_thr, cap)
