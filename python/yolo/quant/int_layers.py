"""Couches en arithmétique entière, telles que le matériel les exécute — §9.3, §10.3.

Contrat (repris par le golden C++ et le HLS, conventions.md « Arithmétique entière ») :

- accumulateur : acc_f = Σ q_w q_x + q_b,f, exact sur 32 bits signés ;
- requantification : y = (acc · M0_f + 2ⁿ⁻¹) >> n, produit sur 64 bits, décalage
  **arithmétique** (plancher) : arrondi au plus proche, demis vers +∞ (−2,5 → −2) ;
- leaky 0,1 ≈ 13/128 : y ≤ 0 → (y · 13 + 2⁶) >> 7, **arrondi** comme la requantification,
  appliqué après elle. Le §9.3 écrit (y · 13) >> 7 (plancher) : ce plancher biaise chaque
  sortie négative de −½ pas en moyenne, biais qui se cumule de couche en couche (mesuré :
  écart moyen aux têtes ×1,7) ; l'arrondi ne coûte qu'une addition ;
- saturation finale dans [−qmax, qmax], qmax = 127 par défaut, plus petit par couche pour
  les activations à b bits (qmax = 2^{b−1} − 1, T9.3 : le conteneur reste int8) ;
- maxpool (stride 1 : réplication du bord), upsample et route : identiques au flottant, car
  le max et la copie sont exacts sur des entiers.

Tout est calculé en `np.int64`. Moteur `"f64"` de la convolution : même produit sur des
entiers rangés en float64 (BLAS). Il est exact tant que |acc| < 2⁵³ (ici < 2³¹), quel que
soit l'ordre des additions, et donne bit à bit le résultat du moteur `"int64"`.
"""

import numpy as np

from yolo.layers.conv import conv_forward
from yolo.layers.pool import maxpool_forward
from yolo.layers.route import route_forward
from yolo.layers.upsample import upsample_forward

QMIN, QMAX = -127, 127
ACC_LIMIT = 1 << 31
LEAKY_MUL, LEAKY_SHIFT = 13, 7
ENGINES = ("int64", "f64")


def rshift_round(x, n):
    """§9.3 : (x + 2ⁿ⁻¹) >> n, décalage arithmétique sur int64."""
    x = np.asarray(x, dtype=np.int64)
    return (x + (np.int64(1) << np.int64(n - 1))) >> np.int64(n)


def leaky_int(y):
    """§9.3 : y si y > 0, sinon (13 y + 2⁶) >> 7 (arrondi, demis vers +∞)."""
    y = np.asarray(y, dtype=np.int64)
    return np.where(y > 0, y, rshift_round(y * LEAKY_MUL, LEAKY_SHIFT))


def clip_q(y, qmax=QMAX):
    return np.clip(y, -qmax, qmax)


def clip8(y):
    return clip_q(y)


def conv_acc(qx, qW, qb, engine="int64"):
    """Accumulateur int32 (rangé en int64) : Σ q_w q_x + q_b ; erreur s'il déborde."""
    if engine == "int64":
        acc, _ = conv_forward(np.asarray(qx, np.int64), np.asarray(qW, np.int64))
    elif engine == "f64":
        acc, _ = conv_forward(np.asarray(qx, np.float64), np.asarray(qW, np.float64))
        acc = acc.astype(np.int64)
    else:
        raise ValueError(f"moteur inconnu : {engine!r}")
    acc = acc + np.asarray(qb, np.int64)[None, :, None, None]
    if np.abs(acc).max(initial=0) >= ACC_LIMIT:
        raise OverflowError("accumulateur hors de l'int32")
    return acc


def split4(qx):
    """x = 16·x_hi + x_lo, x_hi = x ≫ 4 ∈ [−8, 7] (signé 4 bits), x_lo = x & 15 ∈ [0, 15]
    (non signé 4 bits) ; exact en complément à 2 (T9.3.4)."""
    qx = np.asarray(qx, dtype=np.int64)
    return qx >> 4, qx & 15


def conv_acc_split4(qx, qW, qb, engine="int64"):
    """Première couche d'un réseau 4 bits [2024-yan#007.3] : l'entrée 8 bits passe par deux
    convolutions à entrées 4 bits, acc = 16·conv(x_hi) + conv(x_lo) + q_b ; égal à
    `conv_acc` à l'entier près (linéarité)."""
    hi, lo = split4(qx)
    zero = np.zeros(len(qW), dtype=np.int64)
    return 16 * conv_acc(hi, qW, zero, engine) + conv_acc(lo, qW, qb, engine)


def requantize(acc, M0, n):
    """§9.3 : (acc · M0_f + 2ⁿ⁻¹) >> n ; |acc·M0| < 2³¹·2³¹ = 2⁶² tient sur int64."""
    return rshift_round(acc * np.asarray(M0, np.int64)[None, :, None, None], n)


def conv_int(qx, qW, qb, M0, n, act="leaky", engine="int64", qmax=QMAX):
    """Couche conv entière complète (§9.3) ; renvoie (y int64 dans [−qmax, qmax], acc)."""
    acc = conv_acc(qx, qW, qb, engine)
    y = requantize(acc, M0, n)
    if act == "leaky":
        y = leaky_int(y)
    return clip_q(y, qmax), acc


def maxpool_int(qx, k=2, s=2):
    return maxpool_forward(np.asarray(qx, np.int64), k=k, s=s)[0]


def upsample_int(qx, s=2):
    return upsample_forward(np.asarray(qx, np.int64), s=s)[0]


def route_int(qxs):
    return route_forward([np.asarray(q, np.int64) for q in qxs])[0]
