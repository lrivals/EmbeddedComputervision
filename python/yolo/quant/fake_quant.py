"""Fake-quant à estimateur straight-through et réseau d'entraînement quantifié (QAT) —
T9.3.2, T9.3.3, §9.2 « PTQ ou QAT ».

**Poids** : symétriques par canal de sortie, s_w,f = max|W_f| / wqmax, exactement comme
`quantize_weights_per_channel` (le modèle entier exporté a donc les poids vus en QAT). STE :
∂Ŵ/∂W = 1 (tous les poids sont dans la plage, le max définit l'échelle).

**Activations** : pas s = 2^{round(k)}, une **puissance de 2** dont l'exposant k est appris
(2024-yan : écrêtage appris contraint à une puissance de 2 [2024-yan#006.0]). La sortie
suit l'ordre du matériel (`int_layers.conv_int`) : r = round(x/s), leaky entière
r > 0 ? r : (13r + 64) ≫ 7, puis ŷ = clip(r, −qmax, qmax)·s. Avec v = a·x/s (a = 1, ou 13/128
pour la partie négative de la leaky) :

- ∂ŷ/∂x = a si |v| ≤ qmax, 0 sinon (STE, §9.2) ;
- ∂ŷ/∂s = r − v dans la plage, ±qmax hors plage (LSQ, hors base) ;
- ∂s/∂k = s·ln 2, l'arrondi de k étant traversé lui aussi par STE.

Option `lsq_scale` : gradient de k multiplié par g = 1/√(n·qmax) (n : éléments de la
carte), le facteur d'échelle de LSQ (hors base) ; désactivé pour l'entraînement
(`lowbit.qat_from_steps`).

`QATNetwork` : réseau **à BN fusionnée** (convs avec biais, `fuse_network`) dont chaque conv
fake-quantifie ses poids, applique la leaky 13/128 du matériel et fake-quantifie sa sortie ;
l'entrée est quantifiée à 1/127 comme par l'hôte. Les têtes gardent 8 bits au pas fixe des
LUT (1/8). Même interface que `Network` : le `Trainer` l'entraîne tel quel.
"""

import math

import numpy as np

from yolo.layers.conv import conv_backward, conv_forward
from yolo.models.graph import Network
from yolo.quant.int_layers import LEAKY_MUL, LEAKY_SHIFT
from yolo.quant.quantize import INPUT_SCALE, QMAX, weight_scales

LN2 = math.log(2.0)
LEAKY_SLOPE = LEAKY_MUL / 2**LEAKY_SHIFT  # 13/128
STEP_PARAM = "log2_s"


def qmax_of_bits(bits):
    return (1 << (bits - 1)) - 1


def pow2_step(log2_step):
    """Puissance de 2 la plus proche : 2^{⌊k + ½⌋}."""
    return 2.0 ** math.floor(float(np.asarray(log2_step).reshape(-1)[0]) + 0.5)


# ---------------------------------------------------------------------------- poids
def fq_weight(W, qmax):
    """Poids fake-quantifiés par canal (même arrondi ⌊v + ½⌋ que `quantize`)."""
    sw = weight_scales(W, qmax).astype(W.dtype)[:, None, None, None]
    return (np.clip(np.floor(W / sw + 0.5), -qmax, qmax) * sw).astype(W.dtype)


# ----------------------------------------------------------------------- activations
def fq_act_forward(x, log2_step, qmax, leaky=False):
    """Sortie d'une conv telle que le matériel la produit, à l'échelle s = 2^{round(k)} :
    r = round(x/s), puis leaky entière (r > 0 ? r : (13r + 64) ≫ 7) si `leaky`, puis
    clip(·, −qmax, qmax)·s (même ordre que `int_layers.conv_int`). Rend (ŷ, cache)."""
    s = pow2_step(log2_step)
    u = x / x.dtype.type(s)
    r = np.floor(u + 0.5)
    a = 1.0
    if leaky:
        neg = r <= 0
        r = np.where(neg, np.floor((LEAKY_MUL * r + 64) / 128), r)
        a = np.where(neg, LEAKY_SLOPE, 1.0)
    v = a * u  # substitut continu de r
    inside = np.abs(v) <= qmax
    y = (np.clip(r, -qmax, qmax) * s).astype(x.dtype)
    return y, (v, r, a, inside, s, qmax)


def fq_act_backward(dy, cache, grad_scale=1.0):
    """Rend (δx, δk) : STE pour x (pente a de la leaky dans la plage, 0 hors plage), LSQ
    pour s, puis ∂s/∂k = s ln 2 ; δk × `grad_scale`."""
    v, r, a, inside, s, qmax = cache
    dx = np.where(inside, dy * a, 0).astype(dy.dtype)
    ds_local = np.where(inside, r - v, np.sign(v) * qmax)
    dk = float(np.sum(dy.astype(np.float64) * ds_local)) * s * LN2 * grad_scale
    return dx, dk


# ----------------------------------------------------------------------------- réseau
class QATNetwork(Network):
    """Réseau à BN fusionnée, poids et activations fake-quantifiés (voir la docstring).

    `w_qmax`, `a_qmax` : {id conv: qmax} ; une conv absente garde 127. `log2_steps` :
    {id conv: k initial} des sorties à pas appris ; `fixed_steps` : {id conv: s} des sorties
    à pas fixe (têtes, 1/8).
    """

    def __init__(self, fused, w_qmax, a_qmax, log2_steps, fixed_steps, input_scale=INPUT_SCALE,
                 lsq_scale=True):
        super().__init__(fused.net, dtype=fused.dtype, rng=0)
        for i, p in enumerate(fused.params):
            self.params[i] = {k: np.array(v, dtype=self.dtype) for k, v in p.items()}
        self.w_qmax = dict(w_qmax)
        self.a_qmax = dict(a_qmax)
        self.fixed_steps = dict(fixed_steps)
        self.input_scale = input_scale
        self.lsq_scale = lsq_scale
        for i, k in log2_steps.items():
            self.params[i][STEP_PARAM] = np.array([k], dtype=self.dtype)

    def act_step(self, i):
        """Pas de la sortie de la conv i (puissance de 2 si appris)."""
        if STEP_PARAM in self.params[i]:
            return pow2_step(self.params[i][STEP_PARAM])
        return self.fixed_steps[i]

    def act_scales(self):
        return {i: self.act_step(i) for i, layer in enumerate(self.layers)
                if layer["type"] == "conv"}

    def forward(self, x, train=True, all_outputs=False):
        x = np.asarray(x)
        q = np.clip(np.floor(x / x.dtype.type(self.input_scale) + 0.5), -QMAX, QMAX)
        return super().forward((q * self.input_scale).astype(x.dtype), train, all_outputs)

    def _conv_forward(self, i, x, train):
        layer, p = self.layers[i], self.params[i]
        W = fq_weight(p["W"], self.w_qmax.get(i, QMAX))
        y, c_conv = conv_forward(x, W, p["b"], s=layer["s"])
        qmax = self.a_qmax.get(i, QMAX)
        k = p[STEP_PARAM] if STEP_PARAM in p else math.log2(self.fixed_steps[i])
        y, c_q = fq_act_forward(y, k, qmax, leaky=layer["act"] == "leaky")
        return y, [c_conv, c_q]

    def _conv_backward(self, i, dy, caches):
        caches = list(caches)
        c_q = caches.pop()
        g = 1.0 / math.sqrt(dy[0].size * c_q[5]) if self.lsq_scale else 1.0
        dy, dk = fq_act_backward(dy, c_q, g)
        grads = {}
        if STEP_PARAM in self.params[i]:
            grads[STEP_PARAM] = np.array([dk], dtype=self.dtype)
        dx, gc = conv_backward(dy, caches.pop())  # δŴ, transmis à W (STE)
        grads.update(gc)
        return [dx], grads


def pow2_act_steps(values_of, convs, qmax_of):
    """PTQ : pour chaque conv, le pas 2^j qui minimise l'erreur quadratique du fake-quant
    sur les valeurs de calibration `values_of(i)` ; rend {id: j}."""
    out = {}
    for i in convs:
        v = np.asarray(values_of(i), dtype=np.float64)
        qmax = qmax_of(i)
        top = max(float(np.abs(v).max()), 1e-8)
        j0 = math.floor(math.log2(top / qmax))
        best = None
        for j in range(j0 - 6, j0 + 2):
            s = 2.0**j
            err = float(np.mean((v - np.clip(np.floor(v / s + 0.5), -qmax, qmax) * s) ** 2))
            if best is None or err < best[0]:
                best = (err, j)
        out[i] = best[1]
    return out
