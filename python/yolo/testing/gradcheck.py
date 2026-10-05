"""Vérification des gradients par différences finies centrées (§11).

Usage type : on définit une perte `f()` qui relit le tenseur `x` (modifié en place), on
calcule le gradient analytique, puis `check_grad(f, x, analytic)` renvoie l'erreur relative
maximale.

`f` peut renvoyer un scalaire ou le tableau des termes de la perte (L = somme du tableau).
Dans le second cas, la différence f(x+h) − f(x−h) est prise terme à terme avant la somme :
les termes non affectés par x[i] s'annulent exactement, ce qui évite la perte de précision
d'une différence de deux grandes sommes.
"""

import numpy as np

from yolo.backend import get_xp, to_numpy


def numerical_grad(f, x, h=1e-6, idx=None):
    """Gradient de `L = Σ f()` par rapport à `x`, par différences centrées (§11).

    `x` est modifié en place puis restauré. `idx` : liste d'indices (tuples) à évaluer ;
    `None` évalue tous les éléments. Renvoie un tableau de la forme de `x` (nul hors `idx`).
    """
    grad = np.zeros(x.shape, dtype=np.float64)
    indices = np.ndindex(x.shape) if idx is None else idx
    for i in indices:
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        grad[i] = np.sum(to_numpy(fp) - to_numpy(fm)) / (2 * h)
    return grad


def rel_error(a, n, eps=1e-12):
    """|a − n| / max(|a|, |n|, ε) avec |·| = norme max sur le tenseur (§11).

    Norme du tenseur plutôt qu'élément par élément : avec h = 1e-6, le bruit d'arrondi du
    gradient numérique est de l'ordre de 1e-10 en absolu, ce qui rendrait arbitrairement
    grande l'erreur relative d'un élément dont le gradient est accidentellement petit. Une
    erreur réelle sur un seul élément reste visible (≈ son poids relatif dans la norme).
    """
    a = to_numpy(a).astype(np.float64)
    n = to_numpy(n).astype(np.float64)
    if a.size == 0:
        return 0.0
    scale = max(np.max(np.abs(a)), np.max(np.abs(n)), eps)
    return float(np.max(np.abs(a - n)) / scale)


def sample_indices(shape, n_samples, rng=None):
    """`n_samples` indices tirés uniformément (avec remise) dans un tenseur de forme `shape`."""
    rng = np.random.default_rng(rng)
    flat = rng.integers(0, int(np.prod(shape)), size=n_samples)
    return [tuple(int(v) for v in np.unravel_index(k, shape)) for k in flat]


def check_grad(f, x, analytic, h=1e-6, n_samples=None, rng=None, eps=1e-12):
    """Erreur relative maximale entre `analytic` et le gradient numérique de `f` en `x`.

    `n_samples` : ne vérifie qu'un échantillon d'indices (gros tenseurs).
    """
    if n_samples is None or n_samples >= x.size:
        idx = list(np.ndindex(x.shape))
    else:
        idx = sample_indices(x.shape, n_samples, rng)
    if not idx:
        return 0.0
    num = numerical_grad(f, x, h=h, idx=idx)
    rows = tuple(np.array(c) for c in zip(*idx))
    return rel_error(to_numpy(analytic)[rows], num[rows], eps=eps)


def check_layer(fwd, bwd, x, params, n_samples=None, rng=0):
    """Gradcheck d'une couche `fwd(x, params) -> (y, cache)`, `bwd(dy, cache) -> (dx, grads)`.

    Perte L = Σ y·R avec R aléatoire fixé, donc δy = R. `params` : dict de tableaux, modifiés
    en place pendant la mesure. Renvoie {"x": err, nom: err, ...}.
    """
    rng = np.random.default_rng(rng)
    y, cache = fwd(x, params)
    r = get_xp(y).asarray(rng.standard_normal(y.shape))  # sur le backend de y (T12.11)
    dx, grads = bwd(r, cache)

    def loss():
        return fwd(x, params)[0] * r

    errors = {"x": check_grad(loss, x, dx, n_samples=n_samples, rng=rng)}
    for name, p in params.items():
        errors[name] = check_grad(loss, p, grads[name], n_samples=n_samples, rng=rng)
    return errors
