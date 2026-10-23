"""Linear CKA between representations (T36.8, Kornblith 2019).

    CKA(X, Y) = ‖Yᵀ X‖²_F / (‖Xᵀ X‖_F · ‖Yᵀ Y‖_F),   X, Y centred over the examples

X (n, p) and Y (n, q) are the representations of the same n examples (rows): here the
pooled maps of a layer over a probe set (`pool`: mean over the pixels, or the maps reduced
to `grid` × `grid` cells, then flattened). Invariant to orthogonal maps and to isotropic
scaling, not to any invertible linear map.
"""

import numpy as np


def pool(acts, grid=1):
    """(N, C, h, w) → (N, C·grid²): mean over `grid` × `grid` blocks of the map."""
    a = np.asarray(acts, dtype=np.float64)
    n, c, h, w = a.shape
    if grid == 1:
        return a.mean(axis=(2, 3))
    hs, ws = np.array_split(np.arange(h), grid), np.array_split(np.arange(w), grid)
    out = np.stack([a[:, :, r][:, :, :, s].mean(axis=(2, 3)) for r in hs for s in ws], -1)
    return out.reshape(n, -1)


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    # n × n Gram form: cheaper when the features outnumber the examples
    kx, ky = x @ x.T, y @ y.T
    hsic = (kx * ky).sum()
    den = np.sqrt((kx * kx).sum() * (ky * ky).sum())
    return float(hsic / den) if den > 0 else 0.0


def cka_matrix(reps_a, reps_b):
    """CKA of every pair: `reps_a` [X_i], `reps_b` [Y_j] → (len a, len b)."""
    return np.array([[linear_cka(x, y) for y in reps_b] for x in reps_a])
