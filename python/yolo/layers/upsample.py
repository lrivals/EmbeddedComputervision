"""Upsample ×s au plus proche voisin — §4.5."""


def upsample_forward(x, s=2):
    # §4.5 : y[si+u, sj+v] = x[i, j]
    return x.repeat(s, axis=2).repeat(s, axis=3), s


def upsample_backward(dy, cache):
    # §4.5 : δx[i, j] = Σ_{u,v} δy[si+u, sj+v]
    s = cache
    n, c, h, w = dy.shape
    return dy.reshape(n, c, h // s, s, w // s, s).sum(axis=(3, 5)), {}
