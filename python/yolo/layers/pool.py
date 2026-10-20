"""Max pooling k×k, stride 2 ou 1 — §4.4.

Stride 1 (couche 11) : complétion de k − 1 lignes/colonnes par réplication du bord, ce qui
conserve la taille et donne le même maximum qu'un −∞. Comme Darknet (décalage −(k − 1)/2,
division entière), ⌊(k − 1)/2⌋ en haut et à gauche, le reste en bas et à droite : rien en haut
pour k = 2 (couche 11), une fenêtre centrée pour k impair (SPP de YOLOv7-tiny, T26.0).
"""

from yolo.backend import add_at, get_xp, sliding_window_view


def maxpool_forward(x, k=2, s=2):
    """Renvoie (y, cache) ; le cache mémorise l'argmax de chaque fenêtre (§4.4)."""
    xnp = get_xp(x)
    n, c, h, w = x.shape
    lo, hi = pool_pads(k, s)
    xp = xnp.pad(x, ((0, 0), (0, 0), (lo, hi), (lo, hi)), mode="edge") if lo + hi else x
    win = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]  # (N,C,Ho,Wo,k,k)
    ho, wo = win.shape[2:4]
    flat = win.reshape(n, c, ho, wo, k * k)
    # À égalité, argmax prend la première position : l'élément d'origine avant sa réplique.
    arg = flat.argmax(axis=-1)
    y = xnp.take_along_axis(flat, arg[..., None], axis=-1)[..., 0]
    return y, (arg, x.shape, xp.shape, k, s)


def pool_pads(k, s):
    """(haut/gauche, bas/droite) de la complétion par réplication ; aucune en stride 2."""
    if s != 1:
        return 0, 0
    lo = (k - 1) // 2
    return lo, k - 1 - lo


def maxpool_backward(dy, cache):
    """§4.4 : δx[argmax] += δy[i, j], 0 ailleurs. Renvoie (dx, {})."""
    arg, x_shape, xp_shape, k, s = cache
    xnp = get_xp(dy)
    n, c, ho, wo = dy.shape
    u, v = xnp.divmod(arg, k)
    rows = xnp.arange(ho)[None, None, :, None] * s + u
    cols = xnp.arange(wo)[None, None, None, :] * s + v
    nn = xnp.arange(n)[:, None, None, None]
    cc = xnp.arange(c)[None, :, None, None]
    dxp = xnp.zeros(xp_shape, dtype=dy.dtype)
    # Fenêtres disjointes en stride 2, chevauchantes en stride 1 : add.at cumule les doublons.
    add_at(dxp, (nn, cc, rows, cols), dy)
    h, w = x_shape[2:]
    lo, hi = pool_pads(k, s)
    if lo + hi:
        # La complétion réplique le bord : son gradient revient au bord.
        dxp[:, :, lo + h - 1, :] += dxp[:, :, lo + h:, :].sum(axis=2)
        dxp[:, :, :, lo + w - 1] += dxp[:, :, :, lo + w:].sum(axis=3)
        dxp[:, :, lo, :] += dxp[:, :, :lo, :].sum(axis=2)
        dxp[:, :, :, lo] += dxp[:, :, :, :lo].sum(axis=3)
        dxp = dxp[:, :, lo:lo + h, lo:lo + w]
    return dxp, {}
