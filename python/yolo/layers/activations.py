"""Leaky ReLU (§4.3) et sigmoïde stable (§4.5)."""

from yolo.backend import get_xp


def leaky_forward(x, a=0.1):
    # §4.3 : φ(x) = max(x, a·x)
    pos = x > 0
    return get_xp(x).where(pos, x, a * x), (pos, a)


def leaky_backward(dy, cache):
    # §4.3 : δx = δy · (1 si x > 0, a sinon)
    pos, a = cache
    return get_xp(dy).where(pos, dy, a * dy), {}


def sigmoid(t):
    """§4.5 : 1/(1+e^{-t}) si t ≥ 0, e^{t}/(1+e^{t}) sinon ; sans overflow."""
    xnp = get_xp(t)
    e = xnp.exp(-xnp.abs(t))
    return xnp.where(t >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid_forward(t):
    y = sigmoid(t)
    return y, y


def sigmoid_backward(dy, cache):
    # §4.5 : σ'(t) = σ(t)(1 − σ(t))
    y = cache
    return dy * y * (1.0 - y), {}
