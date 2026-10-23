"""Gradient attributions of a detection score (T36.1).

Every method takes a folded `Network` (`targets.explain_net`), one preprocessed image
`x` (1, C, H, W) and a `Target`, and returns a dict: `map` (H, W) float64, `passes` (number
of forward + backward passes, the cost), and method-specific entries.

- **saliency**: |δs/δx|, max over the channels (Simonyan 2014). Known to fail the sanity
  checks; kept as the baseline.
- **SmoothGrad** (Smilkov 2017): mean of δs/δx over `n` copies of x with Gaussian noise of
  standard deviation σ·(max x − min x), then |·| max over the channels.
- **Integrated Gradients** (Sundararajan 2017): (x − x')·∫₀¹ δs/δx(x' + α(x − x')) dα by
  the midpoint rule over `steps` points; baseline x' black (zeros) or blurred x. The map is
  the sum over the channels. Completeness: Σ attributions = s(x) − s(x') up to the
  quadrature error.
- **Grad-CAM** (Selvaraju 2017) on the output A of a layer: w_c = mean over the pixels of
  δs/δA_c, map = ReLU(Σ_c w_c A_c), resized bilinearly to the input. Default layer: the
  input of the head conv of the target's head (L13 for Tiny-YOLOv2, L14 / L21 for v3, L28 /
  L35 for v4).

`randomize_from_top` gives the cascading weight randomization of the sanity checks of
Adebayo 2018.
"""

import copy

import numpy as np

from yolo.backend import get_xp, to_numpy
from yolo.explain.maps import box_blur, resize
from yolo.explain.targets import explain, target_score

CHUNK = 8  # images per forward pass of SmoothGrad and IG


def _grads(net, xs, t, chunk=CHUNK):
    """(scores (n,), δs/δx (n, C, H, W)) for a stack of inputs, in chunks."""
    xp = get_xp(xs)
    scores, grads = [], []
    for k in range(0, len(xs), chunk):
        e = explain(net, xs[k:k + chunk], t)
        scores.append(e.score)
        grads.append(e.dx)
    return xp.concatenate(scores), xp.concatenate(grads)


def saliency(net, x, t):
    e = explain(net, x, t)
    return {"map": to_numpy(get_xp(e.dx).abs(e.dx[0]).max(axis=0)).astype(np.float64),
            "passes": 2, "score": float(e.score[0])}


def smoothgrad(net, x, t, n=25, sigma=0.15, rng=0):
    xp = get_xp(x)
    rng = np.random.default_rng(rng)
    span = float(x.max() - x.min()) or 1.0
    noise = rng.normal(0, sigma * span, (n, *x.shape[1:])).astype(x.dtype)
    _, g = _grads(net, x + xp.asarray(noise), t)
    return {"map": to_numpy(xp.abs(g.mean(axis=0)).max(axis=0)).astype(np.float64),
            "passes": 2 * n}


def baseline_of(x, kind):
    """Black (zeros) or blurred (box blur of radius H/16) baseline of `x`."""
    if kind == "black":
        return np.zeros_like(x)
    if kind == "blur":
        return box_blur(to_numpy(x), max(1, x.shape[-1] // 16)).astype(x.dtype)
    raise ValueError(f"baseline: {kind!r}")


def integrated_gradients(net, x, t, steps=64, baseline="black"):
    xp = get_xp(x)
    base = xp.asarray(baseline_of(x, baseline) if isinstance(baseline, str) else baseline)
    alphas = (np.arange(steps) + 0.5) / steps
    path = xp.concatenate([base + a * (x - base) for a in alphas])
    _, g = _grads(net, path, t)
    attr = (x - base)[0] * g.mean(axis=0)
    s_x = float(target_score(_heads(net, x), net.net, t)[0])
    s_b = float(target_score(_heads(net, base), net.net, t)[0])
    total = float(attr.sum())
    attr = to_numpy(attr).astype(np.float64)
    return {"map": attr.sum(axis=0), "attr": attr, "passes": 2 * steps + 2,
            "score": s_x, "score_baseline": s_b, "sum": total,
            "completeness": abs(total - (s_x - s_b)) / max(abs(s_x - s_b), 1e-12)}


def _heads(net, x):
    return net.forward(x, train=False)


def default_layer(net, t):
    """Input of the head conv of the target's head (the `yolo` / `region` layer reads it)."""
    head_conv = net.inputs[t.head][0]
    return net.inputs[head_conv][0]


def gradcam(net, x, t, layer=None):
    layer = default_layer(net, t) if layer is None else layer
    e = explain(net, x, t, keep=[layer])
    a = to_numpy(e.acts[layer][0]).astype(np.float64)
    g = e.dlayers.get(layer)
    g = np.zeros_like(a) if g is None else to_numpy(g[0]).astype(np.float64)
    w = g.mean(axis=(1, 2))
    cam = np.maximum(np.tensordot(w, a, axes=1), 0)
    return {"map": resize(cam, *x.shape[-2:]), "cam": cam, "weights": w, "layer": layer,
            "passes": 2, "score": float(e.score[0])}


METHODS = {
    "saliency": saliency,
    "smoothgrad": smoothgrad,
    "ig_black": lambda net, x, t: integrated_gradients(net, x, t, baseline="black"),
    "ig_blur": lambda net, x, t: integrated_gradients(net, x, t, baseline="blur"),
    "gradcam": gradcam,
}


def conv_ids(net):
    return [i for i, layer in enumerate(net.layers) if layer["type"] == "conv"]


def randomize_from_top(net, k, rng=0):
    """Copy of `net` with the weights of its last `k` convs redrawn (He init, bias 0): the
    cascading randomization of Adebayo 2018, top layer first."""
    rng = np.random.default_rng(rng)
    out = copy.deepcopy(net)
    for i in conv_ids(net)[::-1][:k]:
        W = out.params[i]["W"]
        f, c, kh, kw = W.shape
        new = rng.normal(0, np.sqrt(2.0 / (c * kh * kw)), W.shape).astype(W.dtype)
        out.params[i]["W"] = get_xp(W).asarray(new)
        if "b" in out.params[i]:
            out.params[i]["b"] = out.params[i]["b"] * 0
    out._caches = None
    return out
