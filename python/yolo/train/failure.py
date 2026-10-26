"""Joint training of the detector and the failure head (T39.2).

The head of `yolo.infer.failure.FailureHead` (target "miss") reads the pooled multi-scale
features of the detector (`feature_layers`, mean and max per channel) and is trained with
the detector. Its loss, per batch of N images with failure labels y and logits f:

    L_fail = (1/N) Σ w_i BCE(σ(f_i), y_i)              w_i = `pos_weight` if y_i else 1
           + λ · max(0, c₀ − (1 − (1/N) Σ σ(f_i)))²      (SelectiveNet coverage constraint)

The asymmetric weight makes a missed person cost more than a false alarm; the coverage term
(Geifman 2019) keeps the share of accepted images (no alarm) above c₀, so the head cannot
reach a low loss by raising the alarm everywhere.

**Labels move while the detector trains**: they are recomputed at every batch from the
current outputs of the detector (decode, threshold `conf`, NMS, `misses.matched` on each
ground-truth person), detached from the graph. Option: `freeze_after` stops the detector's
updates from that iteration on (only the head trains; the BN running statistics still move
with the train-mode forward pass), so the last labels are fixed.

The gradient of L_fail reaches the detector through the pooled features, scaled by
`weight` (0: the head trains on the moving features, the detector ignores it).
"""

import numpy as np

from yolo.infer.failure import FailureHead, _sigmoid, _softplus, feature_layers
from yolo.train.optim import AdamW


def pool_forward(m):
    """(N, C, H, W) → (N, 2C) mean then max, with the cache of `pool_backward`."""
    n, c, h, w = m.shape
    flat = m.reshape(n, c, h * w)
    arg = flat.argmax(axis=2)
    pooled = np.concatenate([flat.mean(axis=2), np.take_along_axis(flat, arg[..., None],
                                                                    axis=2)[..., 0]], axis=1)
    return pooled, (m.shape, arg)


def pool_backward(dp, cache):
    shape, arg = cache
    n, c, h, w = shape
    d = np.repeat((dp[:, :c] / (h * w))[..., None], h * w, axis=2)
    np.put_along_axis(d, arg[..., None],
                      np.take_along_axis(d, arg[..., None], axis=2) + dp[:, c:, None], axis=2)
    return d.reshape(shape)


def miss_labels(outputs, net, gt_boxes, gt_labels, cls, conf, iou_thr):
    """1 per image with at least one ground truth of class `cls` missed at `conf`."""
    from yolo.explain.misses import float_outputs, matched

    hids = [h for h, _ in heads_of(net)]
    y = np.zeros(len(gt_boxes))
    for n, (b, lab) in enumerate(zip(gt_boxes, gt_labels)):
        persons = np.asarray(b, np.float64).reshape(-1, 4)[np.asarray(lab) == cls]
        if not len(persons):
            continue
        out = float_outputs({h: np.asarray(outputs[h][n:n + 1], np.float64) for h in hids},
                            net, conf, iou_thr)
        y[n] = float(any(not matched(out.dets, p, cls) for p in persons))
    return y


def heads_of(net):
    from yolo.data.targets import heads

    return heads(net)


class FailureTerm:
    def __init__(self, net, hidden=32, pos_weight=4.0, coverage=0.9, lam=1.0, weight=1.0,
                 cls=0, conf=0.25, iou_thr=0.45, lr=1e-3, freeze_after=None, seed=0):
        self.net = net  # spec dict
        self.layers = feature_layers(net)
        self.hidden, self.seed = hidden, seed
        self.pos_weight, self.coverage, self.lam, self.weight = pos_weight, coverage, lam, weight
        self.cls, self.conf, self.iou_thr, self.lr = cls, conf, iou_thr, lr
        self.freeze_after = freeze_after
        self.head = None
        self.opt = None
        self.last = {}

    def frozen(self, it):
        """Detector frozen at iteration `it` (only the head trains)."""
        return self.freeze_after is not None and it >= self.freeze_after

    def _init(self, x):
        self.head = FailureHead(x.shape[1], self.hidden, "miss", self.seed).standardize(x)
        self.opt = AdamW(self.head.params, weight_decay=1e-4)

    def loss(self, f, y):
        """(L_fail, ∂L/∂f) for logits f and labels y."""
        n = len(f)
        w = np.where(y > 0.5, self.pos_weight, 1.0)
        bce = float((w * (_softplus(f) - y * f)).sum() / n)
        df = w * (_sigmoid(f) - y) / n
        a = _sigmoid(f)
        gap = self.coverage - (1.0 - a.mean())
        pen = self.lam * max(gap, 0.0) ** 2
        if gap > 0:
            df = df + 2 * self.lam * gap * a * (1 - a) / n
        return bce + pen, df, {"bce": bce, "coverage": float(1.0 - a.mean()), "penalty": pen}

    def __call__(self, outs, y):
        """`outs`: all layer outputs (list, NumPy); y: labels (N,). Returns (loss, {layer:
        δ of its output} scaled by `weight`, head gradients)."""
        pooled, caches = zip(*(pool_forward(np.asarray(outs[i], np.float64))
                               for i in self.layers))
        x = np.concatenate(pooled, axis=1)
        if self.head is None:
            self._init(x)
        f, cache = self.head.forward(x)
        loss, df, parts = self.loss(f, y)
        grads, dx = self.head.backward(cache, df)
        douts, off = {}, 0
        for i, p, c in zip(self.layers, pooled, caches):
            d = dx[:, off:off + p.shape[1]]
            off += p.shape[1]
            if self.weight:
                douts[i] = self.weight * pool_backward(d, c)
        self.last = {"loss": loss, "positives": float(y.mean()), **parts}
        return loss, douts, grads

    def step(self, grads):
        self.opt.step(grads, self.lr)

    # checkpoint: head, standardization and optimizer moments
    def arrays(self):
        if self.head is None:
            return {}
        out = {f"failure/{k}": v for k, v in self.head.arrays().items()}
        out.update({f"failure/opt/{k}": v for k, v in self.opt.state_dict().items()})
        return out

    def load_arrays(self, a):
        a = {k[len("failure/"):]: v for k, v in a.items() if k.startswith("failure/")}
        if not a:
            return
        self.head = FailureHead.from_arrays(a)
        self.opt = AdamW(self.head.params, weight_decay=1e-4)
        opt = {k[len("opt/"):]: v for k, v in a.items() if k.startswith("opt/")}
        if opt:
            self.opt.load_state_dict(opt)
