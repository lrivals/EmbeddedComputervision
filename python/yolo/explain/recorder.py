"""Training recorder (T36.7): how the network's choices form, without touching the trainer.

`Recorder.attach(trainer, loader)` wraps the trainer's `grad_hook` (chained with the one
already there, ADMM for instance) and returns the loader to pass to `fit`, together with
the `fit` callback. Every `every` iterations it stores, per conv layer:

- the norm of the weights, of their gradient, and the update / weight ratio ‖ΔW‖/‖W‖ of the
  step (the weights are copied in the hook, before the step, and compared after);
- on a fixed probe batch, forward in eval mode: the share of **dead** leaky units (channels
  negative at every pixel of every probe image, so stuck on the 0.1 slope), the activation
  percentiles 1 / 50 / 99, and the mean |γ| of the BN with the share |γ| < 0.01.

At every iteration, the loss of **each image** of the batch, Σ of its loss terms
(`LossResult.terms`), at its position in the epoch's order. The loader wrapper reads that
order from `DataLoader._order(epoch)`, the same draw as the batches.

From the per-image losses (`summary`): an image is *learned* in an epoch when its loss is
under τ, the median loss of the last epoch. Images never under τ are **never learned**. An
image that drops back over τ after being under it makes a **forgetting event** (Toneva 2019).
The mean and the standard deviation of its loss over the epochs are the two axes of the data
map (Swayamdipta 2020, with the loss in place of the probability of the true label).

`snap_every` also writes the weights (`snap_<it>.npz`, parameters and BN state) for the CKA
of T36.8.
"""

import time
from pathlib import Path

import numpy as np

from yolo.backend import to_numpy

PCTS = (1, 50, 99)


class IndexedLoader:
    """A `DataLoader` that also exposes the dataset indices of the batch just yielded."""

    def __init__(self, loader):
        self.loader = loader
        self.ids = None

    def __len__(self):
        return len(self.loader)

    def epoch(self, epoch, size_fn, first_iter=0, skip=0):
        order = self.loader._order(epoch)
        bs = self.loader.batch_size
        for b, batch in enumerate(self.loader.epoch(epoch, size_fn, first_iter, skip),
                                  start=skip):
            self.ids = order[b * bs:(b + 1) * bs]
            self.epoch_no = epoch
            yield batch

    def close(self):
        if hasattr(self.loader, "close"):
            self.loader.close()


class Recorder:
    def __init__(self, net, probe, n_images, every=10, out=None, snap_every=0):
        self.net = net
        self.probe = np.asarray(probe)
        self.every = every
        self.snap_every = snap_every
        self.out = Path(out) if out else None
        self.convs = [i for i, layer in enumerate(net.layers) if layer["type"] == "conv"]
        self.rows = {k: [] for k in ("it", "w_norm", "g_norm", "update", "dead", "pct",
                                     "gamma", "gamma_small")}
        self.losses = {}  # epoch -> (n_images,) float, NaN where not seen
        self.n_images = n_images
        self.seconds = 0.0
        self._before = None
        self._g = None
        self.loader = None

    # ------------------------------------------------------------------ wiring
    def attach(self, trainer, loader):
        """Chains the trainer's grad hook; returns (loader for `fit`, callback)."""
        inner = trainer.grad_hook

        def hook(tr, grads):
            if inner:
                inner(tr, grads)
            if (tr.it + 1) % self.every == 0:
                t0 = time.perf_counter()
                self._g = [float(np.sqrt(sum(float((to_numpy(v) ** 2).sum())
                                             for v in grads[i].values()))) for i in self.convs]
                self._before = [to_numpy(self.net.params[i]["W"]).copy() for i in self.convs]
                self.seconds += time.perf_counter() - t0

        trainer.grad_hook = hook
        self.loader = IndexedLoader(loader)
        return self.loader, self.callback

    def callback(self, tr, res, lr):
        t0 = time.perf_counter()
        ep = self.loader.epoch_no
        row = self.losses.setdefault(ep, np.full(self.n_images, np.nan))
        per = sum(res.terms[h].reshape(len(res.terms[h]), -1).sum(axis=1) for h in res.terms)
        row[np.asarray(self.loader.ids)[:len(per)]] = per
        if self._before is not None:
            self._layer_stats(tr.it)
            self._before = None
        if self.snap_every and self.out and tr.it % self.snap_every == 0:
            self.snapshot(tr.it)
        self.seconds += time.perf_counter() - t0

    # --------------------------------------------------------------- statistics
    def _layer_stats(self, it):
        w_norm, upd = [], []
        for i, before in zip(self.convs, self._before):
            w = to_numpy(self.net.params[i]["W"])
            n = float(np.linalg.norm(before))
            w_norm.append(n)
            upd.append(float(np.linalg.norm(w - before)) / n if n else 0.0)
        acts = self.net.forward(self._probe(), train=False, all_outputs=True)
        dead, pct, gam, small = [], [], [], []
        for i in self.convs:
            a = to_numpy(acts[i])
            if self.net.layers[i]["act"] == "leaky":
                dead.append(float((a.max(axis=(0, 2, 3)) <= 0).mean()))
            else:
                dead.append(0.0)
            pct.append(np.percentile(a, PCTS))
            g = self.net.params[i].get("gamma")
            g = np.abs(to_numpy(g)) if g is not None else np.ones(1)
            gam.append(float(g.mean()))
            small.append(float((g < 0.01).mean()))
        for k, v in (("it", it), ("w_norm", w_norm), ("g_norm", self._g), ("update", upd),
                     ("dead", dead), ("pct", pct), ("gamma", gam), ("gamma_small", small)):
            self.rows[k].append(v)

    def _probe(self):
        x = self.probe.astype(self.net.dtype)
        p = next((v for v in self.net.params[self.convs[0]].values()), None)
        if p is not None and not isinstance(p, np.ndarray):
            from yolo.backend import to_device

            x = to_device(x)
        return x

    def snapshot(self, it):
        self.out.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, (p, s) in enumerate(zip(self.net.params, self.net.state)):
            arrays.update({f"param/{i}/{k}": to_numpy(v) for k, v in p.items()})
            arrays.update({f"state/{i}/{k}": to_numpy(v) for k, v in s.items()})
        np.savez(self.out / f"snap_{it:06d}.npz", **arrays)

    # ----------------------------------------------------------------- results
    def loss_matrix(self):
        """(epochs, n_images) per-image loss, NaN where an image was not seen."""
        eps = sorted(self.losses)
        return np.stack([self.losses[e] for e in eps]) if eps else np.zeros((0, self.n_images))

    def arrays(self):
        out = {"convs": np.array(self.convs), "loss_img": self.loss_matrix(),
               "record_seconds": np.array(self.seconds)}
        for k, v in self.rows.items():
            out[k] = np.array(v, dtype=np.float64)
        return out

    def save(self, path=None):
        path = Path(path) if path else self.out / "record.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.arrays())
        return path


def summary(loss_img):
    """Never-learned images, forgetting events and the data map from (epochs, n) losses."""
    loss = np.asarray(loss_img, dtype=np.float64)
    last = loss[-1][~np.isnan(loss[-1])]
    tau = float(np.median(last)) if last.size else np.nan
    learned = loss <= tau  # NaN compares False: an unseen epoch counts as not learned
    seen = ~np.isnan(loss)
    never = ~(learned & seen).any(axis=0)
    forget = np.zeros(loss.shape[1], dtype=np.int64)
    prev = np.zeros(loss.shape[1], dtype=bool)
    for e in range(loss.shape[0]):
        drop = prev & seen[e] & ~learned[e]
        forget += drop
        prev = np.where(seen[e], learned[e], prev)
    return {"tau": tau, "never_learned": never, "forgetting": forget,
            "mean": np.nanmean(loss, axis=0), "std": np.nanstd(loss, axis=0)}


def corrupt_labels(samples, frac, classes, rng=0):
    """Copies of `samples` where a share `frac` of the images (drawn with `rng`) have every
    label replaced by another class drawn uniformly; returns (samples, corrupted indices)."""
    rng = np.random.default_rng(rng)
    n = len(samples)
    bad = np.sort(rng.choice(n, int(round(frac * n)), replace=False))
    out = list(samples)
    for k in bad:
        s = dict(out[k])
        lab = np.asarray(s["labels"]).copy()
        lab = (lab + rng.integers(1, classes, len(lab))) % classes
        s["labels"] = lab
        out[k] = s
    return out, bad
