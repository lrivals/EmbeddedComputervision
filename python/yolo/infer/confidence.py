"""Calibration of the detector's person score against failure prediction (T39.4).

- `ece`: expected calibration error of a score against the correctness of each detection
  (equal-width bins, weighted by their count, Guo 2017);
- `aurc`: area under the risk-coverage curve (Geifman 2018). Inputs are sorted by decreasing
  confidence; at coverage k/N, the risk is the share of failures among the k most confident.
  Lower is better;
- `fit_temperature` / `apply_temperature`: temperature scaling of a sigmoid score,
  s' = σ(logit(s) / T), with T fitted by NLL on a calibration split. For a detector the
  score is σ(t_o)·p_c; the temperature acts on its logit, as for a binary classifier
  (correct detection or not).

Temperature scaling is monotone: it cannot change a ranking, hence neither the AUROC nor the
AURC of a monitor that only reads the score. The degradation reported by Zhu 2024 comes from
training-time methods (label smoothing, `yolo_loss(label_smoothing=…)`), which change the
features and the ranking.
"""

import numpy as np

EPS = 1e-7


def ece(scores, correct, bins=15):
    s = np.clip(np.asarray(scores, np.float64), 0.0, 1.0)
    c = np.asarray(correct, np.float64)
    if not len(s):
        return float("nan")
    idx = np.minimum((s * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(s[m].mean() - c[m].mean())
    return float(total / len(s))


def aurc(confidence, failure):
    """`confidence`: high = trusted; `failure`: bool per item."""
    order = np.argsort(-np.asarray(confidence, np.float64), kind="mergesort")
    f = np.asarray(failure, np.float64)[order]
    if not len(f):
        return float("nan")
    return float((np.cumsum(f) / np.arange(1, len(f) + 1)).mean())


def logit(s):
    s = np.clip(np.asarray(s, np.float64), EPS, 1 - EPS)
    return np.log(s) - np.log1p(-s)


def apply_temperature(scores, t):
    return 1.0 / (1.0 + np.exp(-logit(scores) / t))


def nll(scores, labels):
    s = np.clip(np.asarray(scores, np.float64), EPS, 1 - EPS)
    y = np.asarray(labels, np.float64)
    return float(-(y * np.log(s) + (1 - y) * np.log1p(-s)).mean())


def fit_temperature(scores, labels, lo=-3.0, hi=3.0, steps=601):
    """T minimizing the NLL of `apply_temperature(scores, T)` against `labels`: grid on
    log T, then one finer grid around the best point."""
    for _ in range(2):
        grid = np.linspace(lo, hi, steps)
        losses = [nll(apply_temperature(scores, np.exp(g)), labels) for g in grid]
        k = int(np.argmin(losses))
        step = grid[1] - grid[0]
        lo, hi = grid[k] - step, grid[k] + step
    return float(np.exp(grid[k]))
