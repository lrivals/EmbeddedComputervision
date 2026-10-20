"""Exact accumulator bounds and operator sweeps of the output stage (T28.1, T28.2).

- `acc_range(qW, qb, xmin, xmax)`: per output channel, the smallest and largest int32
  accumulator `conv(x, qW) + q_b` over every input x in the box [xmin, xmax]^n. A conv is
  linear in x, so its extrema over a box are reached at a vertex: each weight takes the end of
  the box that maximizes (or minimizes) w·x. The bound is exact, not an estimate: it is reached
  by the input x_j = xmax if w_j > 0 else xmin (and the reverse for the minimum). This is the
  per-layer form of the worst case of T4.3.
- `sweep(lo, hi, M0, n, qmax, act)`: accumulator values on which the output stage
  (requantization, leaky, saturation) is compared across Python, golden C++ and HLS. When
  [lo, hi] holds at most `EXHAUSTIVE` values, all of them; otherwise every value next to a
  change of the requantized integer that can still change the int8 output, both ends, zero,
  and a dense uniform grid.

NumPy only (ADR 0001).
"""

import numpy as np

from yolo.quant.int_layers import ACC_LIMIT, LEAKY_MUL, LEAKY_SHIFT

X_MIN, X_MAX = -128, 127  # int8 inputs (activations are in [−qmax, qmax] ⊂ this box)
EXHAUSTIVE = 1 << 16
DENSE = 1024


def acc_range(qW, qb, xmin=X_MIN, xmax=X_MAX):
    """(lo, hi) int64 arrays (F,): exact extrema of conv(x, qW) + q_b over x ∈ [xmin, xmax]."""
    w = np.asarray(qW, np.int64).reshape(len(qW), -1)
    b = np.asarray(qb, np.int64)
    pos = np.where(w > 0, w, 0).sum(1)
    neg = np.where(w < 0, w, 0).sum(1)
    hi = pos * xmax + neg * xmin + b
    lo = pos * xmin + neg * xmax + b
    return lo, hi


def fits_int32(lo, hi):
    """True when every reachable accumulator fits a signed 32-bit register."""
    return bool(np.all(lo >= -ACC_LIMIT) and np.all(hi < ACC_LIMIT))


def requant_steps(M0, n, qmax, act):
    """Requantized integers k whose step can change the int8 output: k ∈ [kmin, kmax].

    Above qmax + 1 the output saturates at qmax; below, for the leaky, (13k + 64) >> 7
    saturates at −qmax once k ≤ −(qmax·128 + 64)/13; for a linear output, at −qmax − 1."""
    kmax = qmax + 1
    if act == "silu":  # table index: every step in [−128, 127] changes the index
        return -129, 128
    if act == "leaky":
        kmin = -((qmax << LEAKY_SHIFT) + 64) // LEAKY_MUL - 2
    else:
        kmin = -qmax - 1
    return kmin, kmax


def step_bounds(M0, n, kmin, kmax):
    """First accumulator of each requantized integer k ∈ [kmin, kmax]:
    requant(acc) = ⌊(acc·M0 + 2ⁿ⁻¹) / 2ⁿ⌋ ≥ k ⟺ acc ≥ ⌈(k·2ⁿ − 2ⁿ⁻¹) / M0⌉ (M0 > 0)."""
    k = np.arange(kmin, kmax + 1, dtype=object)
    num = k * (1 << n) - (1 << (n - 1))
    return np.array([-((-v) // int(M0)) for v in num], dtype=np.int64)


def sweep(lo, hi, M0, n, qmax=127, act="leaky", dense=DENSE, exhaustive=EXHAUSTIVE):
    """Sorted unique int64 accumulators in [lo, hi] for one channel (see module docstring)."""
    lo, hi = int(lo), int(hi)
    if hi - lo + 1 <= exhaustive:
        return np.arange(lo, hi + 1, dtype=np.int64)
    pts = [np.array([lo, hi, 0, -1, 1], np.int64),
           np.linspace(lo, hi, dense).round().astype(np.int64)]
    if M0 > 0:
        b = step_bounds(M0, n, *requant_steps(M0, n, qmax, act))
        pts += [b - 1, b, b + 1]
    v = np.unique(np.concatenate(pts))
    return v[(v >= lo) & (v <= hi)]


def layer_sweep(qc, xmin=X_MIN, xmax=X_MAX, **kw):
    """Blocks of one conv: list of (channel, M0, accumulators) over its reachable range.

    The accumulator here includes the bias (`conv_acc` + q_b), as in `conv_out`."""
    lo, hi = acc_range(qc.qW, qc.qb, xmin, xmax)
    act = "leaky" if qc.act == "leaky" else "linear"
    return [(f, int(qc.M0[f]), sweep(lo[f], hi[f], int(qc.M0[f]), qc.n, qc.qmax, act, **kw))
            for f in range(len(lo))]


# ------------------------------------------------------------------ operator file (T28.1)
# Shared by `golden_ops requant` (cpp/golden/tools/golden_ops.cpp) and `tb_ops` (hls/tb/):
# int32 little-endian. Header [n_blocks, B, L]; L = 64: the int8 SiLU table follows (256 bytes,
# T26.2), L = 0: none. Per block [M0, n, qmax, act (0 linear, 1 leaky, 2 SiLU), then B
# accumulators]. Output: n_blocks·B int8 in the same order.
ACT_CODE = {"linear": 0, "leaky": 1, "silu": 2}


def write_ops(path, blocks, B=169, lut=None):
    """`blocks`: (M0, n, qmax, act, acc array). Each array is cut into blocks of B values; the
    last block of each is padded with its first value. `lut`: the SiLU table of the "silu"
    blocks. Returns the flat (unpadded) mask."""
    rec, mask = [], []
    for M0, n, qmax, act, acc in blocks:
        acc = np.asarray(acc, np.int64)
        for s in range(0, len(acc), B):
            part = acc[s:s + B]
            pad = B - len(part)
            rec.append(np.array([M0, n, qmax, ACT_CODE[act]], np.int32))
            rec.append(np.concatenate([part, np.full(pad, part[0])]).astype(np.int32))
            mask.append(np.concatenate([np.ones(len(part), bool), np.zeros(pad, bool)]))
    head = np.array([len(mask), B, 0 if lut is None else 64], np.int32)
    table = [] if lut is None else [np.asarray(lut, np.int8).view("<i4").astype(np.int32)]
    np.concatenate([head, *table, *rec]).astype("<i4").tofile(path)
    return np.concatenate(mask) if mask else np.zeros(0, bool)
