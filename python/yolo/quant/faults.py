"""T25.0: fault model and injector for the bit-exact integer model (`IntNetwork`).

The fault-free `IntNetwork` is the reference: any difference in the output is caused by the
injected fault.

- **Sites**: `weight` (int8 qW), `bias` (int32 q_b), `act` (int8 output of a conv, after
  saturation and before a maxpool), `acc` (int32 accumulator, bias included, before
  requantization).
- **Kinds**: `flip` (one bit of one element: `index`, `bit`, both random when None), `ber`
  (every bit flips independently with probability `rate`), `stuck0` / `stuck1` (one bit forced
  to 0 / 1, on one element, or on a `rate` fraction of the elements).
- `layer` = conv id; None = every conv (a single `flip` then picks its conv in proportion to
  its size, a `ber` hits every conv).
- Bits are those of the two's complement word of the site (8 or 32 bits). A flip of the sign
  bit of an int8 can give −128, outside the [−127, 127] of the fault-free contract: it is
  what the stored bits mean, it is kept.
- `weight` and `bias` faults are **persistent** (`inject_params`, a corrupted copy of the
  `QuantModel`); `act` and `acc` faults are **transient** (`FaultInjector`, an `IntNetwork`
  hook drawn again for each image).
- A faulty network wraps its int32 accumulator modulo 2³² (`IntNetwork.wrap_acc`) as the
  hardware register does, where the fault-free model raises an error.
- Seeded: `inject_params(qm, faults, seed)` and `FaultInjector.start(key)` draw from
  `default_rng((seed, key))`, so a run does not depend on the order of the images.
- BER: the number of flipped bits is drawn from Binomial(n_bits, rate), then their positions
  uniformly (with replacement: two draws on the same bit cancel, as two flips would).
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

WIDTH = {"weight": 8, "bias": 32, "act": 8, "acc": 32}
PERSISTENT = ("weight", "bias")
KINDS = ("flip", "ber", "stuck0", "stuck1")


@dataclass(frozen=True)
class Fault:
    site: str
    kind: str = "ber"
    layer: int = None
    bit: int = None
    rate: float = 0.0
    index: int = None

    def __post_init__(self):
        if self.site not in WIDTH:
            raise ValueError(f"unknown fault site {self.site!r}")
        if self.kind not in KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if self.bit is not None and not 0 <= self.bit < WIDTH[self.site]:
            raise ValueError(f"bit {self.bit} outside the {WIDTH[self.site]}-bit word")
        if self.kind == "ber" and not 0 <= self.rate <= 1:
            raise ValueError("BER outside [0, 1]")

    @property
    def width(self):
        return WIDTH[self.site]

    @property
    def single(self):
        """One element (flip, or stuck-at without a rate)."""
        return self.kind == "flip" or (self.kind != "ber" and self.rate == 0)


def to_unsigned(v, width):
    return np.asarray(v, np.int64) & ((1 << width) - 1)


def to_signed(u, width):
    half = 1 << (width - 1)
    return (np.asarray(u, np.int64) ^ half) - half


def flip_bits(values, idx, bits, width):
    """Copy of `values` (int64) with bit `bits[k]` of element `idx[k]` (flat) flipped."""
    u = to_unsigned(values, width).ravel()
    np.bitwise_xor.at(u, np.asarray(idx, np.int64), np.int64(1) << np.asarray(bits, np.int64))
    return to_signed(u, width).reshape(np.shape(values))


def force_bits(values, idx, bits, one, width):
    """Copy of `values` (int64) with bit `bits[k]` of element `idx[k]` forced to `one`."""
    u = to_unsigned(values, width).ravel()
    idx = np.asarray(idx, np.int64)
    m = np.int64(1) << np.asarray(bits, np.int64)
    u[idx] = (u[idx] | m) if one else (u[idx] & ~m)
    return to_signed(u, width).reshape(np.shape(values))


def draw(rng, fault, size):
    """(idx, bits) of a fault on an array of `size` elements (None: nothing hit)."""
    w = fault.width
    if fault.single:
        idx = rng.integers(size) if fault.index is None else fault.index
        bit = rng.integers(w) if fault.bit is None else fault.bit
        return np.array([idx]), np.array([bit])
    if fault.kind == "ber":
        k = rng.binomial(size * w, fault.rate)
        if k == 0:
            return None
        pos = rng.integers(size * w, size=k)
        return pos // w, pos % w
    k = rng.binomial(size, fault.rate)
    if k == 0:
        return None
    bits = rng.integers(w, size=k) if fault.bit is None else np.full(k, fault.bit)
    return rng.choice(size, size=k, replace=False), bits  # distinct stuck elements


def hit(values, fault, idx, bits):
    if fault.kind in ("flip", "ber"):
        return flip_bits(values, idx, bits, fault.width)
    return force_bits(values, idx, bits, fault.kind == "stuck1", fault.width)


def _pick_layer(rng, fault, sizes):
    """Conv hit by a single-element fault: `fault.layer`, or one drawn ∝ size."""
    if fault.layer is not None:
        return fault.layer
    ids = sorted(sizes)
    p = np.array([sizes[i] for i in ids], np.float64)
    return ids[rng.choice(len(ids), p=p / p.sum())]


def inject_params(qm, faults, seed=0):
    """Copy of `qm` with the persistent faults (`weight`, `bias`) applied; only the convs hit
    are copied, `qm` itself and the ABFT checksums (`cs_w`, `cs_b`) are unchanged.
    Returns (QuantModel, number of bits hit)."""
    from yolo.quant.int_model import QuantModel

    rng = np.random.default_rng((seed, 0))
    convs, n = dict(qm.convs), 0
    for f in faults:
        if f.site not in PERSISTENT:
            continue
        attr = "qW" if f.site == "weight" else "qb"
        sizes = {i: getattr(c, attr).size for i, c in convs.items()}
        layers = [_pick_layer(rng, f, sizes)] if f.single or f.layer is not None \
            else sorted(sizes)
        for i in layers:
            c = convs[i]
            arr = getattr(c, attr)
            d = draw(rng, f, arr.size)
            if d is None:
                continue
            new = hit(arr, f, *d).astype(arr.dtype)
            convs[i] = dataclasses.replace(c, **{attr: new})
            n += len(d[0])
    return QuantModel(net=qm.net, input_scale=qm.input_scale, convs=convs), n


def conv_sizes(qm, chw):
    """{conv id: output elements per image} for an input (C, H, W), from a zero image."""
    from yolo.quant.int_model import IntNetwork

    class _Rec:
        sizes = {}

        def on_acc(self, i, acc, x):
            self.sizes[i] = acc[0].size
            return acc

        def on_out(self, i, y):
            return y

    inet, rec = IntNetwork(qm), _Rec()
    inet.hooks.append(rec)
    inet.forward(np.zeros((1, *chw), np.int8))
    return dict(rec.sizes)


class FaultInjector:
    """`IntNetwork` hook for the transient faults (`act`, `acc`).

    `start(key)` before each image (key: its index); `sizes` ({conv: output elements per
    image}, `conv_sizes`) is needed only by a single-element fault without `layer`.
    `flips[site]`: bits hit since `start`, `total[site]` since the creation.
    """

    def __init__(self, faults, seed=0, sizes=None):
        self.faults = [f for f in faults if f.site not in PERSISTENT]
        self.seed, self.sizes = seed, sizes
        self.total = {s: 0 for s in WIDTH}
        self.start(0)

    def start(self, key):
        self.rng = np.random.default_rng((self.seed, 1, key))
        self.flips = {s: 0 for s in WIDTH}
        self.target = {}
        for k, f in enumerate(self.faults):
            if f.single and f.layer is None:
                if self.sizes is None:
                    raise ValueError("single fault without layer: pass `sizes` (conv_sizes)")
                self.target[k] = _pick_layer(self.rng, f, self.sizes)

    def _apply(self, site, i, v):
        for k, f in enumerate(self.faults):
            if f.site != site or self.target.get(k, f.layer) not in (None, i):
                continue
            d = draw(self.rng, f, v.size)
            if d is not None:
                v = hit(v, f, *d)
                self.flips[site] += len(d[0])
                self.total[site] += len(d[0])
        return v

    def on_acc(self, i, acc, x):
        return self._apply("acc", i, acc)

    def on_out(self, i, y):
        return self._apply("act", i, y)


def apply(inet, faults, seed=0, sizes=None):
    """(IntNetwork with the persistent faults, FaultInjector hooked for the transient ones).

    The hooks already on `inet` (an `AbftMonitor`) follow the injector, so they see the
    faulty accumulator."""
    from yolo.quant.int_model import IntNetwork

    qm, _ = inject_params(inet.qm, faults, seed)
    out = IntNetwork(qm, inet.engine)
    out.wrap_acc = True  # a 32-bit accumulator wraps (faulty bias, sum of faulty products)
    inj = FaultInjector(faults, seed, sizes)
    out.hooks = [inj, *inet.hooks]
    return out, inj


# ------------------------------------------------------- T25.7: faults in the HLS buffers
# A flip in an on-chip buffer of `yolo_conv` (hls/kernels/fault_hook.hpp) is not a fault of a
# T25.0 site: the buffer holds one tile, reloaded from DDR for the next tile, so the fault only
# reaches the outputs that this tile owns. `hls_buffer_reference` maps the buffer address to
# tensor indices and recomputes those outputs; hls/tb/tb_fault.cpp --dump gives the C-sim
# outputs it is compared with (python/tests/test_faults.py).
#
# Buffers (indices a, b, c, d of the CSV of tb_fault):
#   in   in_buf[lane][row][col] of input-channel tile `ti`: lane = ti·Tn + a is the channel
#        (or (channel, kernel row) of a folded conv), row/col relative to the tile window;
#   w    w_buf[o][lane][i][j]: weight (to + a, channel of lane, i, j);
#   out  out_buf[o][r][c]: int32 accumulator **without the bias** (added by the output stage);
#   q    requantized int8 tile read by the fused max-pool;
#   cs   ABFT checksum lane: no output effect.
HLS_BUFFERS = ("in", "w", "out", "q", "cs")


def _flip8(v, bit):
    u = (int(v) & 0xFF) ^ (1 << bit)
    return u - 256 if u >= 128 else u


def _flip32(v, bit):
    from yolo.quant.int_layers import wrap32

    return wrap32((int(v) & 0xFFFFFFFF) ^ (1 << bit))


def hls_lane(lane, k, fold):
    """(channel, kernel row) of an in_buf / w_buf lane (`load_input`, `reorder_weights`)."""
    return (lane // k, lane % k) if fold else (lane, None)


def hls_buffer_reference(x, qc, g, pool, tn, fold):
    """Owned outputs of the faulty tile: (to, prow0, pcol0, int8 array (tm_n, np_r, np_c)).

    `x`: int8 input (C, H, W) of the conv; `qc`: its `QConv`; `g`: one tb_fault CSV row
    (dict of ints: buf name, ti, a, b, c, d, bit, row, col, prow0, pcol0, to, tr_n, tc_n, np_r,
    np_c, tm_n); `pool`: (pk, ps) of the fused max-pool, (1, 1) without, (1, 2) for a stride-2
    conv; `tn`: Tn of the kernel; `fold`: the conv is folded (cin·k ≤ Tn, first conv)."""
    from yolo.quant import int_layers as il

    k = qc.qW.shape[-1]
    pad = (k - 1) // 2
    x = np.asarray(x, np.int64)
    W = np.asarray(qc.qW, np.int64)
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    to, tm, row, col, trn, tcn = (g[n] for n in ("to", "tm_n", "row", "col", "tr_n", "tc_n"))
    Wt = W[to:to + tm]

    def window(ch, dr, dc):  # x_pad[ch, row + dr + trr, col + dc + tcc] over the tile
        return xp[ch, row + dr:row + dr + trn, col + dc:col + dc + tcn]

    # Accumulators of the tile without the bias (out_buf).
    acc = np.zeros((tm, trn, tcn), np.int64)
    for i in range(k):
        for j in range(k):
            acc += np.einsum("oc,crs->ors", Wt[:, :, i, j], xp[:, row + i:row + i + trn,
                                                               col + j:col + j + tcn])
    buf, a, b, c, d, bit = g["buf"], g["a"], g["b"], g["c"], g["d"], g["bit"]
    if buf == "out":
        acc[a, b, c] = _flip32(acc[a, b, c], bit)
    elif buf == "w":
        ch, r = hls_lane(g["ti"] * tn + b, k, fold)
        i = r if fold else c
        o = to + a
        dw = _flip8(W[o, ch, i, d], bit) - W[o, ch, i, d]
        acc[a] += dw * window(ch, i, d)
    elif buf == "in":
        ch, ki = hls_lane(g["ti"] * tn + a, k, fold)
        ir, ic = row - pad + b + (ki or 0), col - pad + c
        inside = 0 <= ir < x.shape[1] and 0 <= ic < x.shape[2] and ch < x.shape[0]
        v = int(x[ch, ir, ic]) if inside else 0
        dv = _flip8(v, bit) - v
        # out[trr, tcc] reads in_buf[trr + i][tcc + j] (folded: kernel row fixed by the lane)
        for i in ([0] if fold else range(k)):
            for j in range(k):
                trr, tcc = b - i, c - j
                if 0 <= trr < trn and 0 <= tcc < tcn:
                    acc[:, trr, tcc] += Wt[:, ch, ki if fold else i, j] * dv
    acc = il.wrap32(acc + np.asarray(qc.qb, np.int64)[to:to + tm, None, None])
    q = il.conv_out(acc[None], np.asarray(qc.M0)[to:to + tm], qc.n, qc.act, qc.qmax)[0]
    if buf == "q":
        q[a, b, c] = _flip8(q[a, b, c], bit)
    # Fused max-pool of the owned outputs, window clamped to the map (edge replication).
    pk, ps = pool
    lo = (pk - 1) // 2 if ps == 1 else 0
    R, C = x.shape[1] + 2 * pad - k + 1, x.shape[2] + 2 * pad - k + 1
    out = np.full((tm, g["np_r"], g["np_c"]), -128, np.int64)
    for pr in range(g["np_r"]):
        for pc in range(g["np_c"]):
            for u in range(pk):
                for v in range(pk):
                    r = min(max((g["prow0"] + pr) * ps - lo + u, 0), R - 1) - row
                    s = min(max((g["pcol0"] + pc) * ps - lo + v, 0), C - 1) - col
                    out[:, pr, pc] = np.maximum(out[:, pr, pc], q[:, r, s])
    return to, g["prow0"], g["pcol0"], out.astype(np.int8)


# ------------------------------------------------------------------ failure head (T25.9)
HEAD_SITES = {0: ("weight", "bias", "act", "acc"), 1: ("weight", "bias", "acc")}


def head_sizes(ih):
    """{(site, layer): elements} of the `IntFailureHead` `ih` (T39.5): layer 0 is the hidden
    layer (d → h, as a 1×1 conv), layer 1 the output (h → 1, whose int32 accumulator is the
    score: no `act` site)."""
    h, d = ih.qW1.shape
    return {("weight", 0): h * d, ("bias", 0): h, ("act", 0): h, ("acc", 0): h,
            ("weight", 1): h, ("bias", 1): 1, ("acc", 1): 1}


def _wrap32(v):
    return to_signed(to_unsigned(v, 32), 32)


def _transient(rng, fault, v):
    """`v` (N, m) with `fault` drawn independently for each of the N images (rows)."""
    n, m = v.shape
    if fault.single:  # one element per image
        idx = (rng.integers(m, size=n) if fault.index is None else np.full(n, fault.index))
        bits = (rng.integers(fault.width, size=n) if fault.bit is None
                else np.full(n, fault.bit))
        return hit(v, fault, np.arange(n) * m + idx, bits)
    d = draw(rng, fault, n * m)  # BER over all rows = an independent BER per row
    return v if d is None else hit(v, fault, *d)


def head_forward(ih, p, faults=(), seed=0):
    """Scores (N,) of the `IntFailureHead` `ih` on the integer features `p` (N, d) under the
    `faults` (`Fault.layer`: 0 hidden, 1 output; None draws nothing), and (N,) whether the
    ABFT check of either layer (T25.2: Σ_i acc_i against the checksum row Σ_i W_i of the
    fault-free weights, plus Σ_i b_i) saw an error. `weight` / `bias` faults are drawn once
    (persistent), `act` / `acc` faults for each image (transient), from (seed, 0) and
    (seed, 1). The accumulators wrap modulo 2³² as the hardware register."""
    from yolo.quant.int_layers import requantize

    rng_p, rng_t = np.random.default_rng((seed, 0)), np.random.default_rng((seed, 1))
    qW1, qb1 = ih.qW1.astype(np.int64), ih.qb1.astype(np.int64)
    qW2, qb2 = ih.qW2.astype(np.int64), np.array([ih.qb2], np.int64)
    by = {}
    for f in faults:
        if f.layer not in HEAD_SITES or f.site not in HEAD_SITES[f.layer]:
            raise ValueError(f"no {f.site} site on layer {f.layer} of the failure head")
        by.setdefault((f.site, f.layer), []).append(f)

    def persistent(v, key):
        for f in by.get(key, ()):
            d = draw(rng_p, f, v.size)
            if d is not None:
                v = hit(v, f, *d)
        return v

    def transient(v, key):
        for f in by.get(key, ()):
            v = _transient(rng_t, f, v)
        return v

    fW1, fb1 = persistent(qW1, ("weight", 0)), persistent(qb1, ("bias", 0))
    fW2, fb2 = persistent(qW2, ("weight", 1)), persistent(qb2, ("bias", 1))
    x = np.asarray(p, np.int64) - ih.zp
    acc1 = transient(_wrap32(x @ fW1.T + fb1), ("acc", 0))
    det = (acc1.sum(1) != x @ qW1.sum(0) + qb1.sum())
    hid = np.clip(requantize(acc1[:, :, None, None], ih.M0, ih.n)[:, :, 0, 0], 0, 127)
    hid = transient(hid, ("act", 0))
    acc2 = transient(_wrap32(hid @ fW2 + fb2)[:, None], ("acc", 1))[:, 0]
    det |= acc2 != hid @ qW2 + qb2[0]
    return ih.sign * acc2, det
