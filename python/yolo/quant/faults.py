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
