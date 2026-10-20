"""Formal proof of the scalar kernels of the output stage (T28.2).

    python tools/prove_kernels.py                     # proof, all shifts, all exports
    python tools/prove_kernels.py --mutant rounding   # wrong rounding → counter-example
    python tools/prove_kernels.py --mutant leaky32    # 13·y on 32 bits → counter-example

**What is proved**, for every shift n ∈ [1, 31] (one SMT query per n and per obligation):
the C++ golden (`cpp/golden/include/golden/conv.hpp`, reused as is by the HLS `store_tile`),
re-expressed in z3 bit-vectors with the C++ widths, equals the Python specification
(`python/yolo/quant/int_layers.py`) written as mathematics (floor division by its defining
inequalities, unbounded integers), **for all** int32 accumulators, all M0 with 0 < M0 < 2ⁿ
(precondition, multiplier < 1, checked by `Model::load`) and all qmax ∈ [1, 127]:

- (a) `int64(acc)·M0 + 2ⁿ⁻¹` does not overflow int64;
- (b) the shifted value fits int32 (the cast `int32_t(… >> n)` loses nothing);
- (c) the result y satisfies 2ⁿ·y ≤ acc·M0 + 2ⁿ⁻¹ < 2ⁿ·(y + 1), i.e. y = ⌊(acc·M0 + 2ⁿ⁻¹)/2ⁿ⌋;
- (d) `leaky_int(y)` = y if y > 0, else ⌊(13y + 64)/128⌋, for every int32 y;
- (e) `clip_q(v, qmax)` = min(max(v, −qmax), qmax) and fits int8.

The composition (d)∘(c) then (e) is the output stage of `conv_out` and `store_tile`.

**Per layer, no int32 overflow of the accumulator** (T4.3, proved from the exported int8
weights): the accumulator is linear in the input, so its extrema over the box of int8 inputs
are at a vertex, computed exactly in integers by `bounds.acc_range` (reached, not estimated).
Every partial sum of the tiled loops (any subset of the products, without the bias, which
`store_tile` adds last) lies between the sum of the negative and the sum of the positive
per-product extrema, so the check covers the HLS accumulator `acc_t` too. A z3 query checks the
vertex argument itself on a small random layer.

**Residual risk**: the z3 model is written by hand from the C++ source; a gap between the two
is caught by the exhaustive and boundary tests of T28.1 (python/tests/test_exhaustive_ops.py,
cpp/golden/tests/test_ops_exhaustive.cpp), which run the real code.

z3 is an optional dependency (`pip install -e "python[formal]"`).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import z3  # noqa: E402

from yolo.quant import bounds  # noqa: E402

W = 72  # wide enough for every exact intermediate (|acc·M0| < 2^62)
NETS = ("tiny-yolov2-voc", "tiny-yolov3-coco", "tiny-yolov4-coco", "tiny-yolov7-coco")


def sx(v, w=W):
    """Sign-extend a bit-vector to w bits (exact value as a w-bit integer)."""
    return z3.SignExt(w - v.size(), v)


def cpp_requant(acc, m0, n, mutant=None):
    """`golden::requantize`: int64 product and rounding, arithmetic shift, int32 cast."""
    p = z3.SignExt(32, acc) * z3.SignExt(32, m0)
    r = p if mutant == "rounding" else p + z3.BitVecVal(1 << (n - 1), 64)
    return z3.Extract(31, 0, r >> n)  # >> on z3 bit-vectors is arithmetic


def cpp_leaky(y, mutant=None):
    """`golden::leaky_int`: y > 0 ? y : int32((int64(13)·y + 64) >> 7)."""
    if mutant == "leaky32":  # the version before T28.1: 13·y in int32
        neg = (y * 13 + 64) >> 7
    else:
        neg = z3.Extract(31, 0, (z3.SignExt(32, y) * 13 + 64) >> 7)
    return z3.If(y > 0, y, neg)


def cpp_clip(v, qmax):
    """`golden::clip_q`: int8(v < −qmax ? −qmax : v > qmax ? qmax : v)."""
    c = z3.If(v < -qmax, -qmax, z3.If(v > qmax, qmax, v))
    return z3.Extract(7, 0, c), c


def _check(solver, goal_violation, timeout_ms):
    solver.set("timeout", timeout_ms)
    solver.add(goal_violation)
    t0 = time.time()
    res = solver.check()
    return res, solver.model() if res == z3.sat else None, time.time() - t0


def prove_requant(n, mutant=None, timeout_ms=600_000):
    """Obligations (a), (b), (c) for one shift n. Returns [(name, result, model, seconds)]."""
    acc, m0 = z3.BitVecs("acc m0", 32)
    pre = z3.And(m0 > 0, z3.ULT(m0, z3.BitVecVal(1 << n, 32)))  # 0 < M0 < 2^n
    exact = sx(acc) * sx(m0) + z3.BitVecVal(1 << (n - 1), W)  # Python: unbounded integers
    y = cpp_requant(acc, m0, n, mutant)
    out = []
    lo64, hi64 = z3.BitVecVal(-(1 << 63), W), z3.BitVecVal((1 << 63) - 1, W)
    s = z3.Solver()
    s.add(pre)
    out.append(("(a) no int64 overflow",) + _check(s, z3.Or(exact < lo64, exact > hi64),
                                                    timeout_ms))
    s = z3.Solver()
    s.add(pre)
    shifted = (exact >> n) if mutant != "rounding" else ((sx(acc) * sx(m0)) >> n)
    lo32, hi32 = z3.BitVecVal(-(1 << 31), W), z3.BitVecVal((1 << 31) - 1, W)
    out.append(("(b) result fits int32",) + _check(s, z3.Or(shifted < lo32, shifted > hi32),
                                                   timeout_ms))
    s = z3.Solver()
    s.add(pre)
    Y, two_n = sx(y), z3.BitVecVal(1 << n, W)
    floor_ok = z3.And(two_n * Y <= exact, exact < two_n * (Y + 1))
    out.append(("(c) y = floor((acc·M0 + 2^(n-1)) / 2^n)",) + _check(s, z3.Not(floor_ok),
                                                                      timeout_ms))
    return out


def prove_leaky_clip(mutant=None, timeout_ms=600_000):
    y = z3.BitVec("y", 32)
    v = cpp_leaky(y, mutant)
    e = 13 * sx(y) + 64
    floor_ok = z3.If(y > 0, sx(v) == sx(y), z3.And(128 * sx(v) <= e, e < 128 * (sx(v) + 1)))
    out = [("(d) leaky = floor((13y + 64) / 128)",) + _check(z3.Solver(), z3.Not(floor_ok),
                                                             timeout_ms)]
    q = z3.BitVec("qmax", 32)
    s = z3.Solver()
    s.add(q >= 1, q <= 127)
    b8, c = cpp_clip(y, q)
    spec = z3.If(y < -q, -q, z3.If(y > q, q, y))
    bad = z3.Or(c != spec, sx(b8) != sx(c))  # equals the spec and survives the int8 cast
    out.append(("(e) clip to [-qmax, qmax], fits int8",) + _check(s, bad, timeout_ms))
    return out


def prove_vertex(seed=0, n_w=12):
    """The maximum of Σ w·x over x ∈ [−128, 127]^n is the vertex value of `acc_range`."""
    import numpy as np

    rng = np.random.default_rng(seed)
    w = rng.integers(-127, 128, (1, n_w))
    lo, hi = bounds.acc_range(w.reshape(1, n_w, 1, 1), np.zeros(1, np.int64))
    xs = [z3.BitVec(f"x{j}", 16) for j in range(n_w)]
    s = z3.Solver()
    s.add(*[z3.And(x >= -128, x <= 127) for x in xs])
    acc = sum(int(w[0, j]) * z3.SignExt(16, x) for j, x in enumerate(xs))
    s.add(z3.Or(acc > int(hi[0]), acc < int(lo[0])))
    return s.check() == z3.unsat


def layer_overflow(nets=NETS):
    """Per layer: exact accumulator extrema, partial sums, margin to int32."""
    from yolo.io.export import load_model

    rows = []
    for net in nets:
        d = ROOT / "model" / net
        if not (d / "manifest.json").exists():
            continue
        qm, _ = load_model(d)
        for i, qc in sorted(qm.convs.items()):
            lo, hi = bounds.acc_range(qc.qW, qc.qb)
            plo, phi = bounds.acc_range(qc.qW, 0 * qc.qb)  # partial sums, no bias
            worst = max(abs(int(lo.min())), abs(int(hi.max())), abs(int(plo.min())),
                        abs(int(phi.max())))
            rows.append(dict(net=net, layer=i, lo=int(lo.min()), hi=int(hi.max()), worst=worst,
                             bits=worst.bit_length() + 1, ok=worst < (1 << 31),
                             m0_ok=bool((qc.M0 > 0).all() and (qc.M0 < (1 << qc.n)).all())))
    return rows


def describe(model, *names):
    return ", ".join(f"{k} = {model[z3.BitVec(k, 32)].as_signed_long()}" for k in names
                     if model[z3.BitVec(k, 32)] is not None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mutant", choices=("rounding", "leaky32"), default=None)
    ap.add_argument("--shifts", default="1-31", help="range of n, e.g. 1-31 or 24")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per query")
    ap.add_argument("--markdown", action="store_true", help="results table in Markdown")
    args = ap.parse_args()
    a, _, b = args.shifts.partition("-")
    shifts = range(int(a), int(b or a) + 1)
    failed, rows = 0, []
    t_all = time.time()
    if args.mutant != "rounding":
        for name, res, model, dt in prove_leaky_clip(args.mutant, args.timeout * 1000):
            rows.append(("—", name, res, dt))
            if res != z3.unsat:
                failed += 1
                if model is not None:
                    print(f"COUNTER-EXAMPLE {name}: {describe(model, 'y', 'qmax')}")
    if args.mutant != "leaky32":
        for n in shifts:
            for name, res, model, dt in prove_requant(n, args.mutant, args.timeout * 1000):
                rows.append((n, name, res, dt))
                if res != z3.unsat:
                    failed += 1
                    if model is not None:
                        print(f"COUNTER-EXAMPLE n = {n}, {name}: {describe(model, 'acc', 'm0')}")
                        if args.mutant:
                            break
            if failed and args.mutant:
                break
    vertex = prove_vertex() if not args.mutant else True
    layers = layer_overflow() if not args.mutant else []
    bad_layers = [r for r in layers if not (r["ok"] and r["m0_ok"])]
    total = time.time() - t_all
    if args.markdown:
        print("| n | obligation | result | s |\n|---|---|---|---|")
        for n, name, res, dt in rows:
            print(f"| {n} | {name} | {'proved' if res == z3.unsat else res} | {dt:.2f} |")
        print("\n| Network | conv layers | widest accumulator (bits, sign incl.) | int32 overflow |"
              " M0 < 2^n |\n|---|---|---|---|---|")
        for net in sorted({r["net"] for r in layers}, key=NETS.index):
            rs = [r for r in layers if r["net"] == net]
            top = max(rs, key=lambda r: r["worst"])
            print(f"| {net} | {len(rs)} | {top['bits']} (L{top['layer']:02d}, "
                  f"±{top['worst']:,}) | {'none' if all(r['ok'] for r in rs) else 'YES'} | "
                  f"{'all' if all(r['m0_ok'] for r in rs) else 'NO'} |")
    proved = sum(1 for r in rows if r[2] == z3.unsat)
    print(f"{proved}/{len(rows)} obligations proved, vertex lemma "
          f"{'proved' if vertex else 'FAILED'}, {len(layers)} layers checked "
          f"({len(bad_layers)} failing), {total:.1f} s")
    if args.mutant:
        # A mutant must be caught: success means a counter-example was found.
        sys.exit(0 if failed else 1)
    sys.exit(0 if not failed and vertex and not bad_layers else 1)


if __name__ == "__main__":
    main()
