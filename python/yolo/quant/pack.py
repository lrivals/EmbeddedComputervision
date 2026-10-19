"""Two int8 products per DSP48E2 multiplier — T23.0, §10.2, [2024-zhang#011.0].

The PE multiplies one activation `x` by the Tm weights of the output channels. Two weights
of consecutive filters are packed into the 27-bit A port, `x` goes to the 18-bit B port:

    (w_a·2^s + w_b)·x = (w_a·x)·2^s + w_b·x        (exact integer product, ≤ 45 bits)

Partial sums of N such products stay exact as long as the low lane Σ w_b·x fits in s
signed bits. Unpacking reads the low lane sign-extended; a negative low lane has borrowed
1 from the high lane, given back by `hi = (p >> s) + (lo < 0)` (Xilinx WP486, same
scheme). Only the multiplication changes: the int32 accumulator, the requantization and
the saturation of the contract (docs/conventions.md) stay.

Budget (DSP48E2: A 27 bits, B 18 bits, P 48 bits):

- s = 18: the largest shift for which every int8 pair fits in A (s = 19 overflows for
  w_a = −128, w_b < 0);
- N = 7 for any int8 operands (|w·x| ≤ 16384), N = 8 under the contract (|w|, |x| ≤ 127:
  −128 is never produced, conventions.md), so the Tn = 24 addition tree is cut into
  3 groups of 8 lanes, each unpacked before the int32 sum.

    python -m yolo.quant.pack    # budget table (results/dsp_packing.md)
"""

import numpy as np

A_BITS, B_BITS, P_BITS = 27, 18, 48  # DSP48E2: multiplier 27 × 18, accumulator 48
S = 18
QMAX = 127  # contract: −128 never produced (weights and activations)


def signed_fits(v, bits):
    """True where v is representable on `bits` signed bits."""
    v = np.asarray(v, dtype=np.int64)
    return (v >= -(1 << (bits - 1))) & (v < (1 << (bits - 1)))


def pack2(w_a, w_b, s=S):
    """A operand of the DSP: w_a·2^s + w_b (int64)."""
    return (np.asarray(w_a, dtype=np.int64) << s) + np.asarray(w_b, dtype=np.int64)


def unpack2(p, s=S):
    """(hi, lo) of a packed product or partial sum: lo sign-extended on s bits, hi with the
    sign borrow of the low lane given back."""
    p = np.asarray(p, dtype=np.int64)
    half = np.int64(1) << (s - 1)
    lo = ((p & ((np.int64(1) << s) - 1)) ^ half) - half
    hi = (p >> s) + (lo < 0)
    return hi, lo


def depth(s, pmin, pmax):
    """Largest N such that any sum of N products in [pmin, pmax] fits on s signed bits."""
    half = 1 << (s - 1)
    return min((half - 1) // pmax if pmax > 0 else 1 << 62,
               half // -pmin if pmin < 0 else 1 << 62)


def product_range(wmin, wmax, xmin, xmax):
    """(min, max) of w·x over the two intervals."""
    c = [wmin * xmin, wmin * xmax, wmax * xmin, wmax * xmax]
    return min(c), max(c)


def max_shift(k, wmin, wmax, a_bits=A_BITS):
    """Largest s for which k weights Σ w_i·2^{i·s} always fit in A (0 if none)."""
    best = 0
    for s in range(1, a_bits):
        hi = sum(wmax << (i * s) for i in range(k))
        lo = sum(wmin << (i * s) for i in range(k))
        if hi < 1 << (a_bits - 1) and lo >= -(1 << (a_bits - 1)):
            best = s
    return best


def lanes(k, wmin, wmax, xmin, xmax, s=None, a_bits=A_BITS):
    """Budget of k products per DSP sharing x: shift s, depth N, packed product width.

    s = None takes the largest shift that fits A. N = 0: the low lanes do not even hold one
    product, k lanes are not possible.
    """
    s = max_shift(k, wmin, wmax, a_bits) if s is None else s
    pmin, pmax = product_range(wmin, wmax, xmin, xmax)
    n = depth(s, pmin, pmax) if k > 1 else 1 << 62
    a_lo = sum(wmin << (i * s) for i in range(k))
    a_hi = sum(wmax << (i * s) for i in range(k))
    prod = max(abs(a_lo), abs(a_hi)) * max(abs(xmin), abs(xmax))
    return {"k": k, "s": s, "n": n, "prod_bits": int(prod).bit_length() + 1,
            "fits_b": xmin >= -(1 << (B_BITS - 1)) and xmax < 1 << (B_BITS - 1)}


def groups(tn, n):
    """Lane groups of the Tn addition tree: sizes of the packed partial sums."""
    return [n] * (tn // n) + ([tn % n] if tn % n else [])


def table(tn=24):
    """Rows (case, k, s, N, groups of Tn) of the budget, for the report."""
    cases = [
        ("int8 × int8, any value", 2, -128, 127, -128, 127, S, A_BITS),
        ("int8 × int8, contract ±127", 2, -QMAX, QMAX, -QMAX, QMAX, S, A_BITS),
        ("int8 × int8, contract ±127, s = 19", 2, -QMAX, QMAX, -QMAX, QMAX, 19, A_BITS),
        ("int8 × int8, contract ±127, DSP48E1 (A 25 bits)", 2, -QMAX, QMAX, -QMAX, QMAX, None,
         25),
        ("int4 (±8) × int8, s = 18", 2, -8, 7, -QMAX, QMAX, S, A_BITS),
        ("int4 (±8) × int8, 3 lanes", 3, -8, 7, -QMAX, QMAX, None, A_BITS),
        ("int4 (±8) × int8, 4 lanes", 4, -8, 7, -QMAX, QMAX, None, A_BITS),
    ]
    rows = []
    for name, k, wmin, wmax, xmin, xmax, s, a_bits in cases:
        b = lanes(k, wmin, wmax, xmin, xmax, s, a_bits)
        g = groups(tn, min(b["n"], tn)) if b["n"] > 0 else []
        rows.append({"case": name, **b, "groups": g})
    return rows


def main():
    print("| Case | Products / DSP | s | N | Packed product (bits) | Groups of Tn = 24 |")
    print("|---|---|---|---|---|---|")
    for r in table():
        n = "∞" if r["n"] >= 1 << 62 else str(r["n"])
        g = r["groups"]
        g = ("impossible" if not g else f"{len(g)} × {g[0]}" if len(set(g)) == 1
             else " + ".join(map(str, g)))
        print(f"| {r['case']} | {r['k']} | {r['s']} | {n} | {r['prod_bits']} | {g} |")


if __name__ == "__main__":
    main()
