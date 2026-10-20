# Model ↔ circuit equivalence (M28)

What is shown, by what evidence, between which stages of the integer chain:

1. Python integer model (`IntNetwork`, the specification);
2. golden C++ (`cpp/golden`);
3. HLS kernel in C-sim (`hls/kernels`, gcc).

Levels as in CoDANN II §3.6.1: **pointwise** (same output), **bounded proximity**,
**statistical**. Generated on 2026-10-20 on the four exports of `model/`: Tiny-YOLOv2 VOC,
Tiny-YOLOv3 COCO, Tiny-YOLOv4 COCO, YOLOv7-tiny COCO.

| Task | Evidence | State |
|---|---|---|
| T28.0 equivalence matrix per layer | `tools/equivalence.py` | to do |
| T28.1 exhaustive operator tests | `python/tests/test_exhaustive_ops.py`, `cpp/golden/tests/test_ops_exhaustive.cpp`, `hls/tb/tb_ops.cpp` | **done** |
| T28.2 formal proof of the scalar kernels | `tools/prove_kernels.py` (z3) | **done** |
| T28.3 per-layer traceability | `tools/traceability.py` → [docs/traceability.md](../docs/traceability.md), in `make ci` | **done** |
| T28.4 tool-chain risks | this file | to do |
| T28.6 post-processing properties | — | to do |

## T28.1: exhaustive and boundary tests of the operators

### Output stage (requantization, leaky, saturation)

The output stage is the only non-linear integer code between two convs: requantization,
leaky and saturation. Max-pool, route and upsample are exact copies or compares.

**Method**: for each conv of each export and each channel, the accumulator range that can
actually be reached is computed **exactly** (`bounds.acc_range`). The extrema of a linear
function over the box of int8 inputs lie at a vertex, which is reached. Over that range:

- **every** value when the range holds ≤ 2¹⁶ values;
- otherwise:
  - every value next to a step of the requantized integer that can still change the int8
    output: k·2ⁿ − 2ⁿ⁻¹ divided by M0, ±1, from the leaky saturation to qmax + 1;
  - both ends, zero;
  - 1,024 evenly spaced values.

  The output is a non-decreasing step function of the accumulator, so these points
  hold both sides of every step. `test_sweep_covers_every_step` checks this property on a
  dense range.

The same accumulators go through three implementations, which are compared byte for byte:

- the Python `conv_out`;
- the golden (`build/golden/golden_ops requant`);
- the HLS output stage `store_tile` with its RQ parallel lanes and DDR writes
  (`build/hls/tb_ops --ops`).

| Network | Channels | Accumulators compared | Python == golden == HLS |
|---|---|---|---|
| tiny-yolov2-voc | 3,181 | 16,021,501 | yes |
| tiny-yolov3-coco | 3,694 | 17,377,783 | yes |
| tiny-yolov4-coco | 3,614 | 16,917,261 | yes |
| tiny-yolov7-coco | 8,669 (512 swept exhaustively) | 55,298,264 | yes |

**Run time** (`pytest tests/test_exhaustive_ops.py`): 33 s for 105.6 M values. Python takes
5 to 12 s per network, the golden 0.1 to 0.2 s, the HLS C-sim 0.4 to 2.9 s.

**Other checks**:

- **`tb_ops` self-check** (ctest): 21.6 M accumulators. They cover random (M0, n), the int32
  extremes and the rounding boundaries, for qmax 127 and 7.
- **`test_ops_exhaustive.cpp`**: the golden against a specification in 128-bit integers
  (floor division, not shifts). It covers every n ∈ [1, 31], 200 M0 per n, every rounding
  boundary of k ∈ [−1400, 1400] ± 1 and the int32 extremes: 31 M assertions.
- **SiLU** (T26.2): the integer SiLU is a requantization to the table scale, then a 256-entry
  table. Python == golden == HLS on every accumulator around each of the 256 table inputs,
  for 5 random (s_mid, s_out, M0, n, qmax).

### Finding: the golden leaky overflowed int32

The first run of `test_ops_exhaustive.cpp` failed. `golden::leaky_int` computed 13·y in
int32, which overflows (undefined behaviour) for y < −165,191,049. The Python computes it in
int64.

- **Effect on the exports**: none. The reachable requantized values stay above −24,000 on
  the four exports, and the M0·2⁻ⁿ multipliers stay ≤ 0.023.
- **Effect on the argument**: it was a hole in the contract "Python == golden for every
  int32".
- **Fix**: the product is now on 64 bits (`conv.hpp`). The HLS reuses the function: 13·y on
  36 bits, a constant multiply.
- **Check**: every bit-exact test is unchanged (`make csim-gcc`, `make golden-check`, 103
  layers in `tb_conv`, 158 layer outputs in `tb_net`).

The same review made a second precondition explicit: the requantization equals the Python
only when **M0 < 2ⁿ** (multiplier < 1). Otherwise the shifted value can leave int32, which the
C++ truncates and the Python does not. `golden::Model::load` now rejects an export that breaks
it. All 103 layers meet it.

### Tables (LUT)

For each head of each export, the σ, e^t and softmax e^{d·s} tables loaded by the golden
from `luts.bin`, indexed as `hw_postproc` indexes them, are compared over their 256 inputs.
They equal the tables recomputed in Python from the head scale (`test_luts_python_golden`).

## T28.2: SMT proof of the scalar kernels

`python tools/prove_kernels.py` (z3 5.3.1, optional extra `formal`) states the following
obligations:

- **Model**: the C++ of `requantize`, `leaky_int` and `clip_q`, re-expressed in z3
  bit-vectors with the C++ widths.
- **Specification**: the Python, written as mathematics. Floor division is stated by its
  defining inequalities, over 72-bit exact integers.
- **Quantifiers**: for **all** int32 accumulators, all M0 with 0 < M0 < 2ⁿ, all
  qmax ∈ [1, 127], and every shift n ∈ [1, 31] (one query per n).

| Obligation | Queries | Result | Solver time |
|---|---|---|---|
| (a) `int64(acc)·M0 + 2ⁿ⁻¹` does not overflow int64 | 31 | proved | ≈ 5 s |
| (b) the shifted value fits int32 (the cast loses nothing) | 31 | proved | ≈ 52 s |
| (c) y = ⌊(acc·M0 + 2ⁿ⁻¹)/2ⁿ⌋ | 31 | proved | ≈ 50 s |
| (d) leaky = y if y > 0, else ⌊(13y + 64)/128⌋, ∀ y int32 | 1 | proved | < 0.1 s |
| (e) clip to [−qmax, qmax], fits int8 | 1 | proved | < 0.1 s |

The run proves 95 / 95 obligations in 135 s, including the loading of the four exports. The
longest query takes ≈ 12 s (n = 31).

**Mutants** (`--mutant`), which must be caught:

| Mutant | Counter-example printed |
|---|---|
| no rounding term (`>> n` of the bare product) | n = 20: acc = −985,670,293, M0 = 1,031,404 |
| leaky with 13·y on 32 bits (the code before T28.1) | y = −1,636,502,221 |

### No int32 overflow of the accumulator, per layer

This is proved from the exported int8 weights.

- **Extrema**: the extrema of Σ w·x + b over x ∈ [−128, 127]ⁿ are those of
  `bounds.acc_range`. A z3 query checks this vertex lemma on a random layer.
- **Partial sums**: every partial sum of the tiled loop (a subset of the products, without
  the bias, which `store_tile` adds last) lies between the sum of the negative and the sum
  of the positive per-product extrema. The HLS `acc_t` register is therefore covered too.

| Network | Convs | Widest accumulator (bits, sign incl.) | int32 overflow | M0 < 2ⁿ |
|---|---|---|---|---|
| tiny-yolov2-voc | 9 | 27 (L13, ±52,931,184) | none | all |
| tiny-yolov3-coco | 13 | 25 (L12, ±16,527,467) | none | all |
| tiny-yolov4-coco | 21 | 25 (L26, ±16,168,086) | none | all |
| tiny-yolov7-coco | 60 | 25 (L30, ±9,234,011) | none | all |

The T4.3 prose bound of L13 was Σ|q_w|·127 + |q_b| = 2^25.7. The exact bound with inputs
down to −128 is 2^25.66: 26 bits plus the sign.

### Residual risk

- **Hand-written model**: the z3 model is written by hand from the C++ source. A gap between
  the two is covered by T28.1, which runs the real code.
- **Toolchain**: the proof says nothing of what Vitis HLS does to the C++ (T28.4, co-sim,
  board).
- **Scope**: the composition of the scalar kernels into the tiled conv is covered by the
  bit-exact tests on the dumps (golden == C-sim == Python on every layer), not by the
  proof.

## T28.3: traceability

[docs/traceability.md](../docs/traceability.md) maps 14 operations of the cfg to their
Python, golden C++ and HLS function and to their `LayerDesc` fields, with file and line. The
operations are conv MACs, requantization, leaky, linear, SiLU, saturation, ABFT, max-pool
2×2 and 5×5, pre-pool map, route, upsample, decode and NMS.

- **Generation**: `python tools/traceability.py`.
- **CI**: `make ci` runs `--check`, which fails if a symbol disappears or the document is
  stale.
