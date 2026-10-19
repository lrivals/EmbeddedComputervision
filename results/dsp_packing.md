# Two MACs per DSP (M23)

The single engine's compute bound only goes down if the MACs per cycle go up. 2024-zhang
reaches its multiplier count by packing two int8 products that share an operand into one
DSP48E2 [2024-zhang#011.0]. Xilinx WP486 ("Deep Learning with INT8 Optimization on Xilinx
Devices", outside the base) describes the same scheme. The contract of
[conventions.md](../docs/conventions.md#integer-arithmetic-python--c--hls-contract) does not
change: only the multiplication changes; the int32 accumulator, requantization and
saturation stay as they are. **C-sim == golden to the byte.**

Code: `python/yolo/quant/pack.py` (reference, budget), `golden::pack2 / mac2 / unpack2`
(`cpp/golden/include/golden/conv.hpp`), `accel::mac2` and the `mac:` loop with
`ACC_PACK=2` (`hls/kernels/`). Table below: `python -m yolo.quant.pack`.

## Scheme (T23.0)

The PE multiplies one activation `x` by the Tm weights of the output channels. The weights
of filters `too` and `too + 1` are packed into the A port (27 bits), and `x` goes to B
(18 bits):

    A = w_a·2^s + w_b,      A·x = (w_a·x)·2^s + w_b·x      (exact, P on 48 bits)

- **Shift s = 18**: the largest shift for which every int8 pair fits in 27 signed bits
  (s = 19 overflows for w_a = −128, w_b < 0).
- **Unpacking**: `lo = sign_extend(P[17:0])`, `hi = (P >> 18) + (lo < 0)`. A negative low
  lane has borrowed 1 from the high lane, which the `+ (lo < 0)` term gives back.
- **Accumulation depth N**: a sum of N packed products stays exact as long as the low lane
  fits in 18 signed bits, i.e. in [−2^17, 2^17). For any int8 operands, max |w·x| = 16384,
  so **N = 7**. Under the contract (|w|, |x| ≤ 127: −128 is never produced; the weights are
  symmetric ±127 and the input is in [0, 127]), max = 16129, so **N = 8**.
- **Tn = 24 addition tree**: one packed sum over all 24 lanes would need a 20-bit low lane,
  hence s = 20, and A would need 28 bits. The tree is therefore cut into **3 groups of
  8 lanes**. Each group sum is unpacked, then the int32 group sums are added. The high lane
  needs at most 34 + 3 bits, well inside P.

| Case | Products / DSP | s | N | Packed product (bits) | Groups of Tn = 24 |
|---|---|---|---|---|---|
| int8 × int8, any value | 2 | 18 | 7 | 34 | 7 + 7 + 7 + 3 |
| int8 × int8, contract ±127 | 2 | 18 | 8 | 33 | 3 × 8 |
| int8 × int8, contract ±127, s = 19 | 2 | 19 | 16 | 34 | 16 + 8 |
| int8 × int8, contract ±127, DSP48E1 (A 25 bits) | 2 | 17 | 4 | 32 | 6 × 4 |
| int4 (±8) × int8, s = 18 | 2 | 18 | 129 | 29 | 1 × 24 |
| int4 (±8) × int8, 3 lanes | 3 | 11 | 1 | 33 | 24 × 1 |
| int4 (±8) × int8, 4 lanes | 4 | 7 | 0 | 32 | impossible |

The kernel keeps s = 18 and N = 8. s = 19 would halve the number of unpacks (16 + 8), but
it relies on the contract even for the A operand. On 7-series (DSP48E1, PYNQ-Z2, Zybo),
the 25-bit A port still allows 2 products per DSP, with groups of 4.

Tests: `python/tests/test_pack.py`, exhaustive over the 2^24 int8 triples (w_a, w_b, x).
The depth N is tight: N + 1 extreme products overflow.

## Golden model and kernel (T23.1, T23.2)

- `golden::mac2` is header-only, with no allocation. The Catch2 test covers all 2^24
  triples, plus depth-N partial sums (100,000 random vectors and the ±127 extremes).
- Kernel, `ACC_PACK=2` (default 1 until the board is measured): the `too` loop runs over
  pairs. Per lane, one `mac2` is bound to a DSP (`BIND_OP … impl=dsp`, an ap_int 27 × 18
  → 45-bit product in a 48-bit sum). Every PACK_N lanes (every Tn lanes for a 4-bit
  layer), the sum is unpacked and added to the two int32 sums. The masked lanes
  (`tii ≥ n`, T10.3) add 0. If Tm were odd, the last filter would keep the unpacked path.
  Today `TM % (2·WORD) == 0` makes Tm even; the odd case that actually occurs is an odd
  number of valid channels in the last tile (heads with 125 and 255 channels), where the
  partner filter is unused.
- C-sim, every result == golden == Python dumps, to the byte:

| Testbench | Configuration | Result |
|---|---|---|
| `tb_pack` | ap_int `accel::mac2` == `golden::mac2`, 2^24 triples | 0 mismatches |
| `tb_net_pack` | Tm 32, Tn 24, ap_int and native integers | 87 layer outputs, 0 mismatches |
| `tb_net_pack` | synthetic CI export (`make_ci_model.py`) | 87 layer outputs, 0 mismatches |
| `tb_net_pack` | Tm 64, Tn 24 | 87 layer outputs, 0 mismatches |
| `tb_net_pack` | Tm 32, Tn 32 | 87 layer outputs, 0 mismatches |
| `tb_net_mixed_pack` | mixed precision of T10.10 (L10, L12, L13 in 4 bits) | 0 mismatches |

- `tb_net_pack` runs in `make csim-gcc` / `make ci` with native integers (`ACC_NO_APINT`).
  The packed ap_int PE is about 7× slower in C-sim; its product is covered by `tb_pack`.
- **Cycles unchanged**: `tb_conv_pack` == `perf_model --check` (22 layers, 0 differences),
  equal to the unpacked kernel at equal tiles; and also at Tm 64 and at Tn 32. Packing
  changes the DSP count, not the schedule.

## Projected gain (T23.3)

`python tools/perf_model.py` (exact cycle model, == C-sim), 200 MHz, accelerator only.
PE DSP = ⌈Tm/2⌉·Tn when packed; +32 for requantization ×8. All rows are **projections**.

| Network | Option | Tm, Tn | PE DSP | ms | compute bound (ms) | GOPS | layers not compute-bound |
|---|---|---|---|---|---|---|---|
| Tiny-YOLOv2 VOC | M10 kernel, 1 MAC / DSP | 32, 24 | 768 | 37.2 | 31.6 | 187.3 | L00 store, L14 load |
| Tiny-YOLOv2 VOC | packed, same tiles (half the DSPs) | 32, 24 | 384 | 37.2 | 31.6 | 187.3 | L00 store, L14 load |
| Tiny-YOLOv2 VOC | packed, Tm 64 (same DSPs) | 64, 24 | 768 | 24.7 | 18.1 | 282.5 | L00 store, L10, L12, L13, L14 load |
| Tiny-YOLOv2 VOC | packed, Tn 32 (roofline point) | 32, 32 | 512 | 30.0 | 23.7 | 232.2 | L00 store, L14 load |
| Tiny-YOLOv3 COCO | M10 kernel, 1 MAC / DSP | 32, 24 | 768 | 36.1 | 27.0 | 154.3 | L00 store, L13, L15, L18, L22 load |
| Tiny-YOLOv3 COCO | packed, same tiles (half the DSPs) | 32, 24 | 384 | 36.1 | 27.0 | 154.3 | same |
| Tiny-YOLOv3 COCO | packed, Tm 64 (same DSPs) | 64, 24 | 768 | 23.8 | 15.8 | 233.4 | L00 store, L10, L12–L15, L18, L21, L22 load |
| Tiny-YOLOv3 COCO | packed, Tn 32 (roofline point) | 32, 32 | 512 | 30.2 | 20.3 | 184.0 | L00 store, L13, L15, L18, L22 load |

- **Same tiles**: 384 DSPs freed, no time gained. Useful only if the DSPs are needed
  elsewhere (a second engine, a post-processing PE).
- **Tm 64**: −34 % on Tiny-YOLOv2 (37.2 → 24.7 ms). This option also needs `out_buf`
  and `w_buf` twice as large. The layer-by-layer check shows why the gain is below the
  compute bound: with Tm doubled, the weight blocks double, and L10, L12, L13 and L14
  become limited by the weight port (64 bits). Since `w_buf` is in registers, 2 × 8 × 64 × 24 × 9 =
  221,184 FF are needed, 94 % of the 234,240 of the XCK26. This option only works if
  `w_buf` moves to LUTRAM.
- **Tn 32** is the best point of [roofline.md](roofline.md) once the flip-flops are
  counted (147,456 FF, 63 %): −19 % (30.0 ms), with the C-sim also byte-exact.
- Roofline ([roofline.md](roofline.md), §10.2 model with 13.44 GB/s of DDR): no conv of
  these points is DDR-bound on the KV260. The limits are the m_axi ports and the output
  stage, which `perf_model` models.

Gap to 2024-zhang (YOLOv2-Tiny 416, 14.49 ms, 687 DSPs on a Kintex-7 325T): 2.6× with the
M10 kernel (37.2 ms), **1.7×** with Tm 64 packed (24.7 ms, about 800 DSPs), and 2.1× at
Tn 32 (30.0 ms). The multiplier gap is closed: 1,536 multipliers at Tm 64 against about
1,374 for 2024-zhang (687 × 2). What remains is time outside the MAC loop: the weight and
input loads of the 1×1 and 13×13 layers, and L00's output stage.

## 4-bit weights (T23.4)

- With ±8 weights and int8 activations, |w·x| ≤ 1016. At s = 18, N = 129 ≥ Tn: **2 per
  DSP, ungrouped**, with one unpack per pair for the whole Tn tree instead of 3. The
  kernel takes this path when `d.wbits == 4` (a runtime mux on the group boundary).
- Three per DSP: s = 11 and N = 1, i.e. one unpack per product. Four per DSP is
  impossible with 8-bit activations: s = 7 does not hold one product of 11 bits. It
  would also need a PE sized for the 4-bit layers, which the fixed PE shared with the
  8-bit layers cannot use. The **selected configuration is 2 per DSP**. "Four 4-bit
  products per DSP" would need 4-bit activations as well (W4A4), on a dedicated engine.
- `tb_net_mixed_pack` (mixed export of T10.10, conditional ctest): golden == C-sim to the
  byte.
- Projection (perf_model, mixed manifest): 37.2 ms for the M10 kernel and the same
  tiles. With **Tm 64 packed, 23.2 ms** (against 24.7 ms in INT8): the 4-bit weights
  halve the weight blocks of L10, L12 and L13, which are then no longer load-bound.

## Winograd

Studied in [winograd.md](winograd.md) (T23.5): not implemented. It does not beat packing
once the loads are counted, and the two do not combine with 8-bit DSP inputs.

## Remaining (Vitis and the board)

- Synthesis of the packed PE (`make hls-synth-pack`): DSP, LUT (unpacking, group sums)
  and timing at 200 MHz. Check that `BIND_OP` does map one product per DSP48E2 with the
  pre-adder unused.
- Tm 64 with `w_buf` in LUTRAM, or Tn 32: choose after synthesis.
- Measurement on the KV260 ([protocole.md](protocole.md)).
