# Streaming architecture against the single engine — Tiny-YOLOv2 VOC, KV260 (M9.4)

Second architecture family of §10.1:

- one hardware stage per conv, pipelined [2018-venieris#004.0];
- sliding window by line buffer [2023-montgomerie-corcoran#011.0];
- fused maxpool, including the 2×2/1 of L11 (border replication, §10.3).

Files:

- resource model: `tools/stream_model.py`;
- kernel: `hls/stream/` (`conv_stage.hpp`, `yolo_stream.cpp`);
- testbench: `hls/tb/tb_stream.cpp`.

**All the resources below are estimates**: Vitis is missing, hence no synthesis.

## Verification (g++ C-sim)

- The output of each stage and the head are **equal to the byte to the dumps** (== golden
  == integer model) on the 3 images (`tb_stream`, ctest).
- The C-sim cycles per stage (PE × SIMD iterations, II = 1) are **equal** to those of the
  model (`python/tests/test_stream_model.py`). The foldings coded in `yolo_stream.hpp` are
  those of the plan.

## Plan for the KV260, 8-bit weights and activations

Budgets: 80 % of the 1,248 DSPs, 80 % of BRAM + URAM (2.3 MB). One int8 MAC per DSP, plus
4 DSPs per PE for the requantization. Clock 200 MHz.

| conv | K | C_in → C_out | H×W | PE × SIMD | DSP | cycles | weights | buffer (KB) |
|---|---|---|---|---|---|---|---|---|
| L00 | 3 | 3 → 16 | 416×416 | 8 × 3 | 56 | 3,115,008 | chip (0 KB) | 8.9 |
| L02 | 3 | 16 → 32 | 208×208 | 4 × 16 | 80 | 3,115,008 | chip (4 KB) | 13.0 |
| L04 | 3 | 32 → 64 | 104×104 | 2 × 32 | 72 | 3,115,008 | chip (18 KB) | 13.0 |
| L06 | 3 | 64 → 128 | 52×52 | 1 × 64 | 68 | 3,115,008 | chip (72 KB) | 13.0 |
| L08 | 3 | 128 → 256 | 26×26 | 1 × 64 | 68 | 3,115,008 | chip (288 KB) | 13.0 |
| L10 | 3 | 256 → 512 | 13×13 | 1 × 64 | 68 | 3,115,008 | chip (1,152 KB) | 13.0 |
| L12 | 3 | 512 → 1024 | 13×13 | 1 × 256 | 260 | 3,115,008 | ddr (4,608 KB) | 84.5 |
| L13 | 3 | 1024 → 1024 | 13×13 | 1 × 256 | 260 | 6,230,016 | ddr (9,216 KB) | 338.0 |
| L14 | 1 | 1024 → 125 | 13×13 | 1 × 32 | 36 | 676,000 | chip (125 KB) | 0.0 |

- II = 6,230,016 cycles → **32.1 img/s** at 200 MHz; latency ≈ 85.2 ms
- DSP 968 / 998 (80 %); on-chip memory 2.15 / 2.31 MB; DDR 13.50 MB of weights per image
- 223.8 GOPS, MAC efficiency 63.0 %
- single engine (perf model, 8-bit ports): 41,299,241 cycles → 4.8 img/s; streaming × 6.6
  in throughput

**The weights do not fit on chip.** Tiny-YOLOv2 has 15.9 MB of int8 weights for 2.3 MB of
usable memory. L12 (4.5 MB) and L13 (9 MB) hold 86 % of them. SATAY keeps all its
parameters on chip [2023-montgomerie-corcoran#006.0], but on a VCU110, a much larger part.
On the KV260, a **hybrid** is therefore needed:

- the 7 stages with on-chip weights work in pure streaming;
- L12 and L13 keep their input map (13×13×C, 85 and 169 KB) and read their weights once per
  image from DDR, i.e. 13.5 MB per image;
- at 32 img/s, that makes 0.43 GB/s, very far from the 13.4 GB/s usable.

The price is the latency: L12 and L13 each wait for a whole map.

## With weights and activations on 4 bits (M9.3)

Assumption: 2 4-bit MACs per DSP.

| conv | K | C_in → C_out | H×W | PE × SIMD | DSP | cycles | weights | buffer (KB) |
|---|---|---|---|---|---|---|---|---|
| L00 | 3 | 3 → 16 | 416×416 | 16 × 3 | 88 | 1,557,504 | chip (0 KB) | 8.9 |
| L02 | 3 | 16 → 32 | 208×208 | 16 × 16 | 192 | 778,752 | chip (2 KB) | 13.0 |
| L04 | 3 | 32 → 64 | 104×104 | 4 × 32 | 80 | 1,557,504 | chip (9 KB) | 13.0 |
| L06 | 3 | 64 → 128 | 52×52 | 2 × 64 | 72 | 1,557,504 | chip (36 KB) | 13.0 |
| L08 | 3 | 128 → 256 | 26×26 | 1 × 128 | 68 | 1,557,504 | chip (144 KB) | 13.0 |
| L10 | 3 | 256 → 512 | 13×13 | 1 × 128 | 68 | 1,557,504 | chip (576 KB) | 13.0 |
| L12 | 3 | 512 → 1024 | 13×13 | 1 × 256 | 132 | 3,115,008 | ddr (2,304 KB) | 84.5 |
| L13 | 3 | 1024 → 1024 | 13×13 | 1 × 512 | 260 | 3,115,008 | ddr (4,608 KB) | 338.0 |
| L14 | 1 | 1024 → 125 | 13×13 | 1 × 64 | 36 | 338,000 | chip (62 KB) | 0.0 |

- II = 3,115,008 cycles → **64.2 img/s** at 200 MHz; latency ≈ 50.3 ms
- DSP 996 / 998 (80 %); on-chip memory 1.34 / 2.31 MB; DDR 6.75 MB of weights per image
- 447.6 GOPS, MAC efficiency 67.9 %
- single engine (perf model, 8-bit ports): 41,299,241 cycles → 4.8 img/s; streaming × 13.3
  in throughput

## Comparison

| | Single engine (M6-M8) | 8-bit streaming | 4-bit streaming |
|---|---|---|---|
| cycles per image | 41.3 M (M6 kernel, 8-bit ports); 7.44 M (M10 kernel) | II = 6.23 M | II = 3.12 M |
| throughput | 4.8 img/s; 26.9 img/s | **32.1 img/s** | **64.2 img/s** |
| latency | 206 ms; 43 ms | ≈ 85 ms | ≈ 50 ms |
| DSP | 896 (Tm·Tn = 768 + 4·Tm requantization) | 968 | 996 |
| on-chip memory | ping-pong buffers (roofline: 112 BRAM18 ≈ 0.25 MB) | 2.15 MB (L12-L13: whole maps, FIFOs) | 1.3 MB |
| weights read from DDR per image | 15.9 MB, plus the activations at each layer | 13.5 MB (L12, L13) | 6.75 MB |
| MAC efficiency | 11 %; 53 % | 63 % | 68 % |
| flexibility | one bitstream for v2 and v3 | one bitstream per network (dimensions as template parameters) | same |

Reading:

- **Throughput.** At comparable DSP counts, streaming does **6.6 times** better than the
  current engine and **1.4 times** better than its best costed direction. All the layers
  compute at the same time, and the DDR traffic of the activations disappears.
- **What limits the streaming design.** The folding is done by divisors of C_in and C_out.
  L13 (1024 → 1024) jumps from 256 to 512 MACs per cycle in one go, and that jump does not
  fit in the budget. L13 therefore sets the II at twice that of the other stages: the
  efficiency is 63 %, not 100 %.
- **Latency.** It stays lower than the current engine, but the optimised engine (43 ms)
  does better than 8-bit streaming (85 ms), because of the two whole-map stages.
- **4 bits.** They halve the weight memory and double the MACs per DSP, which doubles the
  throughput, subject to the mAP of the 4-bit QAT (`results/quant_4bits.md`).

Memory (T10.8): the plan counts the "outer PE" order of L12-L13 (L12 passes to L13 in CHW,
L13 keeps a 169 KB output map for L14) and the FIFOs between stages (one output row, 52 KB
in 8 bits). The plan does not change; the on-chip memory goes from 1.94 to 2.15 MB in 8
bits.

## Remaining

- Synthesis of `yolo_stream`: real DSP, LUT, BRAM/URAM and frequency reached (the PE × SIMD
  loops unrolled by 256 are a timing risk).
- RTL co-simulation of the DATAFLOW (`make hls-cosim-stream`).
- Board measurement (block design `ENGINE=stream`, T10.9).

Done in C-sim (T10.8, T10.9): weights in ROM (`tb_stream_rom`), outer PE order of L12-L13,
FIFO depths, AXI-Stream with TLAST and emulated AXI DMA (`yolo_bench --engine stream`).

Figure: `results/figures/resultats/ressources.png` (T13.24, estimates against the KV260 budget), `python -m tools.figures ressources`.
