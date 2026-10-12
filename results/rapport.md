# Comparative report (T8.3)

This report places the accelerator of this repository (single engine, layer by layer,
Vitis HLS, Kria KV260) against the works of the base (§10.4), and costs the optimisations
that remain to be done.

**State of the measurements (2026-10-05).** The KV260 and Vivado/Vitis are not available
yet. The figures of "this work" are therefore a **projection**: that of the cycle model
`tools/perf_model.py`, equal cycle for cycle to the C-sim counters of the kernel. This
projection is a **lower bound of the accelerator time**: it assumes an inner loop at II = 1
and ignores the pipeline depths, the DDR latency and the ARM control. The board
measurements will replace these rows following [protocole.md](protocole.md); the sections
that depend on them are marked *to measure*.

## 1. This work

| | Tiny-YOLOv2 VOC | Tiny-YOLOv3 COCO |
|---|---|---|
| MACs (convolutions) | 3.486 G | 2.782 G |
| Format | INT8 (per-channel weights, per-layer activations), 32-bit accumulator | same |
| Tiles, MAC/cycle | Tm = 32, Tn = 24, Tr = Tc = 13 (14 for the convs pooled with stride 2) → 768 | same |
| Clock | 200 MHz (timing to confirm, T7.1) | same |
| Accelerator, projection (M10 kernel) | **37.2 ms**, 26.9 img/s, **187.3 GOPS**, efficiency 61.0 % | 36.1 ms, 27.7 img/s, 154.3 GOPS, 50.2 % |
| Accelerator, projection (M6 kernel, 8-bit ports) | 206.5 ms, 4.8 img/s, 33.8 GOPS, 11.0 % | 203.2 ms, 4.9 img/s, 27.4 GOPS, 8.9 % |
| ARM post-processing | < 0.02 ms on x86 (T7.3); A53 *to measure* | same |
| Power | *to measure* (chip: `power.rpt`; SOM: INA260) | — |
| LUT / DSP / BRAM | *after synthesis*; roofline model: 896 DSP, 112 BRAM18 | — |
| mAP VOC2007 test | float 56.30, integer 55.66, **FPGA (C-sim) 55.66** ([map_stades.md](map_stades.md)) | (COCO, not evaluated) |

Bit-exactness is checked end to end. On the **4,952 images** of VOC2007 test, the ARM
driver driving the C-sim kernel produces detections **identical** to those of the Python
integer model, and therefore the same mAP (55.66). The kernel was built with native
integers (`-DACC_NO_APINT`, same arithmetic); the first 1,832 images were also computed
with `ap_int`, with the same result. The loss of accuracy therefore comes entirely from
the quantization (§9, T4.5). The mAP of the board will be that of the integer model if
`run_compare` confirms 0 differences there.

## 2. Comparison with the base

Source: [benchmarks.csv](benchmarks.csv), §10.4. GOPS = 2 × MACs / time, unless the paper
says otherwise.

| Work | Model | FPGA | Format | img/s | ms | GOPS | W (scope) | What prevents a direct comparison |
|---|---|---|---|---|---|---|---|---|
| **this work** (projection, M10 kernel) | Tiny-YOLOv2 VOC 416 | KV260 (XCK26), 200 MHz | INT8 | 26.9 | 37.2 (accel.) | 187.3 | *to measure* | C-sim projection, accelerator only, no measurement; resources after synthesis |
| **this work** (projection, M10 kernel) | Tiny-YOLOv3 COCO 416 | KV260 | INT8 | 27.7 | 36.1 (accel.) | 154.3 | *to measure* | same; COCO weights (80 classes) |
| 2024-zhang | YOLOv2-Tiny 416, batch 1 | Kintex-7 325T | INT8 | n/r | 14.49 | 406 | 12.3 (board); chip 7.8 (Vivado) | FPGA without an ARM, 687 DSP with two MACs per DSP (*packing*), hardware post-processing included; dataset to check; frequency not noted |
| 2024-zhang | YOLOv3-Tiny 416, batch 1 | Kintex-7 325T | INT8 | n/r | 15.2 | 401 | 12.3 (board) | same |
| 2023-zhai | **pruned** YOLOv3-tiny, 1 module | Zynq XC7Z035 | INT16 | 91.65 | — | 67.91 | 12.51 (unspecified) | pruned network (fewer MACs: GOPS and img/s do not convert), INT16, vehicle dataset (detection and tracking), unknown power scope |
| 2023-zhai | pruned YOLOv3-tiny, 2 modules | Zynq XC7Z035 | INT16 | 168.72 | — | 124.01 | 15.18 (unspecified) | same |
| 2026-fata | YOLOv3-tiny **pruned by 70 %** | KV260 (DPU) | INT8 QAT | 24.3 | — | 60.18 | 2.13 (unspecified) | same board, but a preconfigured Vitis AI DPU (not hand-designed), 70 % pruning, QAT |
| 2025-kim | **full** YOLOv2 | Zybo Z7-20 | INT16 | ≈ 0.08 | ≈ 12,000 | — | — | slow working prototype, full network (not Tiny): not a performance reference |

**Reading.**

- **2024-zhang** is the only reference on the same model (Tiny-YOLOv2 416, INT8, batch 1).
  It was **14× faster** than the projection of the M6 kernel (14.49 against 206.5 ms) with
  only 1.8× more multipliers (687 DSP with *packing*, i.e. about 1,374 MAC/cycle, against
  768). The M6 kernel spent 80 % of its time outside the MAC loop (41.3 Mcycles, 8.2 of
  them compute), waiting for its memory ports and its output stage. The M10 kernel
  (section 3) brings the gap down to 2.6× (37.2 ms): what remains is roughly the gap in
  multipliers (1.8×) and 39 % of the time outside the MAC loop.
- **2026-fata** runs on the **same board**: 60 GOPS and 24 img/s with a DPU on a YOLOv3-tiny
  pruned by 70 %. Our 27.4 projected GOPS on an unpruned Tiny-YOLOv3 are 2.2× below. The
  img/s do not compare: pruning divides the MACs by about 3. Our roofline point on this
  board (31.9 ms for Tiny-YOLOv2, about 218 GOPS) is above the DPU: the gap is to be made up
  on the implementation side, not on the choice of the board.
- **2023-zhai**: INT16 and a pruned network. Only the order of magnitude of the GOPS
  compares (68 to 124 GOPS on a Zynq-7035 against our projected 34).
- **Power**: only 2024-zhang gives two scopes (estimated chip and measured board). Our two
  scopes (chip by `report_power`, SOM by the INA260) compare with these two figures
  respectively. Neither compares with the *unspec.* values of 2023-zhai and 2026-fata.
- **mAP**: the base does not note a VOC2007 mAP for these rows (empty column in
  `benchmarks.csv`), and the datasets differ (vehicles for 2023-zhai). Our chain measures
  the quantization loss alone: −0.64 point for the integer model against the float one,
  then 0 between the integer model and the FPGA in C-sim (4,952 identical images).

## 3. Where the time goes, and costed directions

Model: `python tools/perf_model.py` (`make perf-model`), equal to the C-sim counters cycle
for cycle. Cycles at 200 MHz, accelerator only. The wide ports follow a realistic,
row-by-row model: a row of n bytes costs `cdiv(n + 7, 8)` 64-bit words, whatever its
alignment.

### Tiny-YOLOv2 VOC

| Scenario | Mcycles | ms | img/s | GOPS | MAC efficiency |
|---|---|---|---|---|---|
| M6: ports 8 bits | 41.30 | 206.5 | 4.8 | 33.8 | 11.0 % |
| valid channels only (trim) | 32.67 | 163.3 | 6.1 | 42.7 | 13.9 % |
| ports 64 bits | 12.27 | 61.3 | 16.3 | 113.6 | 37.0 % |
| ports 128 bits + trim | 12.13 | 60.6 | 16.5 | 115.0 | 37.4 % |
| ports 64 bits + trim + requantization ×8 (+28 DSP) | 9.29 | 46.4 | 21.5 | 150.1 | 48.9 % |
| … + L00 folding | 8.38 | 41.9 | 23.9 | 166.4 | 54.2 % |
| **… + 14×14 tiles for pooled convs (M10 kernel)** | **7.44** | **37.2** | **26.9** | **187.3** | **61.0 %** |
| compute bound of M6 (free loads) | 8.23 | 41.1 | 24.3 | 169.5 | 55.2 % |
| compute bound of the M10 kernel | 6.32 | 31.6 | 31.7 | 220.7 | 71.8 % |
| KV260 roofline point ([roofline.md](roofline.md)) | — | 31.9 | 31.3 | 218.5 | — |
| ideal (MACs / 768, efficiency 100 %) | 4.54 | 22.7 | 44.1 | 307.2 | 100 % |

Tiny-YOLOv3 follows the same scale: 203.2 → 60.4 ms (64-bit ports) → 45.3 ms
(+ requantization ×8) → 36.1 ms (M10 kernel); compute bound of the M10 kernel: 27.0 ms.

**M10 kernel (T10.1 to T10.4, C-sim == golden to the byte).**
- **What it contains**: 64-bit ports, weights reordered in tile order, trim, requantization
  ×8, folding of L00 and 14 × 14 tiles for the convs pooled with stride 2.
- **Throughput**: Tiny-YOLOv2 goes from 206.5 to 37.2 ms, i.e. ×5.6. It beats 2026-fata
  (24.3 img/s) in image throughput, without pruning.
- **What remains**: 5.3 ms separate it from the roofline point. L00 stays limited by its
  output stage, and the 1×1 convs by the loading of `in_buf` (IR rows per lane).
- **What is missing**: the resources (DSP, BRAM of the 14 × 14 buffers) and the byte mask
  of the partial writes. Only synthesis and co-sim will tell.

**Directions, by decreasing gain:**

1. **Wide m_axi ports (×3.4).** The four ports read and write one byte per cycle, i.e.
   0.2 GB/s per bundle, while the roofline assumes 13.4 GB/s of DDR. Loading them in 64
   bits (`in_buf` by rows, `w_buf` in tile order, reordered once by the driver) brings
   Tiny-YOLOv2 from 206.5 down to 61.3 ms. Moving to 128 bits brings almost nothing more
   (60.6 ms): the bottleneck then moves to the output stage.
2. **Parallel requantization (−24 %, after direction 1).** `store_tile` requantizes one
   channel at a time with a single 32 × 31-bit multiplier (4 DSP), i.e. Tm × 169 cycles
   per tile. With 8 multipliers (+28 DSP, an abundant resource: 896 used out of 1,248), it
   goes down to 46.4 ms.
3. **Valid channels only (−21 % on the current kernel, ≈ 0 after direction 1).** L00 has
   cin = 3, but the kernel loads Tn = 24 channels: 87 % of the load of L00 is useless. The
   direction only matters as long as the ports stay narrow.
4. **Compute bound (41.1 ms) against roofline point (31.9 ms).** Two tiling losses that
   the roofline ignores remain. L00 (cin = 3, cout = 16) only uses 6 % of the 768
   multipliers: 1.86 Mcycles, i.e. 23 % of the compute bound, for 2 % of the MACs. With a
   2×2/2 pooling and Tr = Tc = 13, each tile only produces 12 × 12 useful outputs out of
   13 × 13. The remedies are a smaller Tn or a dedicated PE for L00 (like the 3×3 PE of
   2025-kim) and Tr = Tc = 12 or 14 for the pooled layers.
5. **ARM side (on the board).** Overlapping the preprocessing of image i + 1 with the
   accelerator of image i (two threads, two arenas) would make the throughput the max of
   the stages rather than their sum. The gain will be costed with the `pre` times of the
   board. If the copy and the post-processing in uncached memory (`O_SYNC`) weigh, a
   cached buffer with `sync_for_cpu/device` (u-dma-buf) will be needed.

Directions 1 to 4 are done in C-sim in the M10 kernel (see the table). The projected
Tiny-YOLOv2 reaches **26.9 img/s and 187 GOPS** there, beyond 2026-fata (DPU, same board,
pruned network), and without pruning. Direction 5 is tooled (`yolo_bench --pipeline`,
`--cached`) and waits for the board.

Beyond the single engine (M10, sections C and D):

- **Streaming** ([streaming.md](streaming.md)): 32.1 img/s in 8 bits. Weights in ROM, outer
  PE order and AXI DMA are checked in C-sim.
- **Mixed precision** ([precision_mixte.md](precision_mixte.md)): L10, L12 and L13 in 4
  bits, L02 and L08 in 6 bits. mAP 54.68 (−1.01) for 8.19 MB of weights, and 64.2 img/s in
  streaming.
- **Pruning** ([elagage.md](elagage.md)): up to 82.5 img/s projected at 70 %, but the mAP
  does not recover in 300 fine-tuning iterations.

Note: `results/hls_report.md` gives 210.5 ms instead of 206.5 ms. Its `tb_conv` writes the
pre-pooling map of all the pooled convs to test that path. The program of the driver only
writes it for L08 of Tiny-YOLOv3: that is the case the projection follows.

## 4. What remains to be measured

| Figure | Command | Depends on |
|---|---|---|
| Timing met at 200 MHz, LUT/DSP/BRAM, chip power | `make vivado-build` | Vitis HLS + Vivado |
| Real cycles per layer (II, pipeline) | `make hls-cosim`, then `run_compare --csv` on the board | Vitis, KV260 |
| Latency per stage (mean, p99), FPS, SOM power | `yolo_bench --images` (1,000 images), [protocole.md](protocole.md) | KV260 |
| mAP on FPGA | `yolo_bench --inputs` then `make map-stades` | KV260 |

Once these files are brought back, `tools/bench_report.py` regenerates `benchmarks.csv`
and `mesures.md`. The "projection" row of the tables above will then be replaced by the
measurement, and the per-layer efficiency set against the cycle model.
