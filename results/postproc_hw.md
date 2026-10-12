# Hardware post-processing — Tiny-YOLOv2 VOC (M9.1)

Decoding and NMS in integers only, in an HLS kernel `yolo_post` separate from the
convolution engine, as 2024-zhang does [2024-zhang#013.1, #014.0, #014.2].

- Python reference: `python/yolo/infer/hw_postproc.py`.
- C++ mirror: `cpp/golden/include/golden/hw_postproc.hpp`.
- Kernel: `hls/kernels/postproc.cpp`.
- Integration: `sw/driver`, with the `--hw-post` option of `yolo_app` and `yolo_bench`.

## Arithmetic

| Step | Hardware (`yolo_post`) | ARM post-processing (M7) |
|---|---|---|
| objectness threshold | integer t_o ≥ ⌊logit(θ)/s⌋ + 1 (§9.4) | same |
| σ, e^t, softmax | Q16 tables of `luts.bin`; 3-stage softmax, **one reciprocal ⌊2³²/Σe⌋ per cell** | tables, then division in double |
| boxes | corners in **Q4** pixels: shifts (stride 416/S = 2^k), Q8 anchors | (cx, cy, w, h) in double |
| scores | Q16, threshold score > ⌊θ·2¹⁶⌋ | double |
| IoU > 0.45 | **29·inter > 9·(a₁ + a₂)**, 64-bit integers | division in double |
| NMS | **unsorted**, 256 slots, stream order | stable sort then greedy |

The unsorted NMS works as follows:

- a candidate overlapped by a selected box with a greater or equal score is rejected;
- otherwise, it takes the place of the first selected box it overlaps and invalidates the
  others.

It does not always give the result of the sorted NMS. Example: in a chain a < b < c where
only a∩b and b∩c exceed the threshold, it keeps only c, while the sorted NMS keeps c and
a. This case is tested in `test_hw_postproc.py`.

## Verification

- **Python == C++ to the byte**, through `golden_run hwpp`: on the dumps of both networks,
  at thresholds 0.25 and 0.005, and on 6 sets of random heads
  (`python/tests/test_hw_postproc.py`).
- **Kernel == golden** in g++ C-sim (`tb_post`, `make csim-gcc`): 1,014 cases, including the
  dumps at thresholds 0.25 and 0.005, the worst case where all the cells pass, and 1,000
  sets of random heads (800 for v2, 200 for v3), overflow included.
- **Driver** (sim backend, `run_compare --hw-post`, ctest `run_compare_hw_post`): emulated
  `yolo_post` registers, tables written to DDR. On the heads produced by the accelerator,
  the boxes are equal to those of the golden model, for both networks.
- Boxes identical to the float post-processing on the dumps, to within 2/416 (< 1 Q4
  pixel) for the coordinates and to within 2⁻¹⁵ for the scores.

## mAP (VOC2007 test, 4,952 images, stretch, threshold 0.005)

| Post-processing | mAP | Gap |
|---|---|---|
| integer model, LUT decoding + sorted NMS on the host (T4.5) | 55.66 | — |
| **hardware: all integer, unsorted NMS, 256 slots** | **55.56** | −0.10 |
| hardware, unlimited slots | 55.56 | −0.10 |

The 256-slot selection only overflows 50 times over the 4,952 images at the 0.005
threshold of the mAP, and with no effect on the mAP. The −0.10 point gap therefore comes
from the unsorted NMS and from the Q4 coordinates.

Command: `tools/eval_quant.py --variants int,int-hwpp --resize stretch [--hw-cap N]`.

## Cycles (C-sim estimate, II = 1, latencies ignored)

| Case (Tiny-YOLOv2, 845 cells) | Candidates | Cycles | at 200 MHz |
|---|---|---|---|
| image 000001, threshold 0.25 | 4 | 1,449 | 7.2 µs |
| image 000002, threshold 0.25 | 3 | 1,364 | 6.8 µs |
| image 000002, threshold 0.005 (mAP) | 67 | 9,981 | 50 µs |
| worst case: 845 cells above the threshold, random scores | 1,181 | 342,375 | 1.7 ms |

Reading the cycles:

- **Scan**: 845 cycles, one t_o per cycle.
- **Decoding**: 4 + C + 2C cycles per surviving cell (reads, max, sum and scores of the
  softmax).
- **NMS**: one comparison per occupied slot and per candidate.

Comparison with 2024-zhang, which processes 845 candidates in 0.38 ms, i.e. 76,000 cycles
at 200 MHz [2024-zhang#020.3]:

- On a real image, the kernel is 50 times faster, because the threshold on the integer t_o
  removes almost all the cells before the tables.
- In the worst case (256 selected), the NMS costs 4.5 times more. It would then have to
  compare P slots per cycle. With P = 8, about 34,000 NMS cycles, at the cost of 8 IoU
  tests in parallel.

On the PC, the ARM post-processing of M7 takes between 0.011 and 0.037 ms per image (x86,
threshold 0.25). It remains to be measured on the A53 of the KV260.

## Remaining

- Synthesis of `yolo_post`: LUT/DSP/BRAM, timing at 200 MHz.
- Check of the offsets of `sw/driver/post_regmap.hpp` against the Vitis header.
- Adding the IP to the block design (`hw/boards/kv260/build.tcl`, second UIO node).
- Board measurement of `yolo_bench --hw-post` against the ARM post-processing.

Figure: `results/figures/resultats/hw_postproc.png` (`python -m tools.figures hw_postproc`, T13.25).
