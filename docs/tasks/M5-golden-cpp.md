# M5 — C++ golden model

Goal: a C++ engine that runs the whole network **like the hardware** (tiling, buffers,
fused maxpool, route by addressing) and reproduces the Python dumps exactly, integer for
integer. It serves as the testbench of the HLS stage.

### [x] T5.1 — Reading the manifest and the blobs
- **Spec**: — · **Depends on**: T4.7, T0.3 · **Size**: S
- **Deliverables**: `cpp/golden/include/golden/model.hpp`, `src/model.cpp` (minimal JSON
  reader or header-only nlohmann/json), `.npy` reader for the dumps
- **Acceptance**: all the layers, scales and shifts read back == Python values
- **Done**: `json.hpp` (minimal reader, numbers through `strtod`: exact round trip of the
  Python `repr`; no nlohmann, the library stays dependency-free), `npy.hpp` (int8, read
  and write), `model.hpp` (`Model::load`: layers, scales, little-endian blobs; version
  check, 64 alignment, blob sizes, tensors inside their buffers). CLI
  `golden_run inspect`; `python/tests/test_golden_cpp.py` compares it with `load_model`
  (fields, scales to equality, sums/first/last of the weights, biases, M0, tables).

### [x] T5.2 — Tiled convolution
- **Spec**: §10.2, §9.3 · **Depends on**: T5.1 · **Size**: L
- **Deliverables**: `cpp/golden/include/golden/conv.hpp`: loop nest (Tm, Tn, Tr, Tc) with
  fixed-size `in_buf`, `w_buf`, `out_buf` buffers; output stage bias + requantization +
  leaky + fused maxpool
- **Acceptance**: output == Python dump for each conv layer, **for several tile sets**
  (including tiles that do not divide the dimensions); buffer sizes == formulas $B_{in}$,
  $B_w$, $B_{out}$ of §10.2
- **Notes**: no dynamic allocation in the kernel ([ADR 0002](../adr/0002-hls.md)); reread
  `2015-zhang#014.0`
- **Done**: `conv.hpp` (header only), `conv_layer<Tiles<Tm, Tn, Tr, Tc>>`, buffers
  `Tiles::Buffers` whose size is checked by `static_assert` against B_in, B_w, B_out; no
  allocation in the kernel. Tests: each conv of both networks, alone, on the dumped input ==
  dump (pooled map and pre-pooling map) for `16,16,13,13`, `7,5,6,9` and `32,8,26,4`;
  synthetic cases (1×1, pad, pool s2 and s1 at the border) against a naive reference for
  four tile sets including `1,1,2,2`.
  **Deviations from the pseudo-code of §10.2**: (1) tile loop order row → col → to → ti
  (2015-zhang): the (to, ti, row, col) order of §10.2 with a store at the last ti would
  require an `out_buf` of the whole map; (2) Tr, Tc = output tile of the conv **before**
  pooling (literal formulas): a pooled conv (k_p, s_p) produces
  P = ⌊(Tr − k_p)/s_p⌋ + 1 pooled rows per tile and advances by P·s_p rows; with stride 1
  the last row of the tile is recomputed by the next one; Tr, Tc ≥ 2; (3) inside a tile,
  MACs in the order (too, tii, i, j, trr, tcc), a mere reordering of exact integer sums.
  `2015-zhang#014.0` not reread (`bin/pdb` missing from the repository).

### [x] T5.3 — YOLO-specific layers
- **Spec**: §10.3 · **Depends on**: T5.2 · **Size**: M
- **Deliverables**: 2×2/1 maxpool (border replication), upsample by address division,
  route by contiguous placement in memory
- **Acceptance**: dumps of layers 11, 19 and 20 identical; no memory copy for the upsample
  or the route (checked by an access counter)
- **Done**: `engine.hpp`: the output of each layer is a view (`InView`, ≤ 2 segments
  `{base, C, H, W, up}`); 2×2/1 maxpool in the output stage of the conv (bounded window =
  replication); upsample = view of the source with `up + 1`; route = concatenation of the
  segments, merged when they are end to end (route 17 = L13 in place). Layers 11 (v2, v3),
  19 and 20 == dumps; `AccessStats` counters: 0 bytes read or written by layers 17, 19 and
  20, bytes written by the network = Σ conv outputs + prepool of L08, and the area
  `R:[0, 86528)` that the manifest reserved for the ×2 copy stays untouched.
  **Deviation from the manifest**: the `out` of the upsample is not written; route 20 is
  read in two segments (L18 in B seen ×2, prepool of L08 at `R+86528`) — conventions.md
  updated.

### [x] T5.4 — Full network and comparison
- **Spec**: §10.5 step 3, §11 · **Depends on**: T5.3 · **Size**: M
- **Deliverables**: `cpp/golden/src/engine.cpp`, `tools/compare_dumps.py` (report per
  layer: number of differing values, max gap)
- **Acceptance**: **0 differing values** on all layers for the 3 images of T4.7, for
  Tiny-YOLOv2 and Tiny-YOLOv3
- **Done**: `Engine::run<Tiles>` (default tiles 16,16,13,13), CLI
  `golden_run run <model> <input.npy> <out>` (`Lxx.npy` + `detections.json`),
  `tools/compare_dumps.py`, `make golden-check`: **0 differing values** on all layers,
  3 images × (Tiny-YOLOv2 VOC, Tiny-YOLOv3 COCO — the T4.7 export), identical detections.
  Also checked in `make test-cpp` (`test_engine.cpp`). ≈ 1.5 s per image.

### [x] T5.5 — C++ post-processing
- **Spec**: §8.1, §8.2, §9.4 · **Depends on**: T5.4 · **Size**: M
- **Deliverables**: `cpp/golden/include/golden/postproc.hpp` (threshold on the logit, LUT,
  decoding, NMS)
- **Acceptance**: boxes == `dumps/heads.json`; code reusable as is in `sw/postproc`
  (without depending on the rest of the golden model)
- **Done**: `postproc.hpp` (header only, STL only, `postproc` namespace): threshold on the
  integer t_o, σ / e^t / e^{d·s} tables (three-stage v2 softmax), decoding, per-class NMS
  with a stable sort; an operation-by-operation copy of `decode_head_int` and
  `filter_and_nms`. Boxes, scores and classes **bit-for-bit equal** to `detections.json`
  (the real file; `heads.json` does not exist) for 3 images × 2 networks, heads read from
  the dumps: the test does not use the engine.

### [x] T5.6 — Roofline explorer
- **Spec**: §10.2 · **Depends on**: T0.5 · **Size**: M
- **Deliverables**: `tools/roofline.py`; resource files of the candidate boards
  `hw/boards/<board>.yaml` (DSP, BRAM, DDR bandwidth, frequency)
- **Acceptance**: for each board, enumerates $(T_m, T_n, T_r, T_c)$, computes BRAM, DSP,
  CTC and attainable performance; recovers the order of magnitude of §10.2 (Tm = Tn = 16 →
  ≈ 68 ms at 200 MHz for Tiny-YOLOv2); roofline plot
- **Notes**: can start as early as M0; it is the input of decision T6.0
- **Done**: `tools/roofline.py` (`make roofline`) → `results/roofline.md` and
  `results/roofline_<board>.png`; boards
  `hw/boards/{pynq-z2,zybo-z7-20,ultra96-v2,kv260}.yaml` (public datasheets, DDR efficiency
  0.7 and clock assumed). Model: cycles ⌈M/Tm⌉⌈N/Tn⌉⌈R/Tr⌉⌈C/Tc⌉K²TrTc, DDR traffic B_in,
  B_w, B_out per tile, time max(compute, transfer) per layer, DSP = TmTn + 4Tm
  (requantization), BRAM18 of the `in_buf`/`out_buf` banks in ping-pong, ≤ 80 % of the
  resources. Order of magnitude recovered: 3.49 GMAC / 256 = 68.1 ms at 200 MHz. Best
  points (Tiny-YOLOv2): KV260 32,24,13,13 → ≈ 32 ms; Ultra96-V2 8,32,13,13 → ≈ 86 ms;
  Zynq-7020 8,16,13,13 → ≈ 198 ms (bounded by the DSPs, not by the DDR).
