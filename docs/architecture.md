# Architecture

## 1. Four stages, a single contract

The project follows the roadmap of §10.5 of the spec. Each stage is the reference of the
next one:

| Stage | Folder | Role | Reference for comparison |
|---|---|---|---|
| 1. Float | `python/yolo/` | network, loss, training, reference mAP | gradcheck (finite differences) |
| 2. Integer | `python/yolo/quant/` | BN folding, int8, M0/shift, LUT | stage 1 (mAP gap) |
| 3. Golden | `cpp/golden/` | tiled engine in C++, like the hardware | stage 2 (**0 gap**) |
| 4. Hardware | `hls/`, `sw/`, `hw/` | accelerator + ARM driver | stage 3 (**0 gap**) |

Stages 2, 3 and 4 must produce **the same integers**, layer by layer. All the verification
rests on this principle.

## 2. Exchange format (`model/<name>/`)

Produced by the Python export (T4.7), read by the golden model (T5.1), the HLS testbench and
the ARM driver.

```
model/tiny-yolov3-coco/        # make export: tiny-yolov2-voc, tiny-yolov3-coco
├── manifest.json              # description of the network (below)
├── weights.bin                # int8 weights, concatenated, aligned on 64 bytes
├── bias.bin                   # int32 biases
├── requant.bin                # int32 M0 per output channel
├── luts.bin                   # σ, e^t, e^{d·s} tables of the heads (uint32)
└── dumps/000001/ … 000003/    # 3 images of VOC2007 test
    ├── input.npy              # quantized preprocessed image (int8, NCHW)
    ├── L00.npy … L23.npy      # integer output of each layer (fused conv: before the pool)
    └── detections.json        # boxes after decoding + NMS (post-processing reference)
```

Excerpt of `manifest.json` (schema: [manifest.schema.json](manifest.schema.json), full
example: `model/example_manifest.json`, semantics: [conventions.md](conventions.md)):

```json
{
  "format_version": 1,
  "network": "tiny-yolov3-voc",
  "classes": 20,
  "input": {"shape": [1, 3, 416, 416], "scale": 0.0078125, "buf": {"buf": "A", "offset": 0}},
  "anchors": [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]],
  "blobs": {"weights.bin": 8707264, "bias.bin": 13376, "requant.bin": 13376},
  "buffers": {"A": 519168, "B": 692224, "H13": 12675, "H26": 50700, "R": 259584, "S": 43264},
  "layers": [
    {"id": 0, "type": "conv", "k": 3, "s": 1, "pad": 1, "cin": 3, "cout": 16, "act": "leaky", "fused_pool": {"layer": 1, "k": 2, "s": 2}, "w_offset": 0, "b_offset": 0, "m0_offset": 0, "shift": 31, "in_scale": 0.0078125, "out_scale": 0.0625, "in": {"buf": "A", "offset": 0}, "out": {"buf": "B", "offset": 0}, "out_shape": [16, 208, 208]},
    {"id": 1, "type": "maxpool", "k": 2, "s": 2, "fused_into": 0, "out_shape": [16, 208, 208]},
    …
    {"id": 8, "type": "conv", "k": 3, "s": 1, "pad": 1, "cin": 128, "cout": 256, "act": "leaky", "fused_pool": {"layer": 9, "k": 2, "s": 2}, "prepool_out": {"buf": "R", "offset": 86528}, "w_offset": 97216, "b_offset": 960, "m0_offset": 960, "shift": 31, "in_scale": 0.0625, "out_scale": 0.0625, "in": {"buf": "A", "offset": 0}, "out": {"buf": "B", "offset": 0}, "out_shape": [256, 13, 13]},
    …
    {"id": 20, "type": "route", "from": [19, 8], "layout": "contiguous", "scale": 0.0625, "out": {"buf": "R", "offset": 0}, "out_shape": [384, 26, 26]}
  ]
}
```

The `*_offset` fields point into the binary blobs. `in`/`out`/`prepool_out` place each
tensor in a DDR buffer (ping-pong between layers, contiguous placement of 19 then 8 for
route 20 of §10.3).

## 3. Hardware engine: layer by layer

Choice of §10.1: **a single engine** rather than a dataflow pipeline. It contains:

- a parameterised conv PE (Tm × Tn MACs per cycle, loop nest of §10.2);
- a fused output stage: bias, M0/shift requantization, leaky 13/128, 2×2 maxpool with
  stride 2 or 1;
- upsample and route **by addressing**, without copies (§10.3);
- an AXI-Lite controller that receives the layer parameters from the ARM driver.

Decoding and NMS run on the ARM first (§10.3), with the code of `cpp/golden` reused in
`sw/postproc`.

Board: Kria KV260 ([ADR 0003](adr/0003-choix-carte.md)). The tile sizes remain parameters
(`hls/configs/<board>.tcl` → `-DACC_TM=…`), set by the roofline tool (T5.6).

### `yolo_conv` kernel (M6)

One call = one conv (+ fused maxpool), same tile order as `golden::conv_layer`
(row → col → to → ti), hence the same integers. The tile (Tr = Tc = 13, or 14 for the
convs followed by a stride-2 maxpool) and the folding of L00 (lanes = channel × kernel row)
are chosen per layer by the driver (M10, T10.4). With `n_calls > 0`, a single call chains
all the convs from a table of descriptors in DDR (sequencer, T10.7).

| File | Role |
|---|---|
| `hls/kernels/accel_config.hpp` | compile-time configuration: tiles, bytes per port word, trim, requant ×RQ, folding, tile of the pooled convs — also read by the driver and `tools/perf_model.py` |
| `hls/kernels/layer_desc.hpp` | s_axilite registers (`LayerDesc`, 31 words): shape, 2 input segments `{offset, c, h, w, up}`, output and parameter offsets, tile and folding of the layer — plain C++, shared with the driver |
| `hls/kernels/weight_layout.hpp` | weights in tile order: one contiguous block per (to, ti), reordered once by the driver (T10.1) |
| `hls/kernels/conv_pe.cpp` | top, sequencer, loaders in 64-bit words row by row, Tm × Tn PE (`PIPELINE II=1`), `in_buf`/`w_buf` ping-pong on ti and `out_buf` on the tiles |
| `hls/kernels/output_stage.hpp` | bias + requantization (8 channels per cycle) + leaky + saturation, 2×2 s2/s1 maxpool, pre-pooling map, writes in words |
| `sw/driver/program.hpp` | contiguous DDR arena (buffers of the manifest end to end, plus a read margin), per-layer views, one `LayerDesc` per conv, reordered weights, sequencer table |

Interfaces: `m_axi` `gmem_in` / `gmem_out` (same activation arena, two bundles to overlap
load and store), `gmem_w` (weights in tile order), in 64-bit words; `gmem_p` (bias.bin
then requant.bin, and the descriptor table); addresses = byte indices into these arrays.
Upsample and route are only input segments of the next conv: no kernel call.

Verification: `hls/tb/tb_conv.cpp` (each conv alone) and `hls/tb/tb_net.cpp` (network
through the driver, each layer checked right after its call, before the A/B ping-pong
buffers are overwritten). C-sim with g++ (`make csim-gcc`, open-source `ap_int` headers)
or Vitis (`make csim`); synthesis, co-sim, export: `make hls-synth | hls-cosim |
hls-export`, report `make hls-report` → `results/hls_report.md`. The shift-based PE
(`ACC_WMODE_POW2`, T9.2.4) is synthesized separately with `make hls-synth-pow2`
(`hls/proj_<board>_synth_pow2/`); figure T13.24 reads the csynth reports of both variants
and of the streaming design.

### Streaming architecture (M9.4, T10.8, T10.9)

Second family of §10.1, next to the single engine: `yolo_stream` (hls/stream/) chains one
hardware stage per conv of Tiny-YOLOv2 under DATAFLOW, with the PE × SIMD folding of the
`tools/stream_model.py` plan (KV260). The weights of the on-chip stages are a ROM generated
from the export (`tools/gen_stream_rom.py`, `-DSTREAM_ROM`); L12 and L13 keep their input
map and read their weights from DDR once per image (outer PE order, L12 → L13 in CHW).
Input and head in one-byte AXI-Stream (TLAST), carried by an AXI DMA in direct mode.

| File | Role |
|---|---|
| `hls/stream/conv_stage.hpp` | conv stage: line buffer or whole map (FRAME, outer PE), weights in ROM `[og][ig][i][j][pe·SIMD + s]` or through a pointer, maxpool fused in the stream |
| `hls/stream/yolo_stream.hpp` | stage tables (PE, SIMD, FRAME, CHW, FIFO depths == `stream_model.fifo_depths`) |
| `hls/stream/stream_desc.hpp` | `StreamDesc` registers (45 words), shared with the driver |
| `sw/driver/stream_driver.hpp` | `StreamAccelerator`: HWC input, ap_start, S2MM then MM2S, wait for the end of S2MM, HWC → CHW head |
| `sw/driver/stream_regmap.hpp` | offsets of `yolo_stream` (to be checked against the Vitis header) and of the AXI DMA (PG021) |

Verification: `tb_stream` and `tb_stream_rom` (each stage == dumps, cycles == model, a
single TLAST), `yolo_bench --engine stream` with the sim backend (emulated DMA,
detections == dumps). Synthesis: `make hls-synth-stream | hls-cosim-stream |
hls-export-stream`, block design `make vivado-build ENGINE=stream` (overlay
`pl_stream.dtsi`), not run (Vitis missing).

### SoC integration (M7)

Stack on the KV260: Kria Ubuntu, overlay loaded by `xmutil loadapp yolo`, kernel registers
through `/dev/uioN` (`generic-uio`), contiguous memory through `/dev/udmabuf0`
(u-dma-buf).

```
PS (A53) ── HPM0_FPD ─► s_axi_control @0xA000_0000 ─┐
          ◄─ pl_ps_irq0 ─ interrupt ─────────────────┤ yolo_conv (200 MHz, pl_clk0)
DDR ◄─ HP0_FPD ◄─ gmem_in, gmem_out ─────────────────┤
    ◄─ HP1_FPD ◄─ gmem_w, gmem_p ────────────────────┘
```

| File | Role |
|---|---|
| `hw/boards/kv260/build.tcl` | block design, bitstream, `.xsa`, reports; fails if timing is not met (`make vivado-build`) |
| `hw/boards/kv260/pl.dtsi`, `firmware.sh` | overlay (UIO + IRQ, 32 MB u-dma-buf, clock) and `xmutil` package (`make fpga-firmware`) |
| `sw/driver/regmap.hpp` | s_axilite offsets; aggregated `LayerDesc` = 31 words (`qmax` in T9.3, `tr`, `tc`, `fold` in T10.4, `wbits` in T10.10), then `DESCS` and `N_CALLS` (sequencer, T10.7); checked against the Vitis header (`make check-regmap`) |
| `sw/driver/device.hpp` | hardware access: `uio` (board; `--cached`: cached u-dma-buf and per-range syncs, T10.6) or `sim` (PC: emulated registers → C-sim kernel) |
| `sw/driver/accel_driver.hpp` | `Accelerator`: one contiguous buffer [arena(s) \| weights \| parameters \| descriptors], bounds of each `LayerDesc` checked; one conv per `run_layer`, or the whole pass with `run_all` (one IRQ); two arenas for the inter-image pipeline (T10.5) |
| `sw/postproc/heads.hpp` | heads read in place in the arena → `postproc.hpp` (decoding + NMS on the ARM) |
| `sw/app/run_compare.cpp` | each layer in DDR == dump, then detections == `detections.json`; `--layer N`: an isolated conv |
| `sw/app/yolo_app.cpp` | image → boxes demo, preprocessing / accelerator / post-processing times; `--pipeline` |
| `sw/app/yolo_bench.cpp` | per-stage measurements (pre, load, acc, post, ARM time) on a set of images; `--pipeline`: two threads, two arenas; `--engine stream`: streaming architecture (T10.9) |

The `sim` backend goes through the same path as the board (register encoding, physical
addresses, ap_start / ap_done): `make sw-sim` validates everything except the hardware
itself. Board procedure: [hw/boards/kv260/README.md](../hw/boards/kv260/README.md).

### Measurements (M8)

| File | Role |
|---|---|
| `sw/app/yolo_bench.cpp` | set of images (`--images` JPEG or `--inputs` int8): time per stage and per image (pre, load, acc, post), mean and p99, JSONL detections at the mAP threshold, INA260 power of the SOM (`--power`) |
| `tools/perf_model.py` | cycle model of the kernel, equal to the C-sim counters (`make perf-model`); tiling (`tile_grid`); costed optimisation scenarios; `--manifest`: cycles of any export (pruned, mixed precision) |
| `tools/mixed_precision.py` | per-layer mixed precision (T10.10): sensitivity, greedy front, full mAP of the selected configuration → `results/precision_mixte.md` |
| `tools/prune.py`, `tools/prune_study.sh` | structured pruning by filter norm (T10.11), then fine-tuning, calibration, mAP and cycles per rate → `results/elagage.md` |
| `tools/make_ci_model.py` | synthetic export of `model/` for the CI (T10.13), without VOC or Darknet weights |
| `tools/make_inputs.py` | int8 inputs of VOC2007 test (preprocessing of the integer model), read by the board instead of the JPEGs |
| `tools/bench_sim.sh` | FPGA stage in C-sim on the PC: packets spread over the cores, resume (`make bench-sim`) |
| `tools/map_stades.py` | float / integer / FPGA mAP, image-by-image equality → `results/map_stades.md` |
| `tools/bench_report.py` | `results/benchmarks.csv` (base + this work), `results/mesures.md`; reads `times*.csv`, `utilization.rpt`, `power.rpt` |
| `tools/figures/` | M13 figures (`make figures`): one module per family (models, hardware, results, project, maths, networks), read from the sources above → `results/figures/`, gallery `results/figures.md` |

Measurement scopes: [results/protocole.md](../results/protocole.md); comparison and
directions: [results/rapport.md](../results/rapport.md).

## 4. C++ golden model (`cpp/golden/`, M5)

The golden model runs the network like the hardware engine and serves as the HLS testbench:

| Header | Role |
|---|---|
| `model.hpp`, `json.hpp`, `npy.hpp` | reading of the manifest, the blobs and the dumps (T5.1) |
| `conv.hpp` | tiled conv PE `conv_layer<Tiles<Tm, Tn, Tr, Tc>>`: fixed-size buffers, output stage bias + requantization + leaky + fused maxpool; header only, no allocation, shared with `hls/` (T5.2) |
| `engine.hpp` | full network on a "DDR" arena per manifest buffer; each layer has an output view (`InView`, ≤ 2 segments): upsample = address division, route = segments end to end, no copies; DDR access counters (T5.3, T5.4) |
| `postproc.hpp` | threshold on the logit, LUT, decoding, NMS; STL only, reusable in `sw/postproc` (T5.5) |

`golden_run run <model> <input.npy> <out>` writes `Lxx.npy` and `detections.json`;
`make golden-check` compares them with the Python dumps using `tools/compare_dumps.py`. The
deliberate differences from the pseudo-code of §10.2 (order of the tile loops, tiles before
pooling) are described in [T5.2](tasks/M5-golden-cpp.md).

## 5. Verification flow

```
pytest (gradcheck)          ─► stage 1 correct
float vs integer mAP        ─► measured quantization loss (§9)
tools/compare_dumps.py      ─► golden == Python dumps, layer by layer (make golden-check)
HLS C-sim / co-sim          ─► kernel == golden == dumps (make csim-gcc, make hls-cosim)
run_compare (sim then uio)  ─► DDR output == golden == dumps (make sw-sim, then on the board)
yolo_bench + map_stades     ─► mAP at the three stages (§11), FPGA == integer image by image (M8)
bench_report                ─► benchmarks.csv: board measurements, or perf_model projection (M8)
```

## 6. Live webcam demo (`tools/webcam.py`, M20)

On the PC, outside the four stages: it runs the float or integer NumPy model (stages 1
and 2) on the camera image. Three threads share only the newest frame (capture, inference,
display), so the display follows the camera while an inference takes ≈ 200 ms. Usage and
internals: [webcam.md](webcam.md); tasks and measurements: [M20](tasks/M20-webcam.md). On
the KV260, the FPGA takes the place of the inference thread (T20.9).
