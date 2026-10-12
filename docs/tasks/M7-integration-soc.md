# M7 — SoC integration

Goal: the accelerator runs on the board, driven by the ARM, and produces in DDR exactly the
same integers as the golden model.

### [ ] T7.1 — Vivado block design
- **Spec**: §10.1 · **Depends on**: T6.5 · **Size**: M
- **Deliverables**: `hw/boards/<board>/build.tcl` (PS, HLS IP, AXI interconnects,
  clocks, reset); bitstream + `.xsa`
- **Acceptance**: bitstream generated from the tcl alone (reproducible); timing met
- **Done**: `hw/boards/kv260/build.tcl` (Vivado batch): PS with the board preset,
  `yolo_conv` IP from `make hls-export`, `s_axi_control` on HPM0_FPD at 0xA000_0000,
  `gmem_in`/`gmem_out` → HP0, `gmem_w`/`gmem_p` → HP1 (smartconnects), `pl_clk0` 200 MHz,
  `proc_sys_reset`, interrupt → `pl_ps_irq0`; bitstream, `.xsa`, timing / utilisation /
  power reports; fails if WNS, TNS or WHS < 0. `pl.dtsi` (overlay: `generic-uio` + IRQ,
  32 MB u-dma-buf, clock), `shell.json`, `firmware.sh` (`make fpga-firmware`), board
  procedure in `hw/boards/kv260/README.md`. `make vivado-build`.
- **Remaining**: everything that needs Vivado: bitstream generation, timing met at
  200 MHz. The script has never been run (Vivado missing); `pl.dtsi` not compiled (dtc
  missing).

### [ ] T7.2 — ARM driver
- **Spec**: §10.1, §10.3 · **Depends on**: T7.1, T5.1 · **Size**: L
- **Deliverables**: `sw/driver/`: loading of the manifest and of the blobs into contiguous
  memory (CMA / XRT / PYNQ depending on the board), programming of the AXI-Lite registers,
  layer scheduling, wait for completion
- **Acceptance**: an isolated layer run on the board == golden dump
- **Notes**: the buffer allocation must respect the contiguous placement of the route
- **Done**: selected stack: Kria Ubuntu + UIO + u-dma-buf. `sw/driver/`: `regmap.hpp`
  (s_axilite offsets expected from Vitis HLS, `LayerDesc` aggregated into 26 words; check
  against the generated header: `make check-regmap`, `tools/check_regmap.py`),
  `device.hpp` (HAL: contiguous buffer, registers, wait), `uio_device.cpp` (board:
  `/dev/uioN` + IRQ, `/dev/udmabufN` in O_SYNC), `sim_device.cpp` (PC: emulated registers,
  translation of dummy physical addresses > 4 GB, call of the C-sim kernel at ap_start),
  `accel_driver.{hpp,cpp}` (`Accelerator`: one buffer [arena from `driver::build` |
  weights | parameters], bounds check of each `LayerDesc`, timed `run_layer`). On the PC:
  `run_compare --layer N` (arena prefilled with the dumps of the layers < N) == dump for
  L00 (v2, v3), L13 (v2) and L21 (v3, route in 2 segments, one of them an upsample).
  `make sw-sim`, `make sw-board` (native build, uio backend).
- **Remaining**: run on the KV260.

### [ ] T7.3 — Post-processing on the ARM
- **Spec**: §10.3 (decoding + NMS on the ARM, as 2025-kim and 2024-yan do) ·
  **Depends on**: T5.5, T7.2 · **Size**: S
- **Deliverables**: `sw/postproc/` (reuses `postproc.hpp`), `sw/app/` (image or camera →
  boxes)
- **Acceptance**: boxes == `dumps/heads.json`; post-processing time measured
- **Done**: `sw/postproc/heads.{hpp,cpp}`: heads read in place in the arena (views of the
  driver), then `postproc::postprocess` (`postproc.hpp` as is);
  `golden/detections_io.hpp` (writing shared with `golden_run`, reading and exact
  equality). `sw/app/yolo_app`: `--input x.npy` or `--image` (stb_image, Darknet `stretch`
  like `resize_darknet`, quantization `quantize_input`), `--draw` (boxes on the image),
  preprocessing / accelerator / post-processing times. On the PC: detections ==
  `detections.json` for the 3 images of both networks; post-processing < 0.02 ms on x86.
- **Remaining**: post-processing time measured on the A53.

### [ ] T7.4 — End to end on the board
- **Spec**: §10.5 step 4, §11 · **Depends on**: T7.3 · **Size**: M
- **Deliverables**: `sw/app/run_compare` (runs and compares each layer with the golden
  model)
- **Acceptance**: **0 differences** on all the layers for the 3 reference images;
  detection demo on a real image
- **Done**: `sw/app/run_compare`: full pass, each layer in DDR compared with the dump right
  after its conv (same path as `hls/tb/tb_net.cpp`), then detections; `--csv` of the times
  per layer (M8). Sim backend: **87 layer outputs, 0 differences**, identical detections,
  Tiny-YOLOv2 VOC and Tiny-YOLOv3 COCO, 3 images. Demo `yolo_app --image 000004.jpg`:
  7 cars detected.
- **Remaining**: 0 differences on the KV260 (`hw/boards/kv260/README.md`, step 3); demo on
  the board.
