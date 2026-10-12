# Measurement protocol (T8.1, T8.2)

This protocol fixes what each figure of [`benchmarks.csv`](benchmarks.csv),
[`mesures.md`](mesures.md) and [`map_stades.md`](map_stades.md) covers, so that the
comparison with the base (§10.4) is made with a known scope. The board commands assume
step 3 of [`hw/boards/kv260/README.md`](../hw/boards/kv260/README.md) is done (overlay
loaded, 0 differences with `run_compare`).

## Measured system

| | |
|---|---|
| Board | Kria KV260 (SOM K26, XCK26-SFVC784-2LV-C), Kria Ubuntu 22.04 |
| Accelerator | IP `yolo_conv` (Vitis HLS), single engine layer by layer, Tm = 32, Tn = 24, Tr = Tc = 13 ([ADR 0003](../docs/adr/0003-choix-carte.md)) |
| PL clock | `pl_clk0` = 200 MHz (check `timing.rpt`: WNS ≥ 0) |
| Network | Tiny-YOLOv2 VOC 416×416, INT8 (per-channel weights, per-layer activations); Tiny-YOLOv3 COCO second |
| MACs | 3.486 GMAC (v2), 2.782 GMAC (v3), convolutions only (`tools/perf_model.py`, `tools/count_macs.py`) |
| Software | `sw/app/yolo_bench`, native build `make sw-board` (`-O2`, Release), `uio` backend, interrupt (not `--poll`) |
| Batch | 1 image; images processed one after the other, without overlap of the stages |

## Timed stages

`std::chrono::steady_clock` on the ARM, one value per image and per stage
(`yolo_bench --times`):

| Stage | Contents | Out of scope |
|---|---|---|
| `pre` | `--images`: file reading, JPEG decoding (stb_image), `stretch` resize (Darknet interpolation), int8 quantization | camera, acquisition |
| `load` | zeroing of the arena and copy of the input into the u-dma-buf (uncached, `O_SYNC`) | loading of the weights (once, at start-up) |
| `acc` | for each conv: writing the AXI-Lite registers, `ap_start`, waiting for the interrupt; sum over the 9 (v2) or 13 (v3) convs | — |
| `post` | heads read in place in the arena, decoding by tables, threshold, per-class NMS | drawing the boxes |

With `--hw-post`, the CSV has five more columns for `yolo_post`: `overflow` (candidates
lost by the selection) and, on the sim backend only, `post_cycles`, `nms_cycles`,
`candidates` and `survivors` (estimate of the C-sim, T11.6).

- **Latency** = `pre + load + acc + post`; mean and **p99** (nearest rank: 990th sorted
  value out of 1,000) per stage and end to end.
- **FPS** = 1,000 / mean latency (sequential throughput; a `pre` / `acc` overlap on two ARM
  cores is a direction of T8.3, not a measurement).
- **GOPS** = 2 × MACs / mean `acc` (accelerator only, as in 2024-zhang).
- **Efficiency** = theoretical cycles / measured cycles = (MACs / (Tm·Tn)) / (`acc` ×
  200 MHz). Per layer: `run_compare --csv` (time per conv) compared with the cycle model.

## Conditions

- **Images**: the first 1,000 of VOC2007 test (order of `ImageSets/Main/test.txt`), JPEGs
  copied onto the board (`--images`); **5 warm-up passes** on the first image, not counted
  (`--warmup 5`).
- CPU: `performance` governor (`sudo cpupower frequency-set -g performance`), process
  pinned to one core (`taskset -c 3`), no other load (no graphical desktop).
- Detection threshold: 0.005 (that of the mAP, `--conf 0.005`): the post-processing is
  measured in the busiest case. The demo (0.25) is faster.
- Output files on the board in `tmpfs` or on the SD card; the JSONL writing is outside the
  stages.

## Power: two scopes

| Scope | Source | What is included |
|---|---|---|
| **Chip, estimated** | `build/vivado/kv260/power.rpt` (`report_power` after implementation, default activity) | PL + PS of the XCK26; the PS8 share is also reported |
| **SOM, measured** | INA260 of the K26 SOM read through hwmon (`yolo_bench --power`), like `platformstats` | 5 V rail of the SOM: XCK26, DDR4, eMMC, regulators; **not** the carrier board (fan, USB, Ethernet, SD card) |

```bash
grep . /sys/class/hwmon/hwmon*/name            # find the INA260 of the SOM (ina260_u14)
P=/sys/class/hwmon/hwmonN/power1_input           # µW
```

`yolo_bench --power $P --idle-s 10` measures 10 s **at idle** (overlay loaded, accelerator
inactive) then during the whole loop; the power under load, the idle power and the
difference are reported. A measurement at the wall socket (wattmeter on the 12 V supply)
can complement it: "whole board" scope, to be stated as such.

Comparison with the base (§10.4): 2024-zhang distinguishes estimated chip (7.8 W) and
measured board (12.3 W); the other works are *unspec.*: only compare identical scopes.

## Resources

`build/vivado/kv260/utilization.rpt` (after implementation): `CLB LUTs`, `DSPs`,
`Block RAM Tile` (BRAM36); full design (IP + interconnects + reset), not the IP alone.

## Commands on the board

```bash
make sw-board
N=$(ls /sys/class/uio | head -1)                 # the one whose name == yolo_conv
head -n 1000 data/VOCdevkit/VOC2007/ImageSets/Main/test.txt \
  | sed 's|^|data/VOCdevkit/VOC2007/JPEGImages/|; s|$|.jpg|' > /tmp/voc1000.txt
mkdir -p build/m8/tiny-yolov2-voc/board

# T8.1: latency, power (1,000 JPEG images, ARM preprocessing)
sudo taskset -c 3 build/sw-board/yolo_bench --uio /dev/$N --model model/tiny-yolov2-voc \
  --images /tmp/voc1000.txt --warmup 5 --power $P --idle-s 10 \
  --times build/m8/tiny-yolov2-voc/board/times_0.csv | tee build/m8/tiny-yolov2-voc/board/bench.txt
# time per layer (3 reference images) and 0-difference check
sudo build/sw-board/run_compare --uio /dev/$N --net tiny-yolov2-voc \
  --csv build/m8/tiny-yolov2-voc/board/layers.csv

# T8.2: FPGA stage of the mAP (int8 inputs from make m8-inputs, 2.6 GB, copied onto the board)
sudo build/sw-board/yolo_bench --uio /dev/$N --model model/tiny-yolov2-voc \
  --inputs build/m8/tiny-yolov2-voc/inputs.bin --ids build/m8/tiny-yolov2-voc/ids.txt \
  --dets build/m8/tiny-yolov2-voc/board/dets_0.jsonl
```

Then, on the PC (files of `board/` brought back):

```bash
make map-stades
python tools/bench_report.py --net tiny-yolov2-voc \
  --layer-csv build/m8/tiny-yolov2-voc/board/layers.csv --vivado build/vivado/kv260 \
  --board-power <W under load> --idle-power <W at idle>
```

## Without the board (current state)

- FPGA stage of the mAP: the same `yolo_bench`, `sim` backend (emulated registers in front
  of the C-sim kernel), `make bench-sim`; it validates the driver, the post-processing and
  bit-exactness on the 4,952 images, not the time.
- Performance: projection of the cycle model (`tools/perf_model.py`, equal cycle for cycle
  to the C-sim counters), marked "projection C-sim" in `benchmarks.csv`. It is a lower
  bound of the accelerator time: II = 1, pipeline depths, DDR latency and ARM control
  ignored.
