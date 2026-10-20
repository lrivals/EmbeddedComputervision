# Faults and their detection (M25)

The sections below come from the HLS kernel in C-sim (T25.3, T25.7), 2026-10-20. The
sensitivity campaign on the integer model (T25.1, `tools/fault_campaign.py`, mAP against
the BER on 500 VOC images) has only been smoke-run; its tables go here when it is run
(`tools/m25.sh`).

## T25.3: ABFT in the HLS kernel (C-sim)

The checksum of T25.2 is now in `yolo_conv` (`hls/kernels/conv_pe.cpp`, `output_stage.hpp`),
behind `ACC_ABFT` (default 1).

- **Checksum lane**: `compute` computes one more output lane, Σ_tii cs_w·x, in the same
  II = 1 loop as the Tm filters. It runs at the first `to` of each (row, col) tile, with a
  ping-pong buffer over the (row, col) tiles.
- **Comparison**: in the requantization pass, `store_tile` adds the tm_n lanes + bias to a
  running sum over the `to` tiles. At the last `to`, it compares the sum with the lane + Σ_f
  b_f, modulo 2³².
- **Parameters**: the checksum weights and Σ_f b_f reach the kernel through `prm`. The two
  words before the M0s of each conv are written by `driver::build`, so `LayerDesc` is
  unchanged.
- **Status**: the failing pixels of a call go to the new output register `abft_err`
  (`regmap::ABFT_ERR`, offset to confirm with `make check-regmap`). The board driver adds
  them up in `Accelerator::abft_errors()`, and the `sim` backend emulates them.

**Without fault**, every bit-exact test passes unchanged, with 0 ABFT error:

- `tb_conv`: 103 layers of the four exports;
- `tb_net`: 158 layer outputs;
- the packed, mixed-precision and pow2 variants;
- `make sw-sim`: 13 / 13.

The C-sim cycle counts are identical.

**Projected cost** (`python tools/perf_model.py --abft`; synthesis numbers deferred):

| Network | Widest checksum weight | Extra DSP | Kernel DSP | Cycles not hidden | Buffers | Weight registers |
|---|---|---|---|---|---|---|
| tiny-yolov2-voc | 16 bits | +24 | 800 (+3.0 %) | 0 | 2,352 B | 13,824 bits |
| tiny-yolov3-coco | 17 bits | +24 | 800 (+3.0 %) | 0 | 2,352 B | 13,824 bits |
| tiny-yolov4-coco | 16 bits | +24 | 800 (+3.0 %) | 0 | 2,352 B | 13,824 bits |

- **DSP**: the checksum weights fit the 27-bit port of a DSP48E2, so the lane costs one DSP
  per input lane (Tn = 24).
- **Cycles**: its weights (n·kh·k words per input-channel tile) load on `gmem_p` while
  `in_buf` and `w_buf` load on their own ports, and they are shorter on every layer.

## T25.7: faults in the HLS buffers (C-sim)

`hls/tb/tb_fault` flips one bit in an on-chip buffer of `yolo_conv`, through the C-sim fault
hook `hls/kernels/fault_hook.hpp` (absent from the synthesis). One flip per kernel run, on a
location the tile uses, drawn at random:

| Buffer | Holds | Bits |
|---|---|---|
| in | `in_buf`, int8 input tile of one input-channel tile | 8 |
| w | `w_buf`, int8 weights of one input-channel tile | 8 |
| out | `out_buf`, int32 accumulators of the tile, before the bias | 32 |
| q | requantized int8 tile, read by the fused max-pool (the kernel has no separate line buffer) | 8 |
| cs | ABFT checksum lane | 32 |

### Mapping buffer address → tensor index (checked)

`faults.hls_buffer_reference` (Python) maps a buffer address to the tensor element through
the lane mapping of `load_input` and `reorder_weights`, folded first conv included. It
recomputes the outputs owned by the tile. `test_hls_buffer_faults_match_python_reference`
compares it with the C-sim:

- **Run**: 195 flips (`tb_fault --flips 3 --dump`), on layers 0, 2, 8, 10, 12, 13, 15 and 19
  of Tiny-YOLOv2 and Tiny-YOLOv3. These include the folded L0, a 2×2 stride-1 pool, routes
  and an upsample.
- **Result**: the faulty outputs are equal to the byte, and the output outside the tile's
  own region is the fault-free dump.
- **Sensitivity check**: shifting the address by one column breaks 96 of 104 comparisons.

### Campaign

`tb_fault --net tiny-yolov2-voc --net tiny-yolov3-coco --flips 20 --seed 1`: every conv of
both networks, 20 flips per buffer and per conv, 2,200 flips, image 000001;
`python tools/hls_faults.py`.

| Network | Buffer | Flips | Output changed | Detected (ABFT) | Changed and detected | Silent corruption | Detected, output unchanged |
|---|---|---|---|---|---|---|---|
| tiny-yolov2-voc | in | 180 | 86 % | 78 % | 79 % | 18 % | 9 % |
| tiny-yolov2-voc | w | 180 | 59 % | 96 % | 100 % | 0 % | 37 % |
| tiny-yolov2-voc | out | 180 | 44 % | 100 % | 100 % | 0 % | 56 % |
| tiny-yolov2-voc | q | 180 | 67 % | 0 % | 0 % | 67 % | 0 % |
| tiny-yolov2-voc | cs | 180 | 0 % | 100 % | — | 0 % | 100 % |
| tiny-yolov3-coco | in | 260 | 92 % | 85 % | 85 % | 13 % | 6 % |
| tiny-yolov3-coco | w | 260 | 77 % | 98 % | 100 % | 0 % | 21 % |
| tiny-yolov3-coco | out | 260 | 49 % | 100 % | 100 % | 0 % | 51 % |
| tiny-yolov3-coco | q | 260 | 81 % | 0 % | 0 % | 81 % | 0 % |
| tiny-yolov3-coco | cs | 260 | 0 % | 100 % | — | 0 % | 100 % |

"Detected, output unchanged" covers two cases:

- the flip changed the accumulator, but the requantization or the pooling masked it;
- the flip hit the checksum lane itself, which leaves the outputs alone.

Reading:

- **Accumulators and weights are covered**: every flip of `out_buf` and `w_buf` that changes
  the output is detected, with no silent corruption.
- **The requantized tile is a blind spot**: 67 to 81 % of the flips of `q` change the output,
  and none is detected, because the check is done before the requantization, as T25.2
  states. This buffer, and the DDR writes after it, need another protection: parity or
  duplication of the output stage (T25.4).
- **Input tiles are covered except on narrow layers**:
  - A flip in `in_buf` at a `to > 0` tile is always detected (271 / 271). The checksum lane
    of the (row, col) tile was computed from the clean tile at `to = 0`.
  - At `to = 0`, the lane reads the same corrupted tile, but it sums all output channels
    while the tile feeds only Tm of them. The flip is still caught when the layer has more
    than Tm = 32 outputs.
  - The flips missed (67, i.e. 15 % of the input flips) all fall in L0 and L2 of both
    networks (16 and 32 output channels: a single `to` tile). There the checksum identity
    holds with the corrupted input on both sides.
  - Detecting them would take a parity on `in_buf`, or a lane computed from a second read.
- **Checksum-lane flips** are always flagged and never change the output. They are false
  alarms, at a rate that scales with the lane's share of the buffers: 2.4 KB of 2 × (B_in +
  B_w + B_out).

### Masking by tile reloading

The 54 weight and input flips of the dumped set changed the output. The table compares each
one with the same element corrupted **persistently**, as a fault in DDR would be (T25.0 sites
`weight` and `act`):

| Buffer | Flips compared | Output bytes changed, buffer flip (mean) | Output bytes changed, persistent fault (mean) | Share masked by tile reloading |
|---|---|---|---|---|
| w | 26 | 18.7 | 2,226.1 | 99 % |
| in | 28 | 21.1 | 295.4 | 93 % |

Per flip, a buffer fault does at most 1 % to 7 % of the damage of the same fault in DDR.

- **Why**: an on-chip buffer is reloaded for the next tile, so a flip there reaches only
  the outputs of one tile and Tm channels.
- **Consequence**: a DDR or weight-BRAM fault is the one to protect first (T25.4).

Configuration-memory upsets are out of reach in C-sim:
[deferred/M25](../docs/tasks/deferred/M25-dependability.md).
