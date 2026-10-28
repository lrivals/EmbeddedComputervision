# Faults and their detection (M25)

The first section is the sensitivity campaign on the integer model (T25.1), 2026-10-27.
The others come from the HLS kernel in C-sim (T25.3, T25.7), 2026-10-20.
T25.4 (selective protection), T25.6 (person criterion) and T25.9 (faults in the failure monitor) were added on 2026-10-28.

## T25.1: sensitivity of the integer model

`tools/m25.sh ber`, `layer`, `bit`, then `report` (`tools/fault_campaign.py`, faults of
`yolo.quant.faults`, T25.0): Tiny-YOLOv2 VOC integer, 500 images of VOC2007 test (tier M,
ranking only), seeds 0, 1, 2 (mean ± std), 393 runs (73 + 217 + 103), 9.1 h on 10
processes (CPU shared with other runs). `weight` / `bias` faults are flipped once in the stored model; `act` (int8
layer inputs) and `acc` (int32 accumulators) are drawn again for each image. Figure:
`build/figures/resultats/faults.png` (T13.58).

Fault-free mAP: **57.73** (500 images).

| Site | BER 1e-09 | BER 1e-08 | BER 1e-07 | BER 1e-06 | BER 1e-05 | BER 1e-04 |
|---|---|---|---|---|---|---|
| weight | 57.73 ± 0.00 | 57.82 ± 0.17 | 57.51 ± 0.19 | 57.41 ± 0.22 | 57.54 ± 0.46 | 57.60 ± 1.16 |
| bias | 57.73 ± 0.00 | 57.73 ± 0.00 | 57.73 ± 0.00 | 57.73 ± 0.00 | 56.57 ± 2.01 | 50.77 ± 6.50 |
| act | 57.73 ± 0.00 | 57.69 ± 0.09 | 57.50 ± 0.20 | 57.88 ± 0.12 | 57.02 ± 0.74 | 51.04 ± 0.20 |
| acc | 57.75 ± 0.03 | 57.67 ± 0.12 | 57.74 ± 0.31 | 57.49 ± 0.30 | 50.32 ± 2.00 | 21.25 ± 0.72 |

| Conv | weight 1e-07 | weight 1e-05 | bias 1e-07 | bias 1e-05 | act 1e-07 | act 1e-05 | acc 1e-07 | acc 1e-05 |
|---|---|---|---|---|---|---|---|---|
| L00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.14 | +0.08 | +0.25 | -4.83 |
| L02 | +0.00 | +0.00 | +0.00 | +0.00 | -0.04 | -0.04 | -0.13 | -1.19 |
| L04 | +0.00 | -0.18 | +0.00 | +0.00 | +0.04 | -0.12 | +0.04 | -0.89 |
| L06 | +0.00 | -0.28 | +0.00 | +0.00 | -0.01 | -0.38 | -0.04 | -0.39 |
| L08 | +0.00 | -0.15 | +0.00 | +0.00 | -0.02 | -0.15 | -0.03 | -0.17 |
| L10 | -0.03 | -0.08 | +0.00 | +0.00 | -0.00 | -0.01 | +0.02 | +0.15 |
| L12 | +0.01 | -0.08 | +0.00 | +0.00 | +0.01 | +0.11 | +0.01 | -0.28 |
| L13 | +0.01 | +0.06 | +0.00 | +0.00 | +0.00 | +0.01 | +0.03 | +0.24 |
| L14 | +0.00 | +0.02 | +0.00 | +0.00 | -0.00 | -0.22 | -0.00 | -0.35 |

mAP change in points (negative: loss). Ranking (mean loss): L00 (0.58), L02 (0.18), L04 (0.14), L06 (0.14), L14 (0.07), L08 (0.06), L12 (0.03), L10 (-0.01), L13 (-0.04)

| Bit | weight | bias | act | acc |
|---|---|---|---|---|
| 0 | -0.11 | -0.04 | -0.20 | +0.00 |
| 1 | -0.20 |  | -0.11 |  |
| 2 | -0.04 |  | -0.18 |  |
| 3 | -0.22 |  | -0.03 |  |
| 4 | -0.16 | +0.06 | -0.23 | -0.12 |
| 5 | -0.12 |  | -0.03 |  |
| 6 | -0.14 |  | -0.05 |  |
| 7 | +0.03 |  | -1.31 |  |
| 8 |  | -0.03 |  | -0.00 |
| 12 |  | -2.79 |  | +0.06 |
| 16 |  | -54.66 |  | -0.36 |
| 20 |  | -57.73 |  | -0.50 |
| 24 |  | -57.73 |  | -0.56 |
| 28 |  | -57.73 |  | -0.56 |
| 31 |  | -57.69 |  | -1.42 |

mAP change in points for one flip at that bit (per image for act / acc, per model for weight / bias).

| Site | Images changed | Flagged by ABFT | Changed and flagged | Silent (changed, not flagged) |
|---|---|---|---|---|
| weight | 69.2 % | 69.8 % | 100.0 % | 0.0 % |
| bias | 31.8 % | 32.3 % | 100.0 % | 0.0 % |
| act | 71.0 % | 0.0 % | 0.0 % | 71.0 % |
| acc | 74.2 % | 85.1 % | 100.0 % | 0.0 % |

- **BER**: up to 1e-6, no site moves the mAP by more than 0.35 points, which is the seed
  spread. At 1e-4, accumulator faults take the mAP from 57.7 to 21.3; activation and bias
  faults cost about 7 points (bias ± 6.5: it depends on which bits are hit). Weight faults
  cost nothing measurable even at 1e-4: an int8 weight flip moves one product of a sum of
  thousands.
- **Bit position**: the bias is the weak point. A single flip at bit 16 or higher of an
  int32 bias kills the model (−54.7 to −57.7 points, no detections left); bit 12 costs 2.8.
  One accumulator flip per image costs at most 1.4 points, even on the sign (bit 31),
  because it hits one pixel of one image. For an int8 activation, the sign bit (7) costs
  1.3 points and the low bits stay in the noise.
- **Layers**: the first convs are the most sensitive (L00 −4.8 points for accumulator
  faults at 1e-5, then L02 and L04); from L08 on, the losses are below the seed spread.
- **ABFT** (T25.2): every weight, bias or accumulator fault that changes a result is
  flagged (0 % silent). Activation faults are never flagged (71 % of the images changed
  silently): the checksum is computed on the corrupted input, so it checks the
  computation, not its data. Activation buffers need their own protection (parity or ECC,
  T25.7).

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

## T25.4: selective protection

`tools/m25.sh protect` (`tools/fault_campaign.py run --grid protect`), 2026-10-28.

**Setup:**
- integer Tiny-YOLOv2 VOC on the 500-image T25.1 subset, fault-free mAP 57.73;
- at each BER, **every site is faulty at once**: weights and biases (persistent),
  activations and accumulators (transient), 3 seeds;
- a protection scheme removes from the injection what it masks, so it is modelled as
  perfect: ECC corrects, and ABFT detects then reloads or recomputes.

**Schemes:** the sensitive layers are the first 3 convs (L00, L02, L04), the top of the
T25.1 and T25.6 rankings.
- ECC on every weight and bias word;
- ABFT (T25.2) on every conv or on the first 3 only (`-sel`). It sees weight, bias and
  accumulator faults, never activation faults (T25.1);
- duplication of the first 3 convs (`dup-sel`), which covers their activations and
  accumulators.

| Scheme | Faulty sites | BER 1e-05 | BER 1e-04 |
|---|---|---|---|
| none | weight, bias, act, acc | 49.12 ± 1.94 | 14.22 ± 2.02 |
| ecc | act, acc | 50.17 ± 1.33 | 17.53 ± 0.81 |
| abft-sel | weight (not the first convs), bias (not the first convs), act, acc (not the first convs) | 51.68 ± 8.52 | 35.58 ± 15.84 |
| abft | act | 57.02 ± 0.74 | 51.04 ± 0.20 |
| dup-sel | weight, bias, act (not the first convs), acc (not the first convs) | 55.40 ± 1.50 | 41.29 ± 5.60 |
| ecc+dup-sel | act (not the first convs), acc (not the first convs) | 56.49 ± 0.24 | 48.59 ± 1.38 |
| abft+dup-sel | act (not the first convs) | 57.47 ± 0.14 | 54.54 ± 0.32 |

mAP (mean ± std over 3 seeds) with every site faulty at once at that BER, minus the masked
sites.

**Costs** (`tools/perf_model.py`, Tiny-YOLOv2, kernel Tm = 32, Tn = 24, 800 DSP; exact
figures wait for Vitis):

| Scheme | DSP | On-chip memory | Cycles |
|---|---|---|---|
| ABFT (every conv) | +24 (+3 %): one checksum lane of Tn products | 2,352 B of checksum lanes and sums, 13,824 register bits of checksum weights | 0 exposed (`abft_cost`: the lane's weights load under the longer of `in_buf` / `w_buf`); a detection costs a reload and recompute of the tile |
| ECC on the weights | 0 | the weights are not in BRAM: they stream from DDR into `w_a` / `w_b`, partitioned into 768 banks of 9 int8 (`conv_pe.cpp`). SECDED per int8 is (13,8), **+62.5 % bits** on those banks, plus the encoders and decoders on 768 read ports. On DDR, SECDED (72,64) is **+12.5 %**. Biases: (39,32), +22 % | 0 if the decoder fits in the II = 1 loop |
| Duplication of L00-L04 | 0 if in time (the same PE twice), ×2 the PE DSP if in space | none in time | **+28.1 %** in time (2,089,436 of the 7,443,739 kernel cycles are L00-L04) |
| ABFT + duplication of L00-L04 | +24 | as ABFT | +28.1 % |

**Findings**:
- **ECC alone barely helps** (+1.1 points at 1e-5, +3.3 at 1e-4). Most of the damage
  of a full-site fault load comes from the activations and accumulators, which ECC does
  not cover.
- **ABFT on every conv is the best single scheme**: 57.02 at 1e-5, within 0.7 points of
  the fault-free mAP, and 51.04 at 1e-4. It covers weights, biases and accumulators for
  +3 % DSP and no exposed cycle.
- **ABFT on the first 3 convs only is unstable** (± 8.5 and ± 15.8 points). One seed
  out of three falls to 41.9 at 1e-5. The persistent faults of the later convs remain:
  a high bias bit is enough (T25.1, T25.6). So ABFT has to be on every conv: its cost
  does not depend on how many convs use it, since the lane is in the shared PE.
- **Duplicating the first 3 convs is worth its cycles.** Alone, it raises 1e-4 from
  14.2 to 41.3: the transient faults of the early layers weigh most. Adding it to ABFT
  closes the activation gap: **57.47 at 1e-5 (−0.26) and 54.54 at 1e-4 (−3.2)**.
- **Recommendation: ABFT on every conv, plus the duplication of L00-L04**, at +24 DSP
  and +28 % kernel cycles. ECC brings nothing more once ABFT is there. If the 28 % do not
  fit the frame budget (T30), keep ABFT alone: −0.7 points at 1e-5.
- **Limit of the model**: at BER 1e-4, ABFT flags 100 % of the images, so every image
  would recompute tiles. The masked-fault model hides that time cost, which is real. The
  recompute rate at the BER expected in the field is left to measure.

Exact BRAM, DSP and LUT of these schemes: [deferred/M25](../docs/tasks/deferred/M25-dependability.md).

## T25.6: critical errors for a person detector

`tools/m25.sh person` (`tools/fault_campaign.py run --criterion person`), 2026-10-28.

**Setup:**
- integer Tiny-YOLOv2 VOC on the 500-image T25.1 subset of VOC2007 test, with the same
  grids as T25.1 (layer grid with seeds 0-1, bit grid with seeds 0-2);
- 198 of the 500 images hold a person at score ≥ 0.25 without fault (the operating
  threshold);
- fault-free mAP 57.73.

CrowdHuman, the set of the acceptance, is not on the PC, so this run uses the VOC person
class.

**Matching:** the fault-free person boxes (score ≥ 0.25) are matched one to one, by
decreasing IoU, to the faulty person boxes:
- a fault-free person with no faulty box at IoU ≥ 0.5 is **lost**, and the image is
  **critical**;
- a pair matched below IoU 0.9 is **moved**;
- an unmatched faulty box is an **extra**.

Moved and extra are tolerable. They are counted apart.

| Conv | weight 1e-07 | weight 1e-05 | bias 1e-07 | bias 1e-05 | act 1e-07 | act 1e-05 | acc 1e-07 | acc 1e-05 |
|---|---|---|---|---|---|---|---|---|
| L00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.8 ± 0.4 | 7.3 ± 0.4 | 3.8 ± 0.4 | 27.5 ± 2.5 |
| L02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.4 | 4.5 ± 1.4 | 1.5 ± 1.4 | 14.4 ± 5.4 |
| L04 | 0.0 ± 0.0 | 3.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.0 ± 2.1 | 0.8 ± 1.1 | 7.1 ± 0.7 |
| L06 | 0.0 ± 0.0 | 4.8 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 3.3 ± 2.5 | 0.3 ± 0.4 | 5.8 ± 2.5 |
| L08 | 0.0 ± 0.0 | 3.3 ± 1.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.4 | 3.0 ± 0.7 | 0.3 ± 0.4 | 5.3 ± 1.8 |
| L10 | 0.5 ± 0.7 | 2.5 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.3 ± 0.4 | 0.0 ± 0.0 | 2.8 ± 1.1 |
| L12 | 0.0 ± 0.0 | 1.5 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.5 ± 0.0 | 0.5 ± 0.0 | 6.6 ± 0.7 |
| L13 | 0.0 ± 0.0 | 0.8 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.8 ± 0.4 |
| L14 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

Critical images in % (a fault-free person lost, among the images with a person).

| Rank | By critical images | By mAP loss |
|---|---|---|
| 1 | L00 (5.1 %) | L00 |
| 2 | L02 (2.6 %) | L02 |
| 3 | L04 (1.9 %) | L04 |
| 4 | L06 (1.8 %) | L06 |
| 5 | L08 (1.5 %) | L14 |
| 6 | L12 (1.1 %) | L08 |
| 7 | L10 (0.9 %) | L10 |
| 8 | L13 (0.3 %) | L12 |
| 9 | L14 (0.0 %) | L13 |

| Bit | weight | bias | act | acc |
|---|---|---|---|---|
| 0 | 1.3 ± 0.6 | 2.7 ± 0.8 | 1.9 ± 0.8 | 0.2 ± 0.3 |
| 1 | 1.9 ± 0.3 |  | 2.4 ± 0.8 |  |
| 2 | 2.5 ± 0.5 |  | 2.2 ± 1.1 |  |
| 3 | 2.0 ± 0.9 |  | 2.2 ± 0.8 |  |
| 4 | 1.7 ± 1.3 | 3.0 ± 0.5 | 3.0 ± 1.8 | 0.0 ± 0.0 |
| 5 | 3.4 ± 0.6 |  | 3.5 ± 2.5 |  |
| 6 | 3.9 ± 1.5 |  | 7.4 ± 1.1 |  |
| 7 | 4.0 ± 2.3 |  | 10.3 ± 0.6 |  |
| 8 |  | 5.7 ± 2.5 |  | 2.4 ± 1.1 |
| 12 |  | 29.5 ± 13.7 |  | 2.9 ± 0.6 |
| 16 |  | 100.0 ± 0.0 |  | 10.4 ± 1.2 |
| 20 |  | 100.0 ± 0.0 |  | 10.9 ± 0.8 |
| 24 |  | 100.0 ± 0.0 |  | 11.6 ± 1.3 |
| 28 |  | 100.0 ± 0.0 |  | 11.6 ± 1.3 |
| 31 |  | 99.2 ± 1.5 |  | 10.8 ± 1.1 |

Critical images in % for one flip at that bit.

| Site | Critical images | Persons lost | Persons moved | Extra boxes / image |
|---|---|---|---|---|
| weight | 1.43 % | 0.65 % | 4.28 % | 0.01 |
| bias | 20.00 % | 19.45 % | 2.86 % | 0.01 |
| act | 2.27 % | 1.06 % | 6.18 % | 0.01 |
| acc | 5.15 % | 2.59 % | 12.73 % | 0.02 |

Mean over every run of the site (all grids).

**mAP loss and critical images per bit** (bit grid: 100 flips per image for `act` /
`acc`, 100 per model for `weight` / `bias`; mean over 3 seeds):

| Site, bit | mAP loss (points) | Critical images |
|---|---|---|
| act bit 6 | 0.05 | 7.4 % |
| act bit 7 | 1.31 | 10.3 % |
| acc bits 16-28 | 0.36-0.56 | 10.4-11.6 % |
| acc bit 31 | 1.37 | 10.8 % |
| bias bit 12 | 2.79 | 29.5 % |
| bias bits 16-31 | 54.7-57.7 | 99-100 % |
| weight bit 7 | -0.03 | 4.0 % |

**Findings**:
- **The mAP hides the critical faults of the accumulators and activations.**
  - A high accumulator bit costs under 0.6 mAP points but loses a person on 10-12 % of
    the images.
  - At BER 1e-5, `act` on L00 *raises* the mAP on both seeds (+0.13 and +0.19 points)
    while 7.3 % of the images lose a person. `acc` on L12 moves the mAP by −0.45 to
    +0.23 points, with 6.6 % critical. The mAP averages over every class and every
    score, so a lost person weighs little against small gains elsewhere.

  So the T25.1 mAP figures cannot size the budget of T31.3. The critical-image share
  can.
- **The layer ranking barely changes.** It is L00 > L02 > L04 > L06 for both criteria.
  The exception is **L14**, the linear output conv: it is 5th by mAP loss (0.05 points)
  but has 0 critical images. Its faults move scores and add low boxes (extras) without
  ever dropping a person above 0.25. The early layers stay the ones to protect, which
  is the choice of T25.4.
- **The bias splits in two.**
  - Bits ≤ 8 are tolerable (3-6 % critical).
  - Bit 12 is critical on 30 % of the images.
  - Bits ≥ 16 kill every person.

  Weights cost little: 1-4 % critical, at most 4.8 % for one conv at BER 1e-5. Their
  int8 range bounds the damage.
- **Moved boxes come before lost persons.** The accumulators move 12.7 % of the persons
  for 2.6 % lost. A person criterion with a localisation tolerance (IoU 0.5) is
  therefore much looser than "output unchanged", which every fault in the bit grid
  breaks.

Not run: CrowdHuman (data not on the PC, row in
[long-runs.md](../docs/tasks/long-runs.md)), which has many more persons per image and
small persons.

## T25.9: faults in the failure monitor

`tools/m25.sh monitor` (`tools/fault_campaign.py monitor`, `yolo.quant.faults.head_forward`),
2026-10-28. Monitor: the integer `miss` head of T39.1 / T39.5 (1,536 → 32 ReLU → 1, 50,981 B)
on the stored integer features of YOLOv4-tiny (T39.0 labels at the T27.2 point). The alarm
threshold is set to 5 % alarms on the val split without fault. A fault **silences** the
monitor on an image that it flags without fault and no longer flags. It raises a **false
alarm** on an image that it newly flags. Draws:
- one flip per image for `act` / `acc` (transient);
- one flip per head over 50 draws for `weight` / `bias` (persistent);
- BER 1e-5 to 1e-3 over 50 draws.

The ABFT check of T25.2 is applied to both layers as 1×1 convs.

| Set | Images | Flagged without fault | of which with a missed person |
|---|---|---|---|
| PREWID `test` | 592 | 47 | 26 |
| PREWID `lab` | 3,000 | 146 | 94 |
| COCO val2017 (person images) | 2,693 | 144 | 138 |

Worst single flip per site (share of the flagged images silenced / false alarms among the
others):

| Site | PREWID `test` | PREWID `lab` | COCO | ABFT sees the silencing faults |
|---|---|---|---|---|
| weight, hidden | 0.6 / 0.1 % | 0.5 / 0.0 % | 2.1 / 0.1 % | 100 % |
| bias, hidden (bit 31) | 7.2 / 4.8 % | 31.6 / 1.1 % | 40.4 / 7.2 % | 100 % |
| act, hidden (bits 6-7) | 12.8 / 3.5 % | 37.7 / 7.2 % | 48.6 / 18.4 % | **0 %** |
| acc, hidden (bit 31) | 12.8 / 5.0 % | 29.5 / 0.6 % | 43.1 / 7.1 % | 100 % |
| weight, output (bit 7) | 3.3 / 2.4 % | 10.2 / 0.3 % | 17.1 / 0.5 % | 100 % |
| bias, output (bits 16-28) | **100** / 0.0 % | **100** / 0.0 % | 0.0 / **100** % | 100 % |
| acc, output (bit 31) | 68.1 / 100 % | 100 / 91.7 % | 100 / 55.8 % | 100 % |

At BER 1e-3, the parameters silence the most:
- hidden weights: 5.1 / 7.0 / 26.7 %;
- hidden biases: 6.0 / 12.0 / 11.4 %.

At BER ≤ 1e-5, every site stays under 0.8 % of silenced alarms. Full tables (every bit and
BER): `build/m25/monitor/*.json`.

- **The monitor fails silent on one bit.** One flip of a high bit (16 to 28) of the output
  bias moves the score by 2^b, beyond its whole range (|score| < 46,000).
  - On PREWID, where the bias is negative (−11, −42), the bit is set and the flip
    subtracts: **every alarm is gone**, for as long as the bias stays corrupted (a
    persistent fault).
  - On COCO, where the bias is positive (+24), the same flip adds instead: 100 % alarms.
  - Which way the head fails depends on the sign of one stored word, not on the design.
  - The output accumulator has the same weak point, but only for one image (transient).
- **The ABFT check sees every silencing weight, bias and accumulator fault** (99-100 %, the
  1 % gap at BER 1e-3 is two flips cancelling in the sum). As in the detector (T25.1), it
  never sees the hidden activations. These are the residual hole: up to 49 % of the alarms
  silenced by one sign-bit flip, with no flag.
- **Effect on the misses**: the alarms on the images with a missed person are silenced at
  the same rate as the others (`silenced_miss` in the JSON, e.g. 13.2 % against 13.7 % on
  PREWID `lab` for a hidden-bias flip). So the monitor loses the very alarms it exists for,
  which is the case CoDANN II warns about.
- **Protection**, an input for T25.4 and T31.3:
  - the 40 bytes of the output layer (32 int8 weights, the int32 bias and accumulator) are cheap to
    duplicate or to hold in ECC;
  - the ABFT lane on the hidden layer covers weights, biases and accumulators;
  - the 32 hidden activations need a parity bit or a range check (ReLU output in
    [0, 127]: a set sign bit is itself an error, which catches the worst act flips for free).
