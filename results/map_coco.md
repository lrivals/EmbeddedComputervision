# COCO val2017 AP (T11.1, T24.2, T24.5, T33.4, T34.2, T35.3)

COCO val2017 (5,000 images, 36,781 objects), Darknet weights, 416×416 input, confidence
threshold 0.005, per-class NMS 0.45, COCO metric (101-point interpolation, `iscrowd`
ignored; `python/yolo/infer/coco_eval.py`, equal to pycocotools to 1e-4,
`python/tests/test_coco_eval.py`).

```
tools/get_datasets.sh coco
python tools/eval_voc.py --net tiny-yolov4-coco --weights weights/yolov4-tiny.weights \
    --dataset coco --resize stretch            # also --resize letterbox, and tiny-yolov3-coco
tools/m11.sh coco-v4                           # YOLOv4-tiny: float, integer, C-sim
```

Notebook: [coco/tiny-yolov4-coco_infer.ipynb](../notebooks/coco/tiny-yolov4-coco_infer.ipynb) (`INT8 = True` for the
integer stage, `HW = True` for export, golden and C-sim; `SUBSET = 0` for the full split).

## Float

![COCO AP and YOLOv4-tiny stages](figures/resultats/map_coco.png)

Figure: `python -m tools.figures map_coco` (T13.57), read from the tables below.

| Network | Preprocessing | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---|---|---|---|---|---|---|
| **YOLOv4-tiny** (`tiny-yolov4-coco`) | **stretch** | **21.7** | **41.4** | 20.5 | 9.9 | 26.0 | 30.7 | 35.6 |
| YOLOv4-tiny | letterbox | 21.2 | 39.9 | 20.4 | 8.1 | 24.2 | 33.6 | 33.6 |
| Tiny-YOLOv3 (`tiny-yolov3-coco`) | stretch | 16.7 | 37.2 | 12.7 | 5.7 | 19.0 | 26.1 | 29.7 |
| Tiny-YOLOv3 | letterbox | 17.0 | 36.9 | 13.8 | 5.2 | 18.4 | 28.4 | 29.1 |

Published values (AP50, COCO test-dev, 416): **YOLOv4-tiny 40.2**
([AlexeyAB/darknet README](https://github.com/AlexeyAB/darknet), `yolov4-tiny.cfg` row),
**Tiny-YOLOv3 33.1** ([pjreddie.com/darknet/yolo](https://pjreddie.com/darknet/yolo/)).

### Explaining the gaps

- **Preprocessing**: `yolov4-tiny.cfg` has no `letter_box=1`, so AlexeyAB's Darknet resizes
  without keeping the aspect ratio. `stretch` is therefore the reference for YOLOv4-tiny
  (+1.5 AP50 over letterbox), as it is for Tiny-YOLOv2 ([map_float.md](map_float.md)). For
  Tiny-YOLOv3 both are within 0.3 point.
- **YOLOv4-tiny, +1.2 AP50 over the published value**: val2017 against test-dev, a
  different image set. AlexeyAB trained on train2017, so val2017 is a clean split.
- **Tiny-YOLOv3, +4.1 AP50**: the pjreddie weights were trained on the COCO 2014
  `trainvalno5k` split, and val2017 is drawn from the 2014 images. Part of val2017 was
  probably seen in training, which makes this value optimistic. The overlap was not
  measured: the `5k.part` list of the Darknet scripts is no longer online. The published 33.1
  stays the external reference; the comparison between the two networks on val2017 is
  biased in favour of Tiny-YOLOv3.

## YOLOv5n and YOLOv7-tiny, YOLOv5 protocol (T33.4, T34.2)

Protocol of the published values of these networks: the Ultralytics letterbox to a square
input (gray 114, `letterbox_params_v5`), confidence 0.001, class-aware multi-label NMS at
IoU 0.65, at most 300 detections per image; same COCO metric as above. Weights: the
official `yolov5n.pt` (v7.0, read without torch, T33.0) and the Darknet
`yolov7-tiny.weights` (AlexeyAB).

```
python tools/eval_voc.py --net yolov5n-coco --dataset coco --preset v5 --size 640
python tools/eval_voc.py --net tiny-yolov7-coco --weights weights/yolov7-tiny.weights \
    --dataset coco --preset v5 --size 416 --jobs 5 --blas-threads 4
```

| Model | Input | AP | AP50 | AP75 | APs | APm | APl | AR100 | Published AP / AP50 |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv7-tiny** (`tiny-yolov7-coco`) | 640 | **34.8** | **52.9** | 37.3 | 19.0 | 40.0 | 45.3 | 56.3 | — (leaky not published at 640) |
| YOLOv7-tiny | 416 | 34.3 | 51.6 | 36.0 | 13.8 | 36.7 | 53.8 | 53.2 | 35.2 / 52.8 |
| YOLOv5n (`yolov5n-coco`) | 640 | 28.1 | 45.6 | 29.5 | 14.0 | 32.0 | 36.7 | 49.0 | 28.0 / 45.7 |
| YOLOv5n | 416 | 25.3 | 41.4 | 26.3 | 8.7 | 26.8 | 40.5 | 44.5 | — |

Published: YOLOv5n from the Ultralytics README (val2017, 640); YOLOv7-tiny (leaky) from the
YOLOv7 paper, arXiv 2207.02696, Table 1 (val2017, 416). The 38.7 AP at 640 of the paper is
YOLOv7-tiny-SiLU, another network.

- **YOLOv5n**: within 0.1 point of both published values; the decoding, letterbox and
  NMS of T33.4 reproduce the reference evaluation.
- **YOLOv7-tiny, −0.9 AP at 416**: the same pipeline as YOLOv5n, so the gap is not in
  the decoding. Two differences remain, neither measured: the Darknet weights are a
  conversion of the PyTorch `yolov7-tiny.pt` evaluated in the paper; and the paper's
  `test.py` uses rectangular batches (letterbox to the smallest multiple of 32, pad 0.5)
  where this evaluation uses a square letterbox.

## YOLOv8n, YOLOv8 protocol (T35.3)

The protocol above with NMS at IoU 0.7 (the Ultralytics validation of YOLOv8), on the
official `yolov8n.pt` (v8.2.0) read without torch:

```
python tools/eval_voc.py --net yolov8n-coco --dataset coco --preset v8 --size 640
```

Full val2017 (5,000 images), 2026-10-28: 35 min at 640 and 12 min at 416 on 4 to 5
processes of the PC CPU (`--jobs 4 --blas-threads 1` at 640: 4.6 GB per process).

| Model | Input | AP | AP50 | AP75 | APs | APm | APl | AR100 | Published AP |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv8n** (`yolov8n-coco`) | 640 | **37.3** | **52.5** | 40.6 | 18.7 | 41.0 | 53.3 | 59.3 | 37.3 |
| YOLOv8n | 416 | 32.7 | 46.7 | 35.2 | 11.9 | 35.0 | 54.6 | 52.5 | — |

Published: Ultralytics README (val2017, 640). The NumPy network reproduces it to the tenth
of a point. At 416 the loss is on the small objects (APs 18.7 → 11.9); APl even gains a
point, as with the other networks.

## Tiny-YOLOv3: float, integer, FPGA C-sim (T11.1)

`tools/m11.sh coco-int`, `coco-calib`, `coco-fpga`, 2026-10-28. `stretch` preprocessing at
416, conf 0.005, NMS 0.45; same stages as YOLOv4-tiny below. The integer model uses the
current calibration (500 VOC2007 trainval images), as the export `model/tiny-yolov3-coco`.

| Stage | AP | AP50 | AP75 | Equality with the integer model |
|---|---|---|---|---|
| Float (fused BN, NumPy float32) | 16.75 | 37.22 | 12.75 | — |
| Integer Python (`IntNetwork`, bit-exact), VOC calibration | 16.20 | 36.57 | 12.01 | reference |
| FPGA, C-sim (`yolo_bench`, sim backend, `ACC_NO_APINT`) | 16.20 | 36.57 | 12.01 | **5,000 / 5,000 images identical** |
| Integer Python, COCO calibration (500 train2017 images) | 16.21 | 36.69 | 12.14 | — (other scales) |

**Findings:**
- **Integer loss: −0.55 AP, −0.65 AP50**, the same order as YOLOv4-tiny (−1.2 AP).
  Per class, at most −2.4 AP (train).
- **The C-sim equals the integer model on every image**, with the `-ffp-contract=off`
  build of T24.5: the second network after YOLOv4-tiny to pass the full COCO split
  bit for bit.
- **Calibrating on COCO instead of VOC gains almost nothing** (+0.01 AP, +0.12 AP50). The
  activation scales of the 13 layers differ by −1 to +17 % (median +3 %) between the two
  sets ([calibration_tiny-yolov3-coco-coco.md](calibration_tiny-yolov3-coco-coco.md)), so
  the VOC calibration of the export can stay.
- Costs: float + int 34 min (6 jobs, 394 ms/image), C-sim 2 h 54 on 6 cores (≈ 12.5 s per
  image per core).

## YOLOv7-tiny: float, integer, FPGA C-sim (T26.0)

`build/longruns/t26-0.sh`, 2026-10-29. `stretch` preprocessing at 416, conf 0.005, NMS
0.45 (the protocol of the two sections around, not the YOLOv5 one above: 34.3 AP there).
The integer model is the one of the export `model/tiny-yolov7-coco` (calibration on 500
COCO train2017 images, [calibration_tiny-yolov7-coco.md](calibration_tiny-yolov7-coco.md)).
The 4-source concats are written end to end by their source convs (`layout:
"contiguous"`), and the SPP pools are chained 5 × 5 stride-1 pools of the output stage.

| Stage | AP | AP50 | AP75 | Equality with the integer model |
|---|---|---|---|---|
| Float (fused BN, NumPy float32) | 33.14 | 50.69 | 34.65 | — |
| Integer Python (`IntNetwork`, bit-exact) | 31.51 | 49.93 | 33.37 | reference |
| FPGA, C-sim (`yolo_bench`, sim backend, `ACC_NO_APINT`) | 31.51 | 49.93 | 33.37 | **5,000 / 5,000 images identical** |

- **Integer loss −1.6 AP, −0.8 AP50**, more than Tiny-YOLOv3 (−0.55 AP) and YOLOv4-tiny
  (−1.2 AP); per class, person 44.0 → 42.1.
- **Third network to pass the full split bit for bit**, with no kernel change.
- C-sim: 3 h 35 on 6 cores (≈ 15.5 s per image per core), against 2 h 54 for Tiny-YOLOv3.

## YOLOv4-tiny: float, integer, FPGA C-sim (T24.5)

`stretch` preprocessing. The integer model uses the calibration on 500 COCO train2017
images ([calibration_tiny-yolov4-coco.md](calibration_tiny-yolov4-coco.md)). The C-sim
stage is the HLS kernel in C simulation behind the ARM driver (`yolo_bench`, backend
`sim`), on the same int8 inputs as the integer model. The full pass on the 5,000 images
ran from the notebook on 2026-10-28 (about 1 h 05 for the whole notebook, 15 processes),
and its detections are identical to the integer model on every image.

| Stage | AP | AP50 | AP75 | APs | APm | APl | Equality with the integer model |
|---|---|---|---|---|---|---|---|
| Float (fused BN, NumPy float32) | 21.7 | 41.4 | 20.5 | 9.9 | 26.0 | 30.7 | — |
| Integer Python (`IntNetwork`, bit-exact) | 20.5 | 40.1 | 18.8 | 8.6 | 24.4 | 29.0 | reference |
| FPGA, C-sim | 20.5 | 40.1 | 18.8 | 8.6 | 24.4 | 29.0 | **5,000 / 5,000 images identical** |

**Fix found on the way**: the first full pass gave 4,995 / 5,000 identical images.
- **Cause**: on the 5 others, one box coordinate differed by 1 ULP of float64. The fast
  bench is built with `-march=native`, so GCC fused `σ·α − (α − 1)/2` of the `scale_x_y`
  decode into an FMA, which rounds once where NumPy rounds twice. The integer heads were
  identical.
- **Fix**: `-ffp-contract=off` in `sw/CMakeLists.txt` and `cpp/golden/CMakeLists.txt`.
  This also matters on the board, since GCC fuses by default on aarch64.
- **Check**: the 5 chunks of 10 images that held them were recomputed with the fixed build
  and match the integer model on all 50 images. The other 4,950 images were already
  identical; the fused and unfused decodes give the same double on them.

**Integer vs float: −1.2 AP, −1.3 AP50**, above the 1-point target of T24.5. QAT (as in
[M9.3](../docs/tasks/M9.3-quant-4bits.md), T9.3.6) is left as follow-up. On 200 images, fake-quantizing
the two heads alone costs nothing (45.5 and 45.7 AP50 against 45.5 in float), so the loss
is spread over the backbone. A head scale above 1/8 would reduce the 1 % clipping of L29 but
overflows the exponential table of the post-processing (`yolo.quant.lut`).
