# Channel pruning (T10.11)

Tiny-YOLOv2 VOC, Darknet weights. Structured pruning by L1 norm of the filter after BN
folding (`python/yolo/prune.py`, `tools/prune.py`). The kept filters are rounded to the
nearest multiple of Tm = 32; L00, L02 and the head are not pruned. Each variant follows the
same chain (`tools/prune_study.sh`, tier M of M12):

1. float fine-tuning of 300 iterations, batches of 8, lr 5·10⁻⁴;
2. INT8 calibration on 500 images;
3. integer mAP on 500 images of VOC2007 test (`--resize stretch`);
4. export, then cycles of the current kernel (`perf_model --manifest`, 200 MHz).

Reference: unpruned INT8, **57.78** on the same 500 images; 7,443,739 cycles, 37.2 ms,
26.9 img/s.

| variant | kept filters (L04, L06, L08, L10, L12, L13) | weights | mAP without fine-tuning | mAP after fine-tuning | cycles | ms | img/s | effective GOPS | MAC efficiency |
|---|---|---|---|---|---|---|---|---|---|
| r30 (all) | 32, 96, 192, 352, 704, 704 | 7.59 MB | 2.48 | 11.89 | 4,483,801 | 22.4 | 44.6 | 161 | 52.5 % |
| r50 (all) | 32, 64, 128, 256, 512, 512 | 4.00 MB | 1.04 | 8.02 | 3,308,491 | 16.5 | 60.5 | 137 | 44.6 % |
| r70 (all) | 32, 32, 64, 160, 320, 320 | 1.56 MB | 0.56 | 1.61 | 2,423,017 | 12.1 | 82.5 | 110 | 35.9 % |

"Tail" variant (50 % on L10, L12 and L13 only, 4.59 M parameters): pruned
(`RATES="50:10,12,13" bash tools/prune_study.sh`), then stopped at the start of the
fine-tuning; it is not measured.

Reading:

- **Cycles.** They go down more slowly than the MACs: at 70 %, 2.4 M cycles remain for
  0.67 GMAC per image (3.49 unpruned), i.e. 36 % efficiency against 61 % unpruned. With
  fewer channels, the Tm = 32 and Tn = 24 tiles are less filled: L04 keeps 32 filters and
  feeds L06 with 32 channels, i.e. 2 Tn tiles, the second only a third full. The loads and
  the 13 × 13 layers weigh more.
- **mAP.** A uniform pruning destroys the network (less than 3 points without fine-tuning,
  even at 30 %), and 300 iterations (2,400 images, i.e. 0.15 epoch of VOC 07 + 12) only
  recover a small part of it. At tier M, no variant is usable: it needs a fine-tuning of
  several epochs (tier N), or a progressive pruning.
- **Comparison.** 2026-fata (pruned by 70 %, QAT, DPU of the KV260) reports 24.3 img/s.
  Here, the kernel does 82.5 img/s at 70 % on the same board (C-sim projection, without the
  post-processing or the transfers), but with an unusable mAP. The comparison will only
  hold at a comparable mAP, after a long fine-tuning. 2023-zhai (INT16, vehicles) stays out
  of reach of a direct comparison.
- **Integer contract.** On the r30 export, the C++ golden model, the C-sim (`tb_net --model
  build/m10/prune/r30 --net model`) and the Python dumps give 0 differences on 3 images.
  The C-sim cycles are equal to the model (4,483,801).

## Remaining

- "Tail" variant to measure, then long fine-tuning (tier N, several epochs) and mAP on the
  4,952 images.
- Pruning guided by the per-layer sensitivity, or progressive (small steps and fine-tuning
  in between).
- Own weights fine-tuned on VOC (T10.12, pointer to T2.9).
