# Out-of-domain mAP without fine-tuning (T11.2, T11.3)

VOC and COCO Darknet weights on other datasets, without fine-tuning. Evaluation as in
`tools/m11.sh` (`tools/eval_quant.py`): stretch 416×416, confidence 0.005, NMS 0.45,
VOC11 AP over the classes the dataset shares with the training set (`MAPPINGS`).
Integer model: the INT8 `QuantModel` of the calibration named in the column.

## Float → integer gap out of domain (T11.2)

`HD_SETS="voc exdark visdrone" tools/m11.sh hors-domaine` (2026-10-27, 30 min on 16
processes). VOC weights calibrated on 500 VOC2007 trainval images, COCO weights on 500
COCO train2017 images (`calib2017`): the calibrations of the repository, none made on the
target dataset. Classes: those shared with the training set (`MAPPINGS`), as many as in
the "classes" column.

| Dataset (split, images) | Network | Classes | float | int | int − float |
|---|---|---|---|---|---|
| VOC2007 test (4,952) | tiny-yolov2-voc | 20 | 56.30 | 55.66 | −0.64 |
| VOC2007 test (4,952) | tiny-yolov3-coco | 20 | 58.54 | 57.54 | −1.00 |
| ExDark test (2,563) | tiny-yolov2-voc | 11 | 27.41 | 27.01 | −0.40 |
| ExDark test (2,563) | tiny-yolov3-coco | 12 | 31.97 | 31.23 | −0.74 |
| VisDrone val (548) | tiny-yolov2-voc | 5 | 7.36 | 6.47 | −0.89 |
| VisDrone val (548) | tiny-yolov3-coco | 6 | 8.00 | 7.52 | −0.48 |
| KITTI val (1,496) | both | — | pending | pending | — |

- **The calibration stays valid out of domain**: the integer loss is 0.4 to 0.9 point on
  ExDark and VisDrone, against 0.64 (v2) and 1.00 (v3) on VOC itself. The gap does not
  widen when the domain changes, so the out-of-domain mAP drop comes from the weights,
  not from the quantization.
- **The drop is in the weights**: VOC mAP 56-59 falls to 27-32 in low light and to 7-8 on
  drone views, where most objects are smaller than a 26×26 cell
  ([stats-jeux.md](../docs/tasks/stats-jeux.md)).
- KITTI: `data/kitti/training/image_2` is incomplete (`data_object_image_2.zip.part`);
  the rows come with the end of that download (`tools/m11.sh hors-domaine`).
- Raw outputs: `build/m11/hors-domaine/<dataset>_<net>.json`, figures in
  `build/m11/hors-domaine/figures/`.

## ExDark test, low light (T11.3)

2,563 images. Commands: the ExDark part of `tools/m11.sh hors-domaine`, then
`tools/m11.sh exdark` (2026-10-27).

| Network | float | int, VOC calibration | int, ExDark calibration |
|---|---|---|---|
| tiny-yolov2-voc | 27.41 | 27.01 | 27.17 |
| tiny-yolov3-coco | 31.97 | 31.23 | 30.76 |

- Calibrating on ExDark does not recover the INT8 loss: +0.16 point for v2, −0.47 for
  v3. The INT8 loss on ExDark (0.4-0.7 point) is close to the one on VOC.
- Activations of L00-L04 (`tools/act_hist.py`, 200 test images per dataset): on ExDark
  the p99 of |x| at L00 falls from 9.9 to 5.6 (v2) and from 8.7 to 5.2 (v3), but the max
  does not move (65, 56), and the max is what sets the scale. With the VOC scales, L00 uses
  122 / 128 (v2) and 121 / 128 (v3) of the positive int8 levels on ExDark, and 127 with
  the ExDark scales. L02 and L04 use 127-128 levels in every case.

| Layer | Dataset | p50 \|x\| | p99 | max | levels used, VOC scales | levels used, ExDark scales |
|---|---|---|---|---|---|---|
| v2 L00 | VOC | 0.742 | 9.91 | 65.0 | 128 | 128 |
| v2 L00 | ExDark | 0.879 | 5.57 | 65.1 | 122 | 127 |
| v3 L00 | VOC | 0.687 | 8.73 | 56.4 | 128 | 128 |
| v3 L00 | ExDark | 0.763 | 5.15 | 56.3 | 121 | 127 |

Calibration reports: `results/calibration_{tiny-yolov2-voc,tiny-yolov3-coco}-exdark.md`.
Histograms: `build/m11/exdark/act_calib{voc,exdark}_<net>/act_hist.png`.
