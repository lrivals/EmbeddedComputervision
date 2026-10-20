# Auto-labellers on VOC (M22, T22.7)

VOC 2007 test, 200 images, 473 objects (21 small, < 32² px), the 20 class names as prompts (alias map: `aeroplane` → `airplane`, `diningtable` → `dining table`, `motorbike` → `motorcycle`, `pottedplant` → `potted plant`, `tvmonitor` → `tv monitor`), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.832 | 0.390 | 6.73 | 0.23 / 0.96 | 0.34 / 0.95 | 0.44 / 0.92 | 0.55 / 0.91 | 0.65 / 0.88 | 0.79 / 0.79 | 0.84 / 0.73 | 0.88 / 0.67 |
| `owlv2-base` | 0.862 | 0.373 | 4.80 | 0.48 / 0.67 | 0.57 / 0.56 | 0.67 / 0.52 | 0.75 / 0.46 | 0.81 / 0.38 | 0.90 / 0.26 | 0.91 / 0.22 | 0.95 / 0.19 |
| `yolo:tiny-yolov2-voc` | 0.569 | 0.000 | 0.32 | 0.55 / 0.60 | 0.63 / 0.56 | 0.68 / 0.53 | 0.73 / 0.50 | 0.79 / 0.48 | 0.87 / 0.42 | 0.88 / 0.38 | 0.90 / 0.34 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` | `yolo:tiny-yolov2-voc` |
|---|---|---|---|---|
| aeroplane | 7 | 0.977 | 0.945 | 0.805 |
| bicycle | 10 | 0.940 | 0.921 | 0.636 |
| bird | 16 | 0.937 | 0.995 | 0.597 |
| boat | 13 | 0.655 | 0.428 | 0.282 |
| bottle | 16 | 0.845 | 0.787 | 0.323 |
| bus | 8 | 1.000 | 0.980 | 0.705 |
| car | 57 | 0.841 | 0.833 | 0.713 |
| cat | 14 | 0.994 | 1.000 | 0.626 |
| chair | 37 | 0.711 | 0.671 | 0.418 |
| cow | 11 | 0.809 | 0.969 | 0.394 |
| diningtable | 11 | 0.302 | 0.856 | 0.301 |
| dog | 17 | 0.995 | 1.000 | 0.704 |
| horse | 15 | 1.000 | 0.979 | 0.858 |
| motorbike | 12 | 0.829 | 0.818 | 0.694 |
| person | 184 | 0.855 | 0.783 | 0.535 |
| pottedplant | 10 | 0.474 | 0.656 | 0.545 |
| sheep | 7 | 0.723 | 0.894 | 0.470 |
| sofa | 8 | 0.919 | 0.792 | 0.523 |
| train | 11 | 0.935 | 0.943 | 0.717 |
| tvmonitor | 9 | 0.891 | 1.000 | 0.532 |
