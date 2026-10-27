# Auto-labellers on prewid (M22, T22.10)

prewid test (random subset, seed 0), 100 images, 53 objects (2 small, < 32² px), the 1 class names as prompts (alias map: ), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP@0.75 | IoU matched, median (q10-q90) | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.989 | 0.989 | 0.949 (0.88-0.98, n 53) | 0.545 | 30.60 | 0.31 / 1.00 | 0.56 / 1.00 | 0.73 / 1.00 | 0.85 / 1.00 | 0.91 / 1.00 | 0.95 / 0.98 | 0.95 / 0.98 | 0.94 / 0.96 |
| `owlv2-base` | 0.990 | 0.902 | 0.958 (0.85-0.96, n 3) | 1.000 | 23.04 | 1.00 / 0.57 | 1.00 / 0.23 | 1.00 / 0.06 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` |
|---|---|---|---|
| person | 53 | 0.989 | 0.990 |
