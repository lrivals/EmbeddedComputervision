# Auto-labellers on prewid (M22, T22.10)

prewid lab (random subset, seed 0), 100 images, 53 objects (0 small, < 32² px), the 1 class names as prompts (alias map: ), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP@0.75 | IoU matched, median (q10-q90) | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.964 | 0.961 | 0.971 (0.91-0.99, n 53) | nan | 22.53 | 0.56 / 1.00 | 0.59 / 1.00 | 0.62 / 1.00 | 0.64 / 0.98 | 0.64 / 0.98 | 0.94 / 0.91 | 1.00 / 0.83 | 1.00 / 0.79 |
| `owlv2-base` | 0.898 | 0.884 | nan (nan-nan, n 0) | nan | 19.05 | 1.00 / 0.06 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` |
|---|---|---|---|
| person | 53 | 0.964 | 0.898 |
