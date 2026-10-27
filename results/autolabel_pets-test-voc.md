# Auto-labellers on pets (M22, T22.10)

pets test (random subset, seed 0), 200 images, 200 objects (0 small, < 32² px), the 2 class names as prompts (alias map: ), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP@0.75 | IoU matched, median (q10-q90) | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.906 | 0.864 | 0.942 (0.89-0.98, n 193) | nan | 5.19 | 0.89 / 0.96 | 0.91 / 0.96 | 0.94 / 0.96 | 0.95 / 0.96 | 0.95 / 0.95 | 0.96 / 0.92 | 0.97 / 0.90 | 0.98 / 0.86 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` |
|---|---|---|
| cat | 63 | 0.818 |
| dog | 137 | 0.994 |
