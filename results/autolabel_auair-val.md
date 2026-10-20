# Auto-labellers on auair (M22, T22.10)

auair val (random subset, seed 0), 100 images, 334 objects (36 small, < 32² px), the 8 class names as prompts (alias map: `Human` → `person`, `Motorbike` → `motorcycle`), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.094 | 0.005 | 14.62 | 0.11 / 0.37 | 0.15 / 0.32 | 0.20 / 0.29 | 0.24 / 0.26 | 0.28 / 0.23 | 0.33 / 0.12 | 0.31 / 0.07 | 0.40 / 0.04 |
| `owlv2-base` | 0.115 | 0.337 | 8.45 | 0.15 / 0.51 | 0.18 / 0.48 | 0.21 / 0.45 | 0.25 / 0.40 | 0.28 / 0.32 | 0.30 / 0.04 | 0.36 / 0.03 | 0.25 / 0.01 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` |
|---|---|---|---|
| Human | 50 | 0.023 | 0.057 |
| Car | 236 | 0.225 | 0.201 |
| Truck | 15 | 0.093 | 0.078 |
| Van | 22 | 0.306 | 0.344 |
| Bicycle | 3 | 0.000 | 0.001 |
| Bus | 4 | 0.010 | 0.055 |
| Trailer | 4 | 0.000 | 0.068 |
