# Auto-labellers on visdrone (M22, T22.10)

visdrone val (random subset, seed 0), 100 images, 7181 objects (5050 small, < 32² px), the 10 class names as prompts (alias map: `people` → `sitting person`, `awning-tricycle` → `tricycle with awning`, `motor` → `motorcycle`), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.267 | 0.185 | 14.53 | 0.59 / 0.38 | 0.67 / 0.31 | 0.73 / 0.25 | 0.77 / 0.18 | 0.81 / 0.12 | 0.88 / 0.04 | 0.87 / 0.01 | 0.83 / 0.00 |
| `owlv2-base` | 0.264 | 0.216 | 12.82 | 0.43 / 0.44 | 0.54 / 0.39 | 0.63 / 0.32 | 0.69 / 0.21 | 0.70 / 0.09 | 0.63 / 0.01 | 0.73 / 0.00 | 0.83 / 0.00 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` |
|---|---|---|---|
| pedestrian | 1540 | 0.290 | 0.247 |
| people | 1012 | 0.241 | 0.091 |
| bicycle | 230 | 0.187 | 0.184 |
| car | 2634 | 0.692 | 0.661 |
| van | 372 | 0.192 | 0.242 |
| truck | 157 | 0.219 | 0.259 |
| tricycle | 176 | 0.153 | 0.082 |
| awning-tricycle | 87 | 0.097 | 0.051 |
| bus | 33 | 0.336 | 0.615 |
| motor | 940 | 0.267 | 0.208 |
