# Auto-labellers on flir (M22, T22.10)

flir val (random subset, seed 0), 100 images, 1426 objects (1057 small, < 32² px), the 15 class names as prompts (alias map: `bike` → `bicycle`, `motor` → `motorcycle`, `light` → `traffic light`, `hydrant` → `fire hydrant`, `sign` → `traffic sign`), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.335 | 0.213 | 12.11 | 0.43 / 0.49 | 0.58 / 0.42 | 0.70 / 0.36 | 0.80 / 0.30 | 0.87 / 0.25 | 0.92 / 0.15 | 0.94 / 0.11 | 0.94 / 0.08 |
| `owlv2-base` | 0.346 | 0.218 | 12.22 | 0.47 / 0.39 | 0.58 / 0.32 | 0.69 / 0.26 | 0.79 / 0.20 | 0.87 / 0.15 | 0.78 / 0.02 | 0.72 / 0.01 | 0.83 / 0.01 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` |
|---|---|---|---|
| person | 387 | 0.568 | 0.506 |
| bike | 19 | 0.501 | 0.517 |
| car | 561 | 0.648 | 0.614 |
| motor | 7 | 0.447 | 0.491 |
| bus | 15 | 0.258 | 0.227 |
| truck | 4 | 0.010 | 0.010 |
| light | 195 | 0.078 | 0.169 |
| hydrant | 13 | 0.439 | 0.562 |
| sign | 218 | 0.220 | 0.290 |
| other vehicle | 7 | 0.178 | 0.071 |
