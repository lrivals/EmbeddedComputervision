# Auto-labellers on exdark (M22, T22.10)

exdark test (random subset, seed 0), 100 images, 287 objects (13 small, < 32² px), the 12 class names as prompts (alias map: `Motorbike` → `motorcycle`, `People` → `person`), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.730 | 0.308 | 13.09 | 0.25 / 0.86 | 0.38 / 0.83 | 0.50 / 0.80 | 0.62 / 0.73 | 0.70 / 0.67 | 0.83 / 0.51 | 0.90 / 0.45 | 0.94 / 0.37 |
| `owlv2-base` | 0.752 | 0.398 | 11.98 | 0.58 / 0.70 | 0.66 / 0.57 | 0.71 / 0.47 | 0.75 / 0.36 | 0.83 / 0.32 | 0.92 / 0.23 | 0.93 / 0.19 | 0.98 / 0.14 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` |
|---|---|---|---|
| Bicycle | 8 | 0.831 | 0.705 |
| Boat | 24 | 0.530 | 0.575 |
| Bottle | 18 | 0.627 | 0.745 |
| Bus | 4 | 1.000 | 1.000 |
| Car | 41 | 0.664 | 0.851 |
| Cat | 20 | 0.715 | 0.808 |
| Chair | 24 | 0.644 | 0.690 |
| Cup | 14 | 0.462 | 0.490 |
| Dog | 18 | 0.774 | 0.818 |
| Motorbike | 10 | 1.000 | 0.818 |
| People | 90 | 0.807 | 0.779 |
| Table | 16 | 0.706 | 0.742 |
