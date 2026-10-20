# Auto-labellers on hituav (M22, T22.10)

hituav test (random subset, seed 0), 100 images, 827 objects (639 small, < 32² px), the 4 class names as prompts (alias map: `OtherVehicle` → `vehicle`), raw predictions ≥ 0.05, device cpu.

| labeller | mAP@0.5 | mAP small | s / image | P / R @ 0.2 | P / R @ 0.25 | P / R @ 0.3 | P / R @ 0.35 | P / R @ 0.4 | P / R @ 0.5 | P / R @ 0.55 | P / R @ 0.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `grounding-dino-tiny` | 0.159 | 0.172 | 11.79 | 0.39 / 0.27 | 0.49 / 0.21 | 0.58 / 0.17 | 0.65 / 0.13 | 0.77 / 0.10 | 0.81 / 0.04 | 0.88 / 0.03 | 0.82 / 0.01 |
| `owlv2-base` | 0.231 | 0.184 | 11.77 | 0.68 / 0.19 | 0.72 / 0.11 | 0.77 / 0.08 | 0.81 / 0.06 | 0.89 / 0.05 | 1.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

AP per class (classes present in the subset):

| class | objects | `grounding-dino-tiny` | `owlv2-base` |
|---|---|---|---|
| Person | 557 | 0.346 | 0.396 |
| Car | 186 | 0.292 | 0.457 |
| Bicycle | 79 | 0.000 | 0.007 |
| OtherVehicle | 5 | 0.000 | 0.064 |
