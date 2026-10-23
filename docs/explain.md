# Explaining the detector's decisions (M36)

The explanation tools say why the detector output a box and, above all, why it missed a
person. They work at the three stages of the chain: float, integer, and golden (== C-sim,
M28). The code lives in `python/yolo/explain/` (NumPy only), the command line in
`tools/explain.py`, the viewer in `tools/explain_app.py`. Results:
[results/explain.md](../results/explain.md). Task list: [tasks/M36-explanation.md](tasks/M36-explanation.md).

## Quick start

```bash
pip install -e "python[autolabel]"     # once: gradio (the stack of make autolabel)
make explain                           # viewer on http://127.0.0.1:7861
python tools/explain.py map --image 000004 --methods gradcam,drise,lrp_int@int
python tools/webcam.py --gradcam       # live map of the best detection over the frame
```

The integer and golden stages need the calibration `build/quant/<net>/calib.json`
(`make calibrate`). The golden stage also needs the exports `model/<net>/` (`make export`)
and `build/golden/golden_run` (`make golden-check`).

## The viewer (`make explain`)

1. Pick a network and a VOC2007 test image (the first 300), or upload a photo, and click
   **load**. The target list holds the detections (`det k: class score`), then the ground
   truth boxes (`gt k: class`, VOC images only).
2. Pick a target, a method and a stage, then click **explain**. The map is drawn over the
   image with the target box in white.
3. The text under the map gives:
   - the cost (passes, seconds);
   - the five most relevant channels of the map read by the head: the Grad-CAM weights,
     or the per-channel relevance of the head conv for LRP;
   - for a ground truth box, the miss cause at the float and integer stages, or
     "detected".

| Method | Stages | Cost per map (PC, one process) |
|---|---|---|
| `gradcam`, `saliency` | float | 1 forward + 1 backward, ≈ 1 s |
| `smoothgrad` (25 samples), `ig_black` / `ig_blur` (64 steps) | float | 50 / 130 passes, 20 s to 1 min |
| `lrp_float`, `lrp_int` | float, integer | 1 forward + the propagation, a few s |
| `occlusion` (64 px patch, stride 32), `drise` (300 masks) | float, integer, golden | 145 / 300 inferences, 0.5 to 15 min |

The golden stage runs `golden_run` once per perturbed image: use it to check a map, not
to explore.

Screenshot: *to be added by hand* (`docs/figures/explain_app.png`). The app is checked by
hand: the T36.9 acceptance asks for it, and it cannot be checked headless.

## Webcam overlay (`tools/webcam.py --gradcam`)

On every analysed frame, the map of the best detection (conf 0.25) is blended over the
image:

- **float models**: Grad-CAM of the BN-folded network;
- **integer models** (`int:<net>`): the integer CAM of the 1 × 1 head over the map it
  reads, the per-frame map that T36.10 puts on the chip.

The map is computed in its own pass, after the detection. The headless summary prints its
cost in ms per frame. To measure it on a folder:
`python tools/webcam.py --model int:tiny-yolov2-voc --source <images> --no-window --gradcam --max-frames 10`.

## Command line

`python tools/explain.py <command> --help`. Every command writes
`build/explain/<command>.json`; `report` writes the tables of
`results/explain.md` and the miss examples in `results/explain/`. The notebook
[notebooks/explain.ipynb](../notebooks/explain.ipynb) runs the same commands.

| Command | Task | What it measures |
|---|---|---|
| `map` | T36.9 | PNG of the maps of chosen methods, `build/explain/maps/<net>/<id>.png` |
| `sanity` | T36.1 | IG completeness, Spearman against cascading weight randomization |
| `stages` | T36.2 | D-RISE and occlusion, float against integer; integer == golden |
| `intattr` | T36.3 | Σ qW·qx == acc − qb, relevance per layer, integer against float maps |
| `evaluate` | T36.4 | pointing game, energy in the box, deletion and insertion AUC |
| `misses` | T36.5 | cause of each missed person, float and integer |
| `concepts` | T36.6 | Network Dissection-lite and TCAV on a layer |
| `cka` | T36.8 | CKA float / integer per layer, CKA across checkpoints |
| `record` | T36.7 | never-learned and forgotten images of a `tools/train.py --record` run |
