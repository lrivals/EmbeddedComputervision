"""Decision explanation (M36): why a box is detected and why a person is missed.

NumPy only; the drawing is in `tools/` (ADR 0001):

- `targets`: the scalar to explain (detection or miss) and its δ on the heads (T36.0);
- `gradient`: saliency, SmoothGrad, Integrated Gradients, Grad-CAM, sanity checks (T36.1);
- `perturb`: occlusion and D-RISE at the float, integer and golden stages (T36.2);
- `int_attr`: exact integer contributions and LRP over the integer network (T36.3);
- `evaluate`: pointing game, energy in the box, deletion and insertion (T36.4);
- `misses`: causes of the missed objects (T36.5);
- `concepts`: Network Dissection-lite and TCAV (T36.6);
- `recorder`: per-layer and per-image training records (T36.7);
- `cka`: linear CKA between representations (T36.8);
- `maps`: shared map helpers.
"""
