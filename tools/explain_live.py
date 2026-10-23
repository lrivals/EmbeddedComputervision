"""T36.9: per-frame explanation map for `tools/webcam.py --gradcam` and `tools/explain_app.py`.

`cam_function(spec)` returns `fn(PIL image) -> (map (h, w) in [0, 1] over the image, or None
when nothing is detected)` for the best detection of the frame (conf 0.25):

- float models: Grad-CAM (`yolo.explain.gradient.gradcam`) on the BN-folded network, its own
  forward and backward pass;
- integer models (`int:<net>`): the integer CAM of the 1 × 1 head, Σ_c qW[c]·L[c] over the map
  read by the head conv (`int_attr.cam_l13`, the map of T36.10), ReLU, after an integer
  forward pass.

The letterbox padding is cropped and the map resized to the frame. The cost includes the
second forward pass (the detection of the demo is not reused).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

CONF = 0.25


def _unletterbox(m, w, h, size):
    from PIL import Image

    from yolo.data.letterbox import resize_params

    nw, nh, dx, dy = resize_params(w, h, size, "letterbox")
    crop = np.asarray(m, dtype=np.float32)[dy:dy + nh, dx:dx + nw]
    img = Image.fromarray(crop).resize((w, h), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32)
    hi = out.max()
    return out / hi if hi > 0 else out


def cam_function(spec):
    from yolo.explain.maps import resize
    from yolo.explain.targets import detections
    from yolo.infer.pipeline import preprocess

    if spec.startswith("int:"):
        from yolo.explain.int_attr import cam_l13
        from yolo.io.export import load_model as load_export
        from yolo.quant.int_model import IntNetwork
        from yolo.quant.quantize import quantize_input

        qm = load_export(ROOT / "model" / spec[4:])[0]
        inet = IntNetwork(qm, "f64")
        size = qm.net["input"][1]

        def fn(img):
            x, (w, h) = preprocess(img, size, "letterbox")
            outs = inet.forward(quantize_input(x[None]), all_outputs=True)
            heads = {i: outs[i].astype(np.float64) * qm.scale_of(i) for i in qm.outputs}
            dets = detections(heads, qm.net, CONF)
            if not dets:
                return None
            cam = np.maximum(cam_l13(qm, outs, dets[0]), 0).astype(np.float64)
            return _unletterbox(resize(cam, size, size), w, h, size)
        return fn

    from yolo.explain.gradient import gradcam
    from yolo.explain.targets import explain_net

    net = explain_net(spec, dtype=np.float32)
    size = net.net["input"][1]

    def fn(img):
        x, (w, h) = preprocess(img, size, "letterbox")
        x = x[None]
        dets = detections(net.forward(x, train=False), net.net, CONF)
        if not dets:
            return None
        return _unletterbox(gradcam(net, x, dets[0])["map"], w, h, size)
    return fn
