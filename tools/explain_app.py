"""T36.9: explanation viewer (`make explain`), a gradio app on http://127.0.0.1:7861.

Pick a VOC2007 test image (or upload one), a network, then a target: a detection (why this
box) or a ground-truth box (why it is missed, VOC images only). Pick a method and a stage,
and the app shows the map over the image. For a ground-truth box it also gives the miss
cause at the float and integer stages (`yolo.explain.misses`). The best channels of the
layer read by the head are listed too: the Grad-CAM weights for float maps, the per-channel
relevance of the head conv for LRP.

Same UI stack as `make autolabel` (gradio, the `autolabel` extra), local only, no share link.
The models and the images are those of `tools/explain.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "python", ROOT / "tools", ROOT):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import explain as X  # noqa: E402

METHODS = ("gradcam", "saliency", "smoothgrad", "ig_blur", "ig_black", "occlusion", "drise",
           "lrp_float", "lrp_int")
STAGES = ("float", "int", "golden")
N_IDS = 300


def _targets(name, s, x):
    """[(label, Target)]: the detections, then the ground-truth boxes as miss targets."""
    from yolo.data import datasets
    from yolo.explain.targets import detections, miss_target

    net = X.models(name)[0]
    names = datasets.model_family("voc", net.net["classes"])[1]
    out = [(f"det {k}: {names[d.cls]} {d.score:.2f}", d)
           for k, d in enumerate(detections(net.forward(x, train=False), net.net, X.CONF,
                                            X.IOU))]
    if s is not None:
        boxes, labels = X.gt_of(name, s)
        for k, (b, c) in enumerate(zip(boxes, labels)):
            t = miss_target(net.net, tuple(b), int(c))
            if t is not None:
                out.append((f"gt {k}: {names[c]}", t))
    return out


def _render(x, m, box):
    from yolo.explain.maps import normalize

    import matplotlib

    img = np.asarray(x[0]).transpose(1, 2, 0)
    heat = matplotlib.colormaps["jet"](normalize(m))[..., :3]
    out = (0.55 * img + 0.45 * heat).clip(0, 1)
    cx, cy, bw, bh = box
    h, w = img.shape[:2]
    x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
    y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
    x0, x1, y0, y1 = max(x0, 0), min(x1, w - 1), max(y0, 0), min(y1, h - 1)
    out[y0:y1, [x0, x1]] = 1
    out[[y0, y1], x0:x1] = 1
    return (out * 255).astype(np.uint8)


def _cause(name, s, t, x):
    from yolo.explain import misses as M
    from yolo.explain.targets import grid_sizes
    from yolo.quant.quantize import quantize_input

    net, qm, inet, luts = X.models(name)
    fo = M.float_outputs(net.forward(x, train=False), net.net, X.CONF, X.IOU)
    io = M.int_outputs(inet.forward(quantize_input(x)), qm.net, luts, X.CONF, X.IOU)
    g = grid_sizes(net.net)
    f = M.classify_miss(net.net, t.box, t.cls, fo, g, X.CONF)
    i = M.classify_miss(qm.net, t.box, t.cls, io, g, X.CONF, float_out=fo)
    return f"float: **{f or 'detected'}**, integer: **{i or 'detected'}**"


def _channels(name, method, x, t):
    from yolo.explain import gradient as G
    from yolo.explain import int_attr as IA
    from yolo.quant.quantize import quantize_input

    net, _, inet, _ = X.models(name)
    if method == "lrp_int":
        w = IA.lrp_int(inet, quantize_input(x), t).head_channels
    elif method == "lrp_float":
        w = IA.lrp_float(net, x.astype(np.float64), t).head_channels
    else:
        r = G.gradcam(net, x, t)
        w = r["weights"]
    top = np.argsort(-w)[:5]
    return ", ".join(f"c{int(c)} ({w[c]:.3g})" for c in top)


def main():
    import gradio as gr

    samples = {s["id"]: s for s in X.samples(N_IDS)}
    state = {}

    def load(name, vid, upload):
        from PIL import Image

        from yolo.infer.pipeline import preprocess

        if upload is not None:
            x, _ = preprocess(Image.fromarray(upload), X.SIZE, "stretch")
            x, s = x[None], None
        else:
            s = samples[vid]
            x, _ = X.load_image(s["image"])
        tg = _targets(name, s, x)
        state.update(x=x, s=s, targets=dict(tg), name=name)
        labels = [k for k, _ in tg]
        return (gr.update(choices=labels, value=labels[0] if labels else None),
                (np.asarray(x[0]).transpose(1, 2, 0) * 255).astype(np.uint8),
                f"{len(labels)} targets")

    def show(label, method, kind):
        if not label or "x" not in state:
            return None, "load an image first"
        name, x, t = state["name"], state["x"], state["targets"][label]
        if method == "lrp_int":
            kind = "int"
        if kind == "golden" and method not in ("occlusion", "drise"):
            return None, "golden stage: occlusion and drise only (black box)"
        r = X.make_map(name, method, kind, x, t)
        info = [f"{method} @ {kind}: {r['passes']} passes, {r['seconds']:.1f} s",
                f"top channels of the head input: {_channels(name, method, x, t)}"]
        if t.kind == "miss" and state["s"] is not None:
            info.insert(0, "miss cause: " + _cause(name, state["s"], t, x))
        return _render(x, r["map"], t.box), "\n\n".join(info)

    with gr.Blocks(title="M36 explanation viewer") as app:
        gr.Markdown("## Why this box, why this miss (M36)")
        with gr.Row():
            name = gr.Dropdown(list(X.NETS), value=X.NETS[0], label="network")
            vid = gr.Dropdown(list(samples), value=next(iter(samples)), label="VOC test image")
            upload = gr.Image(label="or upload an image", type="numpy")
            go = gr.Button("load")
        with gr.Row():
            target = gr.Dropdown([], label="target (detection or ground truth)")
            method = gr.Dropdown(list(METHODS), value="gradcam", label="method")
            kind = gr.Dropdown(list(STAGES), value="float", label="stage")
            run = gr.Button("explain")
        with gr.Row():
            image = gr.Image(label="image", interactive=False)
            out = gr.Image(label="map", interactive=False)
        info = gr.Markdown()
        go.click(load, [name, vid, upload], [target, image, info])
        run.click(show, [target, method, kind], [out, info])
    app.launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
