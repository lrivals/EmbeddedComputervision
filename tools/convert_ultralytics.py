"""Converter of YOLOv5n / YOLOv8n PyTorch checkpoints to NumPy, without torch (T26.1, T33.0).

    python -I tools/convert_ultralytics.py --weights weights/ultralytics/yolov8n.pt --out weights/
    python tools/convert_ultralytics.py --net yolov5n --random --out build/m26/   # test

The `.pt` file is read by `yolo.io.torch_pickle` (whitelisted unpickling, no torch, no
`ultralytics`, no yolov5 repository); the network description comes from the model YAML
stored in the checkpoint (`yolo.models.ul_yaml`), or from `--net` for a plain state dict.
Output: `<out>/<net>.npz` (BN folded into the convs, float32) and `<out>/<net>.json` (flat
layer list, run by `python/yolo/models/ul_graph.py`); anchors in pixels per level.

A `.pt` file from the web is untrusted data (pickle): keep it in its own download folder and
run this tool with `python -I`.

**Licence**: the Ultralytics weights and code are AGPL-3.0. The converted arrays are a
derived work of the weights: check the licence before storing them on Drive or showing them
in the article (docs/tasks/M26-recent-models.md).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.io.torch_pickle import load_checkpoint  # noqa: E402
from yolo.models import ul_graph, ul_net, ul_yaml  # noqa: E402


def describe(net, cfg=None, nc=None, anchors=None):
    """Unfolded description; `cfg`: model YAML dict (default: `net`'s), `anchors` in pixels."""
    desc = ul_yaml.parse(cfg or ul_yaml.CONFIGS[net], nc=nc, name=net)
    if anchors is not None and "anchors" in desc:
        desc["anchors"] = anchors
    return desc


def load(path, net=None, nc=None):
    """(unfolded description, state dict) of a `.pt` file; `net` read from its YAML."""
    sd, meta = load_checkpoint(path)
    cfg = meta.get("yaml")
    if net is None:
        if not cfg:
            raise SystemExit(f"{path}: no model YAML in the file, give --net")
        net = "yolov5n" if "anchors" in cfg else "yolov8n"
    return describe(net, cfg, nc, meta.get("anchors")), sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", choices=sorted(ul_yaml.CONFIGS),
                    help="architecture (default: read from the checkpoint)")
    ap.add_argument("--weights", type=Path, help="checkpoint or state dict (.pt)")
    ap.add_argument("--random", action="store_true", help="random weights (tests)")
    ap.add_argument("--classes", type=int)
    ap.add_argument("--out", type=Path, default=ROOT / "weights")
    ap.add_argument("--check", type=int, default=1,
                    help="images to compare the folded graph with the unfolded network")
    args = ap.parse_args()
    if args.weights:
        full, sd = load(args.weights, args.net, args.classes)
    elif args.random and args.net:
        full = describe(args.net, nc=args.classes)
        sd = ul_net.random_state_dict(full)
    else:
        ap.error("--weights, or --random with --net")
    desc, w = ul_net.fold(full, sd)
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / desc["name"]
    np.savez(base.with_suffix(".npz"), **w)
    base.with_suffix(".json").write_text(json.dumps(desc, indent=1) + "\n")
    print(f"{base}.npz / .json: {len(desc['layers'])} layers, "
          f"{sum(v.size for v in w.values()):,} parameters")
    if args.check:
        x = np.random.default_rng(1).random((args.check, 3, 640, 640))
        want = ul_net.heads(ul_net.build(full, sd), x)
        got = ul_graph.forward(desc, w, x)
        err = max(float(np.abs(a - b).max()) for a, b in zip(got, want))
        print(f"folded vs unfolded, {args.check} image(s) 640×640: max abs error {err:.2e}")


if __name__ == "__main__":
    main()
