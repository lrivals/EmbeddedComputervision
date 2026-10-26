"""T39.3: KGFP baseline (Zimmermann, Holzbach, Lerch, "Knowledge-Guided Failure Prediction:
Detecting When Object Detectors Miss Safety-Critical Objects", CVPR 2026 SAIAD workshop,
arXiv 2603.25499), next to the learned heads of T39.1 and T39.2.

    python tools/kgfp_baseline.py cost                      # DINO against the failure head
    python tools/kgfp_baseline.py metric --labels build/m39/labels/<model>_coco-val2017.npz \
        [--scores <kgfp scores .csv>]                        # KGFP metric for every monitor

KGFP compares the detector's features with those of a frozen DINO ViT (dual encoder,
angular distance) and flags the images where a person is missed. Its metric: the threshold
rejects 5 % of the images without a miss (false alarms), and the person recall is measured
among the accepted images (64.3 % → 84.5 % for YOLOv8 on COCO persons in the paper).

The published code, models and features are on the Fraunhofer GitLab
(gitlab.cc-asp.fraunhofer.de/iosb_public/KGFP, per the paper). It runs on a GPU and is
trained for YOLOv8 features, so this wrapper does not import it: the KGFP run is done in its
own environment, on the same COCO person images (the `ids` of the labels file), and exported
as a CSV `image_id,score` (high = alarm); `metric --scores` scores it on our labels. Limits
of the comparison, stated in the results: KGFP watches another detector (YOLOv8 against
Tiny-YOLOv3), so the failing images are not the same; only the metric and the cost compare
directly until KGFP is retrained on Tiny-YOLOv3 features.

`cost`: parameters and MACs of the DINO encoder (ViT-B/16 at 224 × 224, counted from the
architecture: 12 blocks, width 768, MLP 3,072, 197 tokens) against the failure head.
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from failure_labels import Labels, pct, write_section  # noqa: E402
from yolo.infer import failure as F  # noqa: E402

KGFP_REPO = "gitlab.cc-asp.fraunhofer.de/iosb_public/KGFP"
PAPER = "arXiv 2603.25499"


def vit_cost(depth=12, width=768, mlp=3072, image=224, patch=16, channels=3):
    """(parameters, MACs) of a ViT encoder (patch embedding, blocks, final norm; no head)."""
    tokens = (image // patch) ** 2 + 1
    embed_p = channels * patch * patch * width + width + tokens * width + width  # + cls
    embed_m = (tokens - 1) * channels * patch * patch * width
    block_p = (4 * width * width + 4 * width          # qkv + projection
               + 2 * width * mlp + mlp + width        # MLP
               + 4 * width)                           # two layer norms
    block_m = (4 * tokens * width * width             # qkv + projection
               + 2 * tokens * tokens * width          # QKᵀ and attention × V
               + 2 * tokens * width * mlp)            # MLP
    return embed_p + depth * block_p + 2 * width, embed_m + depth * block_m


def cmd_cost(args):
    p, m = vit_cost()
    head = F.FailureHead(1536, 32)
    rows = [f"| DINO ViT-B/16 (KGFP's vision model) | {p / 1e6:.1f} M | {m / 1e9:.2f} G | "
            f"{4 * p / 2**20:.0f} MiB (float32) |",
            f"| Failure head (T39.1, float32) | {head.memory_bytes // 4:,} | {head.macs:,} | "
            f"{head.memory_bytes / 1024:.0f} KiB |",
            "| Failure head (T39.5, integer) | same | same | ≈ 50 KiB (int8 weights) |"]
    body = ("| Model | Parameters | MACs per image | Weights |\n|---|---|---|---|\n"
            + "\n".join(rows)
            + f"\n\nRatio: {p / (head.memory_bytes // 4):,.0f}× the parameters, "
            f"{m / head.macs:,.0f}× the MACs. KGFP also runs its own feature encoder on the "
            "detector features (not counted here); the failure head reuses maps the detector "
            "already computes.")
    print(body)
    if args.write:
        write_section("kgfp-cost", "KGFP baseline: cost (T39.3)", body)


def read_scores(path, ids):
    with open(path) as f:
        s = {r["image_id"]: float(r["score"]) for r in csv.DictReader(f)}
    missing = [i for i in ids if str(i) not in s]
    if missing:
        sys.exit(f"{path}: {len(missing)} images without a score (e.g. {missing[:3]})")
    return np.array([s[str(i)] for i in ids])


def cmd_metric(args):
    from failure_prediction import head_path, load_head

    L = Labels(args.labels)
    stage = args.stage
    m, pm, pimg = L.select(args.split)
    fails = getattr(L, f"miss_{stage}")[m]
    missed = L.missed(stage)[pm]
    monitors = {"max_score": 1.0 - getattr(L, f"top_{stage}")}
    for target in ("miss", "tcp"):
        p = head_path(L, stage, target)
        if p.exists():
            monitors[f"head_{target} (T39.1)"] = load_head(p)[0].score(L.features(stage))
    if args.joint:
        with np.load(args.joint) as a:
            h = F.FailureHead.from_arrays({k[len("failure/"):]: a[k] for k in a.files
                                           if k.startswith("failure/")})
        monitors["head_joint (T39.2)"] = h.score(L.features("float"))
    if args.scores:
        monitors["KGFP"] = read_scores(args.scores, L.ids)
    rows = []
    for name, s in monitors.items():
        flagged, accepted, rec = F.recall_at_false_alarm(s[m], fails, 0.05, pimg, missed)
        rows.append(f"| {name} | {pct(rec)} | {pct(accepted)} | {pct(flagged)} |")
    base = float((~missed).mean())
    body = (f"`{L.name}`, {stage} stage, {args.split} split ({int(m.sum())} images), detector "
            f"person recall over all images {pct(base)}. Threshold rejecting 5 % of the "
            "failure-free images.\n\n"
            "| Monitor | Person recall among accepted images | Images accepted "
            "| Failing images rejected |\n|---|---|---|---|\n" + "\n".join(rows))
    if not args.scores:
        body += (f"\n\nKGFP itself is not run here: its code ({KGFP_REPO}, {PAPER}) needs a "
                 "GPU and YOLOv8 features. Paper figure, for YOLOv8 on COCO persons: "
                 "64.3 % → 84.5 %, a different detector and different failing images.")
    print(body)
    if args.write:
        write_section(f"kgfp-{L.name}", f"KGFP metric (T39.3): {L.name}", body)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cost")
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("metric")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--stage", choices=("float", "int"), default="float")
    p.add_argument("--split", choices=("fit", "val", "test"), default="test")
    p.add_argument("--scores", type=Path, default=None, help="KGFP scores, image_id,score")
    p.add_argument("--joint", type=Path, default=None, help="T39.2 checkpoint")
    p.add_argument("--write", action="store_true")
    args = ap.parse_args()
    {"cost": cmd_cost, "metric": cmd_metric}[args.cmd](args)


if __name__ == "__main__":
    main()
