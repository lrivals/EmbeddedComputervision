"""T39.1, T39.2, T39.4, T39.5: learned failure monitor on the labels of T39.0.

    python tools/failure_prediction.py fit --labels build/m39/labels/<model>_<set>.npz
    python tools/failure_prediction.py eval --labels …            # T29.1 protocol, test split
    python tools/failure_prediction.py eval --labels <labels of the joint detector> \
        --joint <checkpoint.npz>                                   # T39.2 head next to T39.1
    python tools/failure_prediction.py calib --labels … [--smoothed <labels>]   # T39.4
    python tools/failure_prediction.py stages --labels … [--check] # T39.5 float / int / C-sim

`fit`: one head per target (`miss`, `tcp`) and stage (`float` features, or the integer
features dequantized), trained on the fit split. The imbalance strategy (positive weight or
balanced batches) is the one of best AUROC on the val split; the test split is never seen.
Heads in `build/m39/heads/<labels>_<stage>_<target>.npz`.

`eval`: on the test split, for the max score (the best post-hoc monitor of T29.0 on ExDark)
and each head: AUROC of "this image has a missed person", share of the missed persons
flagged at 1 % and 5 % alarm rate, recall of "detector or alarm" against the detector
alone, the KGFP metric (failing images flagged at 5 % false alarms), memory and MACs. The
other monitors of T29.0 are in `tools/ood_monitor.py compare --labels`.

`stages`: the miss head of the float stage, at float (float features), integer (the
`IntFailureHead` on the integer features) and C-sim (`hls/tb/tb_failure.cpp`, g++); flag at
5 % alarm rate (threshold chosen on val per stage); the test images whose flag changes
between stages, with the causes of their misses; `--check`: integer == C-sim on every image.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from failure_labels import OUT, Labels, pct, write_section  # noqa: E402
from yolo.infer import failure as F  # noqa: E402

HEADS = OUT / "heads"
EVAL = OUT / "eval"
STAGES = ("float", "int")
TARGETS = ("miss", "tcp")


def head_path(L, stage, target):
    return HEADS / f"{L.name}_{stage}_{target}.npz"


def targets_of(L, stage, target):
    miss = getattr(L, f"miss_{stage}")
    return (miss.astype(float) if target == "miss" else getattr(L, f"tcp_{stage}")), miss


def metrics(L, stage, scores, split="test"):
    m, pm, pimg = L.select(split)
    fails = getattr(L, f"miss_{stage}")[m]
    out = F.monitor_metrics(scores[m], fails, pimg, L.missed(stage)[pm])
    # Images without a person never fail: AUROC among the images with persons too.
    hp = L.n_persons[m] > 0
    sp, fp = scores[m][hp], fails[hp]
    out["auroc_persons"] = (F.ood.auroc(sp[~fp], sp[fp]) if fp.any() and (~fp).any()
                            else float("nan"))
    (out["kgfp_flagged"], out["kgfp_accepted"],
     out["kgfp_recall"]) = F.recall_at_false_alarm(scores[m], fails, 0.05, pimg,
                                                   L.missed(stage)[pm])
    return out


# ------------------------------------------------------------------------------ fit
def cmd_fit(args):
    L = Labels(args.labels)
    HEADS.mkdir(parents=True, exist_ok=True)
    fit, val = L.split == 0, L.split == 1
    for stage in args.stages.split(","):
        x = L.features(stage)
        for target in args.targets.split(","):
            y, miss = targets_of(L, stage, target)
            best = None
            for imb in ("weight", "balanced"):
                h = F.FailureHead(x.shape[1], args.hidden, target, seed=args.seed)
                h.fit(x[fit], y[fit], pos=miss[fit], imbalance=imb, iters=args.iters,
                      batch=args.batch, lr=args.lr, seed=args.seed, std_floor=L.scales)
                s = h.score(x)
                auc = F.ood.auroc(s[val & ~miss], s[val & miss])
                print(f"{L.name} {stage} {target} {imb}: val AUROC {auc:.3f}")
                if best is None or auc > best[0]:
                    best = (auc, imb, h)
            auc, imb, h = best
            np.savez(head_path(L, stage, target), **h.arrays(),
                     meta=np.array(json.dumps({"imbalance": imb, "val_auroc": auc,
                                               "hidden": args.hidden, "iters": args.iters,
                                               "labels": L.name, "stage": stage})))
            print(f"→ {head_path(L, stage, target)} ({imb})")


def load_head(path):
    with np.load(path) as a:
        return F.FailureHead.from_arrays(a), json.loads(str(a["meta"]))


# ----------------------------------------------------------------------------- eval
def eval_rows(L, joint=None):
    rows = []
    for stage in STAGES:
        rows.append({"monitor": "max_score", "stage": stage, "memory": 0, "macs": 0,
                     **metrics(L, stage, 1.0 - getattr(L, f"top_{stage}"))})
        for target in TARGETS:
            p = head_path(L, stage, target)
            if not p.exists():
                continue
            h, meta = load_head(p)
            rows.append({"monitor": f"head_{target}", "stage": stage, "memory": h.memory_bytes,
                         "macs": h.macs, "imbalance": meta["imbalance"],
                         **metrics(L, stage, h.score(L.features(stage)))})
    if joint is not None:
        with np.load(joint) as a:
            h = F.FailureHead.from_arrays({k[len("failure/"):]: a[k] for k in a.files
                                           if k.startswith("failure/")})
        rows.append({"monitor": "head_joint", "stage": "float", "memory": h.memory_bytes,
                     "macs": h.macs, **metrics(L, "float", h.score(L.features("float")))})
    return rows


def eval_markdown(rows):
    lines = ["| Monitor | Stage | AUROC | AUROC, images with persons | Missed flagged @1 % | @5 % | System recall @1 % "
             "| @5 % | Detector recall | Person recall accepted @5 % FPR | Accepted | Memory "
             "| MACs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['monitor']} | {r['stage']} | {r['auroc']:.3f} | "
                     f"{r['auroc_persons']:.3f} | "
                     f"{pct(r['flagged_1'])} | {pct(r['flagged_5'])} | {pct(r['system_1'])} | "
                     f"{pct(r['system_5'])} | {pct(r['detector_recall'])} | "
                     f"{pct(r['kgfp_recall'])} | {pct(r['kgfp_accepted'])} | {r['memory']:,} B | "
                     f"{r['macs']:,} |")
    return "\n".join(lines)


def cmd_eval(args):
    EVAL.mkdir(parents=True, exist_ok=True)
    bodies = []
    for path in args.labels:
        L = Labels(path)
        rows = eval_rows(L, args.joint)
        (EVAL / f"{L.name}.json").write_text(json.dumps(rows, indent=1) + "\n")
        n = int((L.split == 2).sum())
        bodies.append(f"**{L.name}** (test split, {n} images, threshold {L.meta['conf']}):\n\n"
                      + eval_markdown(rows))
        print(bodies[-1])
    if args.write:
        write_section("joint" if args.joint else "head",
                      "Joint training (T39.2)" if args.joint else "Post-hoc learned monitor "
                      "(T39.1)", HEAD_NOTE + "\n\n" + "\n\n".join(bodies))


HEAD_NOTE = """Heads: MLP d → 32 (ReLU) → 1 on the pooled features of the two head inputs (mean
and max per channel, d = 1,536 for Tiny-YOLOv3), trained on the fit split; imbalance
strategy chosen on val; numbers on test. `max_score`: 1 − the top person score, the best
post-hoc monitor of T29.0 on ExDark. Alarm rates are over all test images; "system recall":
persons detected or in an image that raises an alarm. The KGFP metric: threshold rejecting 5 %
of the failure-free images, person recall among the accepted images (against the detector
recall over all images), and the share of images accepted."""


# ---------------------------------------------------------------------------- calib
def calib_rows(L, tag):
    from yolo.infer import confidence as CF

    rows = []
    for stage in STAGES:
        d_img = getattr(L, f"d_image_{stage}")
        sc, ok = getattr(L, f"d_score_{stage}"), getattr(L, f"d_correct_{stage}")
        fit, test = L.split[d_img] == 0, L.split[d_img] == 2
        t = CF.fit_temperature(sc[fit], ok[fit])
        m = L.split == 2
        fails = getattr(L, f"miss_{stage}")[m]
        top = getattr(L, f"top_{stage}")[m]
        for name, tt in (("none", 1.0), ("temperature", t)):
            conf = CF.apply_temperature(top, tt)
            dsc = CF.apply_temperature(sc[test], tt)
            row = {"model": tag, "stage": stage, "calibration": name, "T": tt,
                   "ece": CF.ece(dsc, ok[test]),
                   "auroc_max": F.ood.auroc((1 - conf)[~fails], (1 - conf)[fails]),
                   "aurc_max": CF.aurc(conf, fails)}
            p = head_path(L, stage, "miss")
            if p.exists():
                s = load_head(p)[0].score(L.features(stage))[m]
                row["auroc_head"] = F.ood.auroc(s[~fails], s[fails])
                row["aurc_head"] = CF.aurc(-s, fails)
            rows.append(row)
    return rows


def cmd_calib(args):
    rows = calib_rows(Labels(args.labels), "base")
    if args.smoothed:
        rows += calib_rows(Labels(args.smoothed), "label smoothing")
    lines = ["| Detector | Stage | Calibration | T | ECE person score | AUROC max score | "
             "AURC max score | AUROC head | AURC head |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['stage']} | {r['calibration']} | {r['T']:.2f} | "
                     f"{r['ece']:.4f} | {r['auroc_max']:.3f} | {r['aurc_max']:.3f} | "
                     f"{r.get('auroc_head', float('nan')):.3f} | "
                     f"{r.get('aurc_head', float('nan')):.3f} |")
    table = "\n".join(lines)
    print(table)
    (OUT / f"calib_{Path(args.labels).stem}.json").write_text(json.dumps(rows, indent=1) + "\n")
    if args.write:
        write_section(f"calib-{Path(args.labels).stem}",
                      f"Calibration against failure prediction (T39.4): {Path(args.labels).stem}",
                      CALIB_NOTE.format(name=Path(args.labels).stem) + "\n\n" + table)


CALIB_NOTE = """`{name}`. Temperature fitted by NLL on the person detections (down to 0.005) of
the fit split, s' = σ(logit(s)/T); ECE (15 bins) on the test split; AUROC and AURC (lower is
better) of the failure prediction on the test images. Temperature scaling is monotone: the
max-score monitor keeps its ranking, so its AUROC and AURC cannot move, and the head does
not read the score. The question of Zhu 2024 is open until the label-smoothing fine-tune
(`tools/train.py --label-smoothing`) gives its labels (`--smoothed`)."""


# --------------------------------------------------------------------------- stages
def build_tb(dst):
    exe = Path(dst) / "tb_failure"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(ROOT / "hls" / "kernels"),
                    str(ROOT / "hls" / "tb" / "tb_failure.cpp"), "-o", str(exe)], check=True)
    return exe


def csim_scores(ih, p):
    if shutil.which("g++") is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        exe = build_tb(d)
        blob = Path(d) / "head.bin"
        blob.write_bytes(ih.blob(p))
        out = subprocess.run([str(exe), str(blob), str(ih.sign)], check=True,
                             capture_output=True, text=True).stdout.split()
    return np.array(out, np.int64)


def cmd_stages(args):
    from yolo.quant.int_failure import IntFailureHead

    L = Labels(args.labels)
    h, _ = load_head(head_path(L, "float", "miss"))
    p = L.feat_int.astype(np.int64)
    ih = IntFailureHead.from_float(h, L.scales, p[L.split == 0])
    scores = {"float": h.score(L.features("float")), "int": ih.score(p).astype(np.float64)}
    cs = csim_scores(ih, p)
    if cs is not None:
        scores["csim"] = cs.astype(np.float64)
    equal = None if cs is None else bool(np.array_equal(cs, ih.score(p)))
    val, test = L.split == 1, L.split == 2
    rows, flags = [], {}
    for st, s in scores.items():
        lab = "float" if st == "float" else "int"
        thr = F.alarm_threshold(s[val], 0.05)
        flags[st] = s > thr
        r = metrics(L, lab, s)
        rows.append(f"| {st} | {r['auroc']:.3f} | {pct(r['flagged_5'])} | "
                    f"{pct(flags[st][test].mean())} |")
    changed = []
    ids = np.flatnonzero(test)
    for k in ids[flags["float"][ids] != flags["int"][ids]]:
        causes = sorted({str(c) for c in L.p_cause_int[L.p_image == k] if c})
        changed.append(f"| {L.ids[k]} | {int(flags['float'][k])} → {int(flags['int'][k])} | "
                       f"{int(getattr(L, 'miss_float')[k])} / {int(L.miss_int[k])} | "
                       f"{', '.join(causes) or '–'} |")
    res = {"labels": L.name, "int_equals_csim": equal, "changed": len(changed),
           "memory_int": ih.memory_bytes, "memory_float": h.memory_bytes, "macs": ih.macs}
    (OUT / f"stages_{L.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    body = (f"`{L.name}`, miss head of the float stage. Integer head: zero point and "
            f"standardization folded, int8 weights, int32 accumulators, hidden layer "
            f"requantized as a conv (§9.3); {ih.memory_bytes:,} B against {h.memory_bytes:,} B "
            f"in float32, {ih.macs:,} MACs. Flag at 5 % alarm rate, threshold chosen on val per "
            f"stage; AUROC and missed persons flagged on test (at the integer stage against "
            f"the integer labels). Integer == C-sim on every image: **{equal}**.\n\n"
            "| Stage | AUROC | Missed flagged @5 % | Alarm rate on test |\n|---|---|---|---|\n"
            + "\n".join(rows)
            + f"\n\nTest images whose flag changes between float and integer: {len(changed)}.\n")
    if changed:
        body += ("\n| Image | Flag float → int | Fails float / int | Causes (int) |\n"
                 "|---|---|---|---|\n" + "\n".join(changed[:args.max_list]))
        if len(changed) > args.max_list:
            body += f"\n\n({len(changed) - args.max_list} more in the JSON.)"
    print(body)
    if args.write:
        write_section(f"stages-{L.name}", f"Failure monitor at the three stages (T39.5): "
                      f"{L.name}", body)
    if args.check and equal is not True:
        sys.exit("integer and C-sim scores differ (or g++ absent)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--stages", default="float,int")
    p.add_argument("--targets", default="miss,tcp")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("--labels", type=Path, nargs="+", required=True)
    p.add_argument("--joint", type=Path, default=None, help="Trainer checkpoint (T39.2)")
    p.add_argument("--write", action="store_true", help="section of results/")
    p = sub.add_parser("calib")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--smoothed", type=Path, default=None,
                   help="labels of the label-smoothing fine-tune")
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("stages")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--max-list", type=int, default=20)
    p.add_argument("--write", action="store_true")
    args = ap.parse_args()
    {"fit": cmd_fit, "eval": cmd_eval, "calib": cmd_calib, "stages": cmd_stages}[args.cmd](args)


if __name__ == "__main__":
    main()
