"""T30.1: distribution of the post-processing cycles over real images.

Reads the `times_*.csv` of `yolo_bench --hw-post` (sim backend, written by
`tools/m30_cycles.sh`) and gives, per set and over all of them, the spread of the
`yolo_post` cycles, NMS cycles and candidates, against the static bound of T30.0
(`perf_model.postproc_wcet`).

    python tools/m30_cycles.py build/m30/cycles/voc build/m30/cycles/coco ... \
        --net tiny-yolov2-voc [--conf 0.005] [--json out.json]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import perf_model as pm  # noqa: E402

COLS = ("post_cycles", "nms_cycles", "candidates", "survivors", "overflow")


def read_times(d):
    """{column: int array} of every image of the `times_*.csv` in `d`."""
    rows = [r for p in sorted(Path(d).glob("times_*.csv")) for r in csv.DictReader(open(p))]
    rows = [r for r in rows if r.get("post_cycles")]
    return {c: np.array([int(r[c]) for r in rows], dtype=np.int64) for c in COLS}


def spread(v):
    q = np.percentile(v, [50, 99, 99.9]) if len(v) else [0, 0, 0]
    return {"min": int(v.min()), "median": float(q[0]), "p99": float(q[1]),
            "p999": float(q[2]), "max": int(v.max()), "mean": float(v.mean())}


def summary(dirs, net, conf):
    bound = pm.postproc_wcet(ROOT / "model" / net / "manifest.json", conf)
    sets = {Path(d).name: read_times(d) for d in dirs}
    sets["all"] = {c: np.concatenate([s[c] for s in sets.values()]) for c in COLS}
    out = {"net": net, "conf": conf, "bound": bound, "sets": {}}
    for name, s in sets.items():
        out["sets"][name] = {"images": int(len(s["post_cycles"])),
                             "overflow_images": int((s["overflow"] > 0).sum()),
                             **{c: spread(s[c]) for c in COLS[:3]}}
    return out


def markdown(rep):
    b = rep["bound"]
    lines = [f"`{rep['net']}`, threshold {rep['conf']}, static bound {b['total']:,} cycles "
             f"({b['candidates']:,} candidates).", "",
             "| Set | Images | Cycles median | p99 | p99.9 | max | max / bound | "
             "Candidates median | max | NMS cycles max | Overflow images |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rep["sets"].items():
        p, c, n = s["post_cycles"], s["candidates"], s["nms_cycles"]
        lines.append(f"| {name} | {s['images']:,} | {p['median']:,.0f} | {p['p99']:,.0f} | "
                     f"{p['p999']:,.0f} | {p['max']:,} | {p['max'] / b['total']:.2%} | "
                     f"{c['median']:,.0f} | {c['max']:,} | {n['max']:,} | "
                     f"{s['overflow_images']} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--conf", type=float, default=0.005,
                    help="threshold of the run (yolo_bench default 0.005)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    rep = summary(args.dirs, args.net, args.conf)
    print(markdown(rep))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
