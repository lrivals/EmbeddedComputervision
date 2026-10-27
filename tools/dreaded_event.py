"""Dreaded event on video (T27.4): a person present and not reported during N consecutive
frames, per hour of video, against the per-frame miss rate.

    python tools/dreaded_event.py --dataset prewid --split test --net tiny-yolov4-coco \
        --fps 5 --write
    python tools/dreaded_event.py --dataset mot17 --split train --net tiny-yolov4-coco --write

Reads the detections of `tools/person_safety.py infer` (one JSONL per stage). At the
operating point (lowest threshold with FPPI ≤ `--fppi` on the same images, or `--thr`), a
ground truth is reported when a detection kept above the threshold matched it (VOC rule, as
in `yolo.infer.metrics.match_class`). Persons under `--min-height` pixels and `difficult`
ones are left out.

- **track mode** (objects carry `track`, MOT17): an episode is a run of consecutive frames
  where one person is present; a miss run is a run of consecutive frames of an episode where
  that person is not reported;
- **frame mode** (no identities, PREWID): an episode is a run of consecutive frames with at
  least one person; a frame is missed when one of its persons is not reported.

A gap in the frame numbers ends every run. An event of order N is a miss run of length
≥ N (counted once, whatever its length). The rate per hour divides by the duration of the
video (all frames, `--fps`; MOT17 gives its own). Next to it, the rate an independent
per-frame miss with the measured probability p would give: an episode of T frames holds
p^N·(1 + (T − N)(1 − p)) runs of length ≥ N on average. Measured ≫ independent means the
misses come in bursts (the same worker, the same pose, the same occluder).

Output: `build/m27/<dataset>-<split>/<net>/events_<stage>.json` (input of
`person_safety.py deel` and `unflagged --events`: per episode its images and miss runs), and
with `--write` a section of `results/person_safety.md`.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import person_safety as P  # noqa: E402
from yolo.data import datasets  # noqa: E402
from yolo.infer import metrics as M  # noqa: E402
from yolo.models.tiny_yolo import load_cfg  # noqa: E402

MAX_N = 10
MIN_HEIGHT = 30.0  # px: the "far" Caltech bin is left out by default


def sequence_key(s):
    return str(s.get("sequence", s.get("camera", "all")))


def contiguous_runs(samples):
    """Lists of samples of the same sequence with consecutive frame numbers."""
    groups = {}
    for s in sorted(samples, key=lambda s: (sequence_key(s), s.get("frame", 0), s["id"])):
        groups.setdefault(sequence_key(s), []).append(s)
    runs = []
    for g in groups.values():
        cur = [g[0]]
        for s in g[1:]:
            if "frame" in s and s["frame"] == cur[-1].get("frame", -2) + 1:
                cur.append(s)
            else:
                runs.append(cur)
                cur = [s]
        runs.append(cur)
    return runs


def eligible(sample, min_height):
    """Mask of the persons counted: not `difficult`, full box at least `min_height` px."""
    return ~np.asarray(sample["difficult"], bool) & (M.gt_heights(sample) >= min_height)


def episodes(runs, reported, min_height, track_mode):
    """[(images, missed flags)] per episode; `reported[id]` (k,) bool per ground truth."""
    out = []
    for run in runs:
        if track_mode:
            open_ = {}
            for s in run:
                keep = eligible(s, min_height)
                tracks = np.asarray(s["track"])[keep]
                rep = reported[s["id"]][keep]
                seen = set()
                for t, r in zip(tracks.tolist(), rep.tolist()):
                    e = open_.setdefault(t, ([], []))
                    e[0].append(s["id"])
                    e[1].append(not r)
                    seen.add(t)
                for t in [t for t in open_ if t not in seen]:  # the person left: episode ends
                    out.append(open_.pop(t))
            out += open_.values()
        else:
            cur = ([], [])
            for s in run:
                keep = eligible(s, min_height)
                if keep.any():
                    cur[0].append(s["id"])
                    cur[1].append(bool((~reported[s["id"]][keep]).any()))
                elif cur[0]:
                    out.append(cur)
                    cur = ([], [])
            if cur[0]:
                out.append(cur)
    return out


def miss_runs(images, missed):
    """[(start, length)] of the maximal runs of True in `missed`."""
    out, start = [], None
    for i, m in enumerate(list(missed) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - start))
            start = None
    return out


def expected_runs(T, p, n):
    """Mean number of maximal runs of length ≥ n in T independent Bernoulli(p) frames."""
    return p ** n * (1 + (T - n) * (1 - p)) if T >= n else 0.0


def count_events(eps, max_n=MAX_N):
    """{n: [events]}, {n: episodes ≥ n frames}, (missed units, units) of `episodes`."""
    events = {n: [] for n in range(1, max_n + 1)}
    dangerous = {n: 0 for n in range(1, max_n + 1)}
    k = u = 0
    for images, missed in eps:
        k, u = k + sum(missed), u + len(missed)
        for n in dangerous:
            dangerous[n] += len(missed) >= n
        for a, L in miss_runs(images, missed):
            for n in range(1, min(L, max_n) + 1):
                events[n].append({"images": images[a:a + L], "length": L})
    return events, dangerous, (k, u)


def analyse(samples, dets, det_class, thr, fps, min_height=MIN_HEIGHT, max_n=MAX_N,
            reported=None):
    """Event statistics of one stage at threshold `thr` (samples reduced to class 0);
    `reported` ({id: (k,) bool}) replaces the detector's own (the temporal vote, T29.3)."""
    if reported is None:
        gts = M.class_gts(samples, 0)
        ids, scores, boxes = P.person_dets(dets, samples, det_class)
        _, _, _, gt_score = M.match_class(ids, scores, boxes, gts)
        reported = {i: np.nan_to_num(v, nan=np.inf) >= thr for i, v in gt_score.items()}
    track_mode = bool(samples) and all("track" in s for s in samples)
    runs = contiguous_runs(samples)
    eps = episodes(runs, reported, min_height, track_mode)
    events, dangerous, (k, u) = count_events(eps, max_n)
    p = k / u if u else 0.0
    # Each frame lasts 1 / its own rate (MOT17: 14, 25 or 30 frames/s per sequence).
    hours = sum(1.0 / s.get("fps", fps) for s in samples) / 3600.0
    present = sum(1 for s in samples if eligible(s, min_height).any())
    return {
        "mode": "track" if track_mode else "frame", "thr": thr, "fps": fps,
        "frames": len(samples), "hours": hours, "presence": present / max(len(samples), 1),
        "frame_fn_rate": p, "missed_units": k, "units": u, "episodes": len(eps),
        "runs": [{"sequence": sequence_key(r[0]), "first": r[0].get("frame"),
                  "last": r[-1].get("frame"), "frames": len(r)} for r in runs],
        "counts": {str(n): len(v) for n, v in events.items()},
        "per_hour": {str(n): len(v) / hours for n, v in events.items()},
        "independent_per_hour": {str(n): sum(expected_runs(len(m), p, n) for _, m in eps)
                                 / hours for n in events},
        "dangerous": {str(n): v for n, v in dangerous.items()},
        "events": {str(n): v for n, v in events.items()},
        "episode_list": [{"images": images, "runs": miss_runs(images, missed)}
                         for images, missed in eps],
    }


# --------------------------------------------------------------- temporal vote (T29.3)
def voted(samples, dets, det_class, thr, k, n, iou=0.3):
    """k-of-n vote of the detections ≥ `thr`, run by run: ({id: (k,) bool reported per
    ground truth}, false positives per image)."""
    from yolo.infer import temporal as TV

    ids, scores, boxes = [], [], []
    for run in contiguous_runs(samples):
        frames = []
        for s in run:
            b, sc, lab = dets[s["id"]]
            m = (lab == det_class) & (sc >= thr)
            frames.append((b[m], sc[m]))
        for s, (b, sc) in zip(run, TV.vote(frames, k, n, iou)):
            ids += [s["id"]] * len(sc)
            scores.append(sc)
            boxes.append(b)
    scores = np.concatenate(scores) if scores else np.zeros(0)
    boxes = np.concatenate(boxes).reshape(-1, 4) if boxes else np.zeros((0, 4))
    _, _, fp, gt_score = M.match_class(ids, scores, boxes, M.class_gts(samples, 0), 0.5,
                                       M.ignore_regions(samples))
    reported = {i: ~np.isneginf(v) for i, v in gt_score.items()}  # NaN: difficult
    return reported, float(fp.sum()) / max(len(samples), 1)


def first_reported(eps):
    """Index of the first reported unit of each episode (None if never reported)."""
    return [next((i for i, m in enumerate(missed) if not m), None) for _, missed in eps]


def vote_threshold(samples, dets, det_class, k, n, fppi, lo=0.005, hi=0.99, iters=14):
    """Lowest threshold in [lo, hi] whose FPPI after the vote is ≤ `fppi`: bisection in log
    scale (the FPPI falls as the threshold rises), to a ratio (hi / lo)^(2^−iters) ≈ 1.04 %
    with the defaults."""
    if voted(samples, dets, det_class, lo, k, n)[1] <= fppi:
        return float(lo)
    if voted(samples, dets, det_class, hi, k, n)[1] > fppi:
        return float(hi)
    a, b = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = (a + b) / 2
        if voted(samples, dets, det_class, float(np.exp(mid)), k, n)[1] <= fppi:
            b = mid
        else:
            a = mid
    return float(np.exp(b))


def vote_rows(ev, dets, det_class, thr0, fps, votes, fppi, min_height, stage):
    """Rows of the T29.3 table for one stage: each (k, n) at the T27.4 threshold `thr0` and
    at a threshold re-tuned to `fppi` after the vote."""
    base, _ = voted(ev, dets, det_class, thr0, 1, 1)
    track_mode = bool(ev) and all("track" in s for s in ev)
    eps0 = episodes(contiguous_runs(ev), base, min_height, track_mode)
    first0 = first_reported(eps0)
    rows = []
    for k, n in votes:
        for label, thr in (("T27.4", thr0),
                           (f"{fppi} FPPI", vote_threshold(ev, dets, det_class, k, n, fppi))):
            rep, fp = voted(ev, dets, det_class, thr, k, n)
            r = analyse(ev, dets, det_class, thr, fps, min_height, reported=rep)
            eps = episodes(contiguous_runs(ev), rep, min_height, track_mode)
            lag = [b - a for a, b in zip(first0, first_reported(eps))
                   if a is not None and b is not None]
            rows.append([P.STAGE_NAMES[stage], f"{k}/{n}", label, f"{thr:.3g}", f"{fp:.3f}",
                         f"{100 * r['frame_fn_rate']:.2f}",
                         *(f"{r['counts'][str(m)]} ({r['per_hour'][str(m)]:.3g})"
                           for m in (3, 5, 10)),
                         f"{np.mean(lag):+.2f} / {max(lag, default=0):+d}" if lag else "–"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=P.ALL_DATASETS, default="prewid")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--net", default="tiny-yolov4-coco")
    ap.add_argument("--stages", default="float,int")
    ap.add_argument("--out", type=Path, default=P.BUILD)
    ap.add_argument("--fppi", type=float, default=0.1, help="operating point (per stage)")
    ap.add_argument("--thr", type=float, default=None, help="fixed threshold instead")
    ap.add_argument("--fps", type=float, default=None,
                    help="frame rate (assumption when the set gives none)")
    ap.add_argument("--min-height", type=float, default=MIN_HEIGHT)
    ap.add_argument("--max-n", type=int, default=MAX_N)
    ap.add_argument("--votes", default=None,
                    help="k/n[,k/n…]: temporal vote of T29.3 instead of the T27.4 tables")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    split = args.split or datasets.info(args.dataset).test
    samples = datasets.load(args.dataset, args.data_root, split)
    fps, fps_source, given = args.fps, "argument (assumption)", []
    if fps is None:
        given = sorted({float(s["fps"]) for s in samples if "fps" in s})
        if not samples or any("fps" not in s for s in samples):
            raise SystemExit(f"{args.dataset}: no frame rate in the data, pass --fps "
                             "(written as an assumption)")
        # One rate per sequence (seqinfo.ini): `analyse` sums the frame durations.
        fps, fps_source = (given[0] if len(given) == 1 else None), "seqinfo.ini"
    rows, fn_rate = [], {}
    for st in args.stages.split(","):
        path = P.dets_path(args.dataset, split, args.net, st, None, args.out)
        if not path.exists():
            print(f"{st}: no run ({path})")
            continue
        dets = P.read_dets(path)
        sub = [s for s in samples if s["id"] in dets]
        ev, det_class = P.eval_samples(args.dataset, sub, load_cfg(args.net)["classes"])
        thr = args.thr
        if thr is None:
            m = M.person_metrics(P.person_dets(dets, ev, det_class), ev, fppi_targets=(args.fppi,))
            thr = m["at"][args.fppi]["thr"]
        if args.votes:
            votes = [tuple(int(x) for x in v.split("/")) for v in args.votes.split(",")]
            rows += vote_rows(ev, dets, det_class, thr, fps, votes, args.fppi,
                              args.min_height, st)
            continue
        r = analyse(ev, dets, det_class, thr, fps, args.min_height, args.max_n)
        r.update(dataset=args.dataset, split=split, net=args.net, stage=st,
                 fps_source=fps_source, fppi=None if args.thr is not None else args.fppi,
                 min_height=args.min_height)
        out = P.run_dir(args.dataset, split, args.net, args.out) / f"events_{st}.json"
        out.write_text(json.dumps(r, indent=1) + "\n")
        fn_rate[st] = r["frame_fn_rate"]
        print(f"{st}: thr {thr:.3g}, per-frame miss {r['frame_fn_rate']:.4f} "
              f"({r['missed_units']}/{r['units']}), {r['episodes']} episodes, "
              f"{len(r['runs'])} runs, {r['hours'] * 60:.1f} min → {out}")
        for n in (1, 2, 3, 5, 10):
            if str(n) in r["per_hour"]:
                rows.append([P.STAGE_NAMES[st], n, r["counts"][str(n)], r["dangerous"][str(n)],
                             f"{r['per_hour'][str(n)]:.3g}",
                             f"{r['independent_per_hour'][str(n)]:.3g}",
                             ratio(r["per_hour"][str(n)], r["independent_per_hour"][str(n)])])
        last = r
    if not rows:
        raise SystemExit("no stage to analyse")
    if args.votes:
        return write_votes(args, split, rows)
    unit = "person-frames" if last["mode"] == "track" else "frames with a person"
    body = [f"{args.net} on {args.dataset} `{split}`, {last['mode']} mode, persons ≥ "
            f"{args.min_height:g} px, operating point "
            + (f"{args.fppi} FPPI per stage" if args.thr is None else f"threshold {args.thr}")
            + ". Frame rate **"
            + (f"{fps:g} frames/s" if fps else f"{', '.join(f'{v:g}' for v in given)} frames/s "
               "by sequence")
            + f"** ({fps_source}): {last['frames']} frames = "
            f"{last['hours'] * 60:.1f} min of video in {len(last['runs'])} runs of consecutive "
            "frames (a gap in the frame numbers ends a run); "
            f"presence {100 * last['presence']:.0f} % of the "
            f"frames. Per-frame miss rate over the {unit}: "
            + ", ".join(f"{P.STAGE_NAMES[st]} {100 * v:.2f} %" for st, v in fn_rate.items())
            + ".", "",
            P.table(["Stage", "N", "Events (miss runs ≥ N)", "Episodes ≥ N frames",
                     "Events / h", "Independent / h", "Measured / independent"], rows), "",
            "Runs used: " + ", ".join(f"{r['sequence']} {r['first']}-{r['last']}"
                                      for r in last["runs"] if r["frames"] >= args.max_n)
            + f" (runs of at least {args.max_n} frames)."]
    print("\n".join(body))
    if args.write:
        P.write_section(f"events-{P.section_tag(args.dataset, split)}",
                        f"Dreaded event on video: {args.dataset} {split} (T27.4)",
                        "\n".join(body))


def write_votes(args, split, rows):
    body = [f"{args.net} on {args.dataset} `{split}`, persons ≥ {args.min_height:g} px. "
            "k-of-n vote of `yolo.infer.temporal` (IoU association ≥ 0.3, no Re-ID; a track "
            "is reported when detected in k of its last n frames, with its last box), at the "
            f"T27.4 threshold ({args.fppi} FPPI of the detector alone) and at a threshold "
            f"re-tuned so the voted output stays at {args.fppi} FPPI. 1/1 is the detector "
            "alone. **Events**: miss runs of at least N frames (count, and per hour) as "
            "T27.4. **Added latency**: frames from the first report of the detector alone to "
            "the first report with the vote, per episode (mean / max; a hold can report "
            "earlier when it bridges a gap).", "",
            P.table(["Stage", "k/n", "Threshold of", "Threshold", "FPPI", "Per-frame miss %",
                     "Events N = 3 (/h)", "N = 5 (/h)", "N = 10 (/h)",
                     "Added latency (frames)"], rows)]
    print("\n".join(body))
    if args.write:
        P.write_section(f"vote-{P.section_tag(args.dataset, split)}-{Path(args.net).stem}",
                        f"Temporal vote: {args.net}, {args.dataset} {split} (T29.3)",
                        "\n".join(body))


def ratio(a, b):
    if b == 0:
        return "–" if a == 0 else "∞"
    return f"{a / b:.3g}"


if __name__ == "__main__":
    main()
