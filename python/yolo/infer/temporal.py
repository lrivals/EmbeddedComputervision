"""Temporal vote on the person detections (T29.3): k-of-n persistence per track, simple IoU
association, no Re-ID.

A track is a run of detections of consecutive frames linked by IoU (greedy, best overlap
first, ≥ `iou`). At each frame a track is **reported** when it was detected in at least `k`
of its last `n` frames, with the box of its last detection (held through the frames where it
is not detected). A track not detected in its last `n` frames is dropped.

- `k = 1, n > 1`: a hold. A person missed for up to n − 1 frames stays reported, and a
  false alarm stays n − 1 frames too.
- `k = n`: a confirmation. A detection is reported only after k frames in a row, which
  removes isolated false alarms and delays every new person by k − 1 frames.
- `k = n = 1`: the detector alone (each frame's detections, unchanged).

The vote runs on one contiguous run of frames (`dreaded_event.contiguous_runs`). A gap ends
every track.
"""

import numpy as np


def iou_matrix(a, b):
    """IoU (len(a), len(b)) of VOC corners (pixel + 1 convention, as `metrics._overlaps`)."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    area = lambda x: (x[:, 2] - x[:, 0] + 1) * (x[:, 3] - x[:, 1] + 1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def greedy_pairs(ov, thr):
    """Pairs (i, j) of the greedy assignment on the overlap matrix `ov`, best first, ≥ thr."""
    pairs = []
    if ov.size == 0:
        return pairs
    ov = ov.copy()
    while True:
        i, j = np.unravel_index(int(np.argmax(ov)), ov.shape)
        if ov[i, j] < thr:
            return pairs
        pairs.append((int(i), int(j)))
        ov[i, :] = -1.0
        ov[:, j] = -1.0


def vote(frames, k, n, iou=0.3):
    """k-of-n vote over one contiguous run. `frames`: list of (boxes (m, 4), scores (m,)) of
    the detections kept at the threshold. Returns per frame (boxes (r, 4), scores (r,)) of
    the reported tracks; the score is the one of the track's last detection."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 ≤ k ≤ n, got k={k}, n={n}")
    tracks = []  # [box, score, hits (list of bool, newest last)]
    out = []
    for boxes, scores in frames:
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64)
        ov = iou_matrix([t[0] for t in tracks], boxes) if tracks and len(boxes) else \
            np.zeros((len(tracks), len(boxes)))
        pairs = greedy_pairs(ov, iou)
        hit, matched = {j for _, j in pairs}, {i for i, _ in pairs}
        for i, j in pairs:
            tracks[i][0], tracks[i][1] = boxes[j], scores[j]
            tracks[i][2].append(True)
        for i, t in enumerate(tracks):
            if i not in matched:
                t[2].append(False)
            del t[2][:-n]
        tracks = [t for t in tracks if any(t[2])]
        tracks += [[boxes[j], scores[j], [True]] for j in range(len(boxes)) if j not in hit]
        rep = [t for t in tracks if sum(t[2]) >= k]
        out.append((np.array([t[0] for t in rep]).reshape(-1, 4),
                    np.array([t[1] for t in rep], np.float64)))
    return out
