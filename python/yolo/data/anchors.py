"""Ancres par k-means, distance 1 − IoU (§5.2, méthode YOLOv2).

Jamais de distance euclidienne : elle favoriserait les grandes boîtes [1612.08242#002.4].
"""

import numpy as np

from yolo.infer.boxes import iou_wh


def kmeans_anchors(wh, k, iters=300, rng=0):
    """Centroïdes (k, 2) triés par aire croissante (conventions.md).

    `wh` : (n, 2) largeurs et hauteurs. Un cluster vide garde son centroïde.
    """
    wh = np.asarray(wh, dtype=np.float64).reshape(-1, 2)
    centroids = _init_plusplus(wh, k, np.random.default_rng(rng))
    for _ in range(iters):
        # §5.2 : affectation au centroïde le plus proche, d = 1 − IoU(boîte, centroïde)
        assign = np.argmax(iou_wh(wh, centroids), axis=1)
        new = centroids.copy()
        for q in range(k):
            members = wh[assign == q]
            if len(members):
                new[q] = members.mean(axis=0)
        if np.array_equal(new, centroids):
            break
        centroids = new
    return centroids[np.argsort(centroids[:, 0] * centroids[:, 1], kind="stable")]


def _init_plusplus(wh, k, rng):
    """Tirage des k boîtes initiales (hors base : k-means++ avec d = 1 − IoU).

    Le §5.2 tire les k boîtes uniformément ; deux centroïdes initiaux dans le même amas
    piègent alors facilement l'algorithme dans un minimum local. k-means++ tire chaque
    nouvelle boîte avec une probabilité proportionnelle à d², d = distance au centroïde le
    plus proche.
    """
    centroids = [wh[rng.integers(len(wh))]]
    for _ in range(1, k):
        d = 1.0 - iou_wh(wh, np.array(centroids)).max(axis=1)
        p = d * d
        idx = rng.choice(len(wh), p=p / p.sum()) if p.sum() > 0 else rng.integers(len(wh))
        centroids.append(wh[idx])
    return np.array(centroids)


def mean_best_iou(wh, anchors):
    """IoU moyenne de chaque boîte avec son centroïde le plus proche (critère de §5.2)."""
    return float(iou_wh(wh, anchors).max(axis=1).mean())
