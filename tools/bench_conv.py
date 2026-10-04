"""T1.2 : temps de la convolution im2col sur la couche 13 de Tiny-YOLOv2 (13×13×1024→1024, 3×3).

Usage : python tools/bench_conv.py [--batch N] [--repeat R]
"""

import argparse
import time

import numpy as np

from yolo.layers.conv import conv_backward, conv_forward


def bench(fn, repeat):
    fn()  # échauffement
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    macs = args.batch * 13 * 13 * 1024 * 1024 * 9
    rng = np.random.default_rng(0)
    print(f"conv 3×3 13×13×1024→1024, lot {args.batch} : {macs / 1e9:.2f} GMAC")
    for dtype in (np.float32, np.float64):
        x = rng.standard_normal((args.batch, 1024, 13, 13)).astype(dtype)
        W = rng.standard_normal((1024, 1024, 3, 3)).astype(dtype)
        y, cache = conv_forward(x, W)
        dy = np.ones_like(y)
        tf = bench(lambda: conv_forward(x, W), args.repeat)
        tb = bench(lambda: conv_backward(dy, cache), args.repeat)
        print(f"  {np.dtype(dtype).name:8s} avant {tf * 1e3:7.1f} ms ({macs / tf / 1e9:5.1f} GMAC/s)"
              f"   arrière {tb * 1e3:7.1f} ms")


if __name__ == "__main__":
    main()
