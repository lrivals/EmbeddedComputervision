"""Synthetic degradations of the ODD stress test (T27.3), NumPy, seeded per image.

    # contact sheet of one image: every degradation at its 3 severities
    python tools/degrade.py --show data/prewid/images/test/211000_CAM0.jpg \
        --out build/m27/degrade_sheet.jpg

Each degradation takes an RGB uint8 image (H, W, 3) and a severity 1-3, and returns an
image of the same shape and type. The random draws come from `rng_for(seed, image id)`, so a
run is the same at every call, whatever the order or the number of processes.

| Kind | Model | Severity 1 / 2 / 3 |
|---|---|---|
| `low_light` | exposure k·x^γ, then shot + read noise (σ² = a·y + b²) | k 0.5 / 0.3 / 0.15, γ 1.5 / 2 / 2.5 |
| `fog` | haze model x·t + A·(1 − t), veil A = 0.8 (contrast loss + veil) | t 0.7 / 0.5 / 0.3 |
| `motion_blur` | line kernel, random angle, length a share of the width | 1 / 2 / 4 % |
| `sensor_noise` | additive Gaussian noise on [0, 1] | σ 0.03 / 0.06 / 0.10 |
| `jpeg` | JPEG re-encoding (PIL) | quality 30 / 15 / 5 |
| `occlusion` | grey patch over each person box, from a random side | 25 / 50 / 75 % of the box |

`occlusion` needs the boxes (continuous pixel corners); the others ignore them. These are the
degradations proposed as the ODD axes of T31.0 (lighting, dust / fog, camera motion, sensor,
compression, partial occlusion).
"""

import argparse
import io
import zlib
from pathlib import Path

import numpy as np

SEVERITIES = (1, 2, 3)
LOW_LIGHT = {1: (0.5, 1.5), 2: (0.3, 2.0), 3: (0.15, 2.5)}  # (exposure k, gamma)
SHOT, READ = 0.01, 0.01  # noise of the dark image: variance SHOT·y + READ²
FOG = {1: 0.7, 2: 0.5, 3: 0.3}  # transmission t
VEIL = 0.8
BLUR = {1: 0.01, 2: 0.02, 3: 0.04}  # kernel length / image width
NOISE = {1: 0.03, 2: 0.06, 3: 0.10}
JPEG = {1: 30, 2: 15, 3: 5}
OCCLUSION = {1: 0.25, 2: 0.5, 3: 0.75}
GREY = 0.5


def rng_for(seed, image_id):
    """Generator of one image: same draws for the same (seed, id) in any process."""
    return np.random.default_rng([int(seed), zlib.crc32(str(image_id).encode())])


def _f(img):
    return np.asarray(img, np.float64) / 255.0


def _u8(x):
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def low_light(img, sev, rng, boxes=None):
    k, g = LOW_LIGHT[sev]
    y = k * _f(img) ** g
    return _u8(y + rng.normal(0.0, 1.0, y.shape) * np.sqrt(SHOT * y + READ ** 2))


def fog(img, sev, rng, boxes=None):
    t = FOG[sev]
    return _u8(_f(img) * t + VEIL * (1.0 - t))


def motion_blur(img, sev, rng, boxes=None):
    x = _f(img)
    h, w = x.shape[:2]
    n = max(3, int(round(BLUR[sev] * w)))  # at least 3 taps: a 2-tap line may round to one
    a = rng.uniform(0.0, np.pi)
    pad = n
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(x)
    for s in np.linspace(-(n - 1) / 2, (n - 1) / 2, n):
        dx, dy = int(np.floor(s * np.cos(a) + 0.5)), int(np.floor(s * np.sin(a) + 0.5))
        acc += xp[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
    return _u8(acc / n)


def sensor_noise(img, sev, rng, boxes=None):
    x = _f(img)
    return _u8(x + rng.normal(0.0, NOISE[sev], x.shape))


def jpeg(img, sev, rng, boxes=None):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, format="JPEG", quality=JPEG[sev])
    with Image.open(buf) as out:
        return np.asarray(out.convert("RGB"))


def occlusion(img, sev, rng, boxes=None):
    """A grey patch covering a share `OCCLUSION[sev]` of each box, attached to a random side
    (left, right, top or bottom), the way a machine part or a rack hides a worker."""
    out = np.array(img, np.uint8, copy=True)
    h, w = out.shape[:2]
    f = OCCLUSION[sev]
    for x0, y0, x1, y1 in np.asarray(boxes if boxes is not None else [], np.float64).reshape(-1, 4):
        side = int(rng.integers(4))
        if side == 0:
            x1 = x0 + f * (x1 - x0)
        elif side == 1:
            x0 = x1 - f * (x1 - x0)
        elif side == 2:
            y1 = y0 + f * (y1 - y0)
        else:
            y0 = y1 - f * (y1 - y0)
        a, b = int(np.clip(np.floor(x0), 0, w)), int(np.clip(np.ceil(x1), 0, w))
        c, d = int(np.clip(np.floor(y0), 0, h)), int(np.clip(np.ceil(y1), 0, h))
        out[c:d, a:b] = int(round(GREY * 255))
    return out


KINDS = {"low_light": low_light, "fog": fog, "motion_blur": motion_blur,
         "sensor_noise": sensor_noise, "jpeg": jpeg, "occlusion": occlusion}


def parse(spec):
    """`kind:sev` → (kind, sev); raises on an unknown kind or severity."""
    kind, _, sev = spec.partition(":")
    if kind not in KINDS or not sev.isdigit() or int(sev) not in SEVERITIES:
        raise ValueError(f"degradation {spec!r}: expected <kind>:<1-3>, kind in "
                         f"{', '.join(KINDS)}")
    return kind, int(sev)


def all_specs(kinds=None):
    return [f"{k}:{s}" for k in (kinds or KINDS) for s in SEVERITIES]


def apply(img, spec, rng, boxes=None):
    """Degradation `spec` (`kind:sev`) of `img` (H, W, 3) uint8."""
    kind, sev = parse(spec)
    return KINDS[kind](np.asarray(img, np.uint8), sev, rng, boxes)


def sample_boxes(sample):
    """Continuous pixel corners of the objects of a sample (VOC `xyxy` back to x0 = x1 − 1)."""
    b = np.asarray(sample["xyxy"], np.float64).reshape(-1, 4).copy()
    b[:, :2] -= 1.0
    return b


def contact_sheet(img, boxes=None, seed=0, image_id="sheet", width=320):
    """One row per degradation, one column per severity (plus the original first)."""
    from PIL import Image

    src = Image.fromarray(np.asarray(img, np.uint8))
    scale = width / src.width
    th = int(round(src.height * scale))
    rows = []
    for kind in KINDS:
        row = [np.asarray(src.resize((width, th)))]
        for s in SEVERITIES:
            out = apply(img, f"{kind}:{s}", rng_for(seed, image_id), boxes)
            row.append(np.asarray(Image.fromarray(out).resize((width, th))))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", type=Path, required=True, help="image to degrade")
    ap.add_argument("--out", type=Path, default=Path("build/m27/degrade_sheet.jpg"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--box", type=float, nargs=4, action="append", default=None,
                    help="person box x0 y0 x1 y1 for `occlusion` (repeatable)")
    args = ap.parse_args()
    from PIL import Image

    with Image.open(args.show) as im:
        img = np.asarray(im.convert("RGB"))
    sheet = contact_sheet(img, args.box, args.seed, args.show.stem)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(args.out, quality=90)
    print(f"→ {args.out} (rows: {', '.join(KINDS)}; columns: original, severity 1-3)")


if __name__ == "__main__":
    main()
