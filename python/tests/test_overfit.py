"""T2.8 : surapprentissage d'une image (§11) — la perte tend vers 0 et les boîtes coïncident.

Tiny-YOLOv3-VOC complet, initialisation He, entrée 256×256 (grilles 8×8 et 16×16) pour
tenir en une minute environ. Lancer avec `pytest -m slow` (ou `make test-slow`).
"""

import numpy as np
import pytest

from yolo.infer.boxes import iou
from yolo.infer.decode import decode
from yolo.models.tiny_yolo import tiny_yolov3_voc
from yolo.train.optim import SGD
from yolo.train.schedule import StepSchedule
from yolo.train.trainer import Trainer

SIZE = 256
ITERS = 200
BOXES = np.array([[0.30, 0.35, 0.30, 0.40],    # ancres 13×13 ou 26×26 selon la forme
                  [0.70, 0.65, 0.40, 0.30],
                  [0.75, 0.20, 0.12, 0.15]])
LABELS = np.array([7, 14, 2])
COLORS = [(0.9, 0.2, 0.1), (0.1, 0.3, 0.9), (0.1, 0.8, 0.2)]


def _image():
    img = np.full((3, SIZE, SIZE), 0.5, np.float32)
    for (cx, cy, w, h), c in zip(BOXES, COLORS):
        x1, x2 = round((cx - w / 2) * SIZE), round((cx + w / 2) * SIZE)
        y1, y2 = round((cy - h / 2) * SIZE), round((cy + h / 2) * SIZE)
        img[:, y1:y2, x1:x2] = np.array(c, np.float32)[:, None, None]
    return img[None]


@pytest.mark.slow
def test_overfit_single_image():
    x = _image()
    net = tiny_yolov3_voc(rng=0)
    schedule = StepSchedule(1e-3, burn_in=50, steps=(160,), scales=(0.1,))
    trainer = Trainer(net, SGD(net.params), schedule, size=SIZE)
    losses = [trainer.step(x, [BOXES], [LABELS])[0].total for _ in range(ITERS)]
    assert losses[-1] < 1e-3 * losses[0]
    assert np.mean(losses[-10:]) < 0.2

    # Après décodage (T3.1), la meilleure boîte de chaque classe vérité coïncide avec elle.
    boxes, _, scores = decode(net.forward(x, train=False), net.net)
    for gt, c in zip(BOXES, LABELS):
        k = scores[0, :, c].argmax()
        assert scores[0, k, c] > 0.9
        assert iou(boxes[0, k], gt)[0, 0] > 0.9
