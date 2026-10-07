"""Chargement des lots d'entraînement, en parallèle (§7.1).

`VOCDataset` rend une image prétraitée (CHW float32 dans [0, 1]) et ses boîtes normalisées
dans le repère de l'entrée (S × S, ou H × W pour un réseau non carré). `DataLoader` assemble des lots, la taille d'entrée de
chaque lot étant donnée par `size_fn(itération)` (multi-échelle, §2.2), et charge les images
dans un `multiprocessing.Pool` avec un nombre borné d'images d'avance.

Tirages reproductibles : chaque image d'une époque reçoit le générateur
`default_rng([seed, epoch, index])`, indépendamment du nombre de processus.
"""

import multiprocessing as mp
from collections import deque

import numpy as np

from yolo.data.augment import IDENTITY, augment, random_crop, random_params


class VOCDataset:
    """`samples` : sortie de `yolo.data.voc.load_split`.

    Les objets `difficult` sont retirés à l'entraînement (convention Darknet, hors base).
    `resize` : `letterbox` ou `stretch` (celui de l'éval du jeu). `crop` > 0 : à
    l'entraînement, découpe aléatoire `crop` × `crop` px de l'image d'origine avant
    l'augmentation (tuiles, objets vus à une résolution plus proche de l'origine).
    """

    def __init__(self, samples, train=True, keep_difficult=False, jitter=0.2, hsv=1.5,
                 flip=True, channels=3, resize="letterbox", crop=0):
        self.samples = samples
        self.channels = channels
        self.resize = resize
        self.crop = crop
        self.train = train
        self.keep_difficult = keep_difficult
        self.aug = {"jitter": jitter, "hsv": hsv, "flip": flip}

    def __len__(self):
        return len(self.samples)

    def load(self, index, size, rng=None):
        from PIL import Image

        s = self.samples[index]
        keep = np.ones(len(s["labels"]), bool) if self.keep_difficult else ~s["difficult"]
        params = random_params(rng, **self.aug) if self.train else IDENTITY
        boxes, labels = s["boxes"][keep], s["labels"][keep]
        with Image.open(s["image"]) as img:
            if self.crop and self.train:
                img, boxes, labels = random_crop(img, boxes, labels, self.crop, rng)
            out, boxes, labels = augment(img, boxes, labels, size, params, self.channels,
                                         self.resize)
        return np.ascontiguousarray(out.transpose(2, 0, 1)), boxes, labels


_WORKER_DATASET = None


def _init_worker(dataset):
    global _WORKER_DATASET
    _WORKER_DATASET = dataset


def _load_task(task):
    index, size, seed = task
    return _WORKER_DATASET.load(index, size, np.random.default_rng(seed))


class DataLoader:
    """Lots `(images (N, C, H, W), [boîtes], [labels])` d'une époque."""

    def __init__(self, dataset, batch_size, shuffle=True, workers=0, seed=0, prefetch=2,
                 drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.seed = seed
        self.prefetch = prefetch
        self.drop_last = drop_last
        self._pool = None

    def __len__(self):
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            n += 1
        return n

    def _order(self, epoch):
        idx = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.default_rng([self.seed, epoch]).shuffle(idx)
        return idx

    def epoch(self, epoch, size_fn, first_iter=0, skip=0):
        """Lots de l'époque `epoch` ; le lot b a la taille `size_fn(first_iter + b)`.

        `skip` : nombre de lots à sauter en début d'époque (reprise d'un checkpoint).
        """
        order = self._order(epoch)
        tasks = []
        for b in range(skip, len(self)):
            size = size_fn(first_iter + b - skip)
            size = tuple(map(int, size)) if isinstance(size, (tuple, list)) else int(size)
            for k in order[b * self.batch_size:(b + 1) * self.batch_size]:
                tasks.append((int(k), size, [self.seed, epoch, int(k)]))
        results = self._run(tasks)
        batch = []
        for item in results:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield _collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield _collate(batch)

    def _run(self, tasks):
        if self.workers <= 0:
            for t in tasks:
                yield self.dataset.load(t[0], t[1], np.random.default_rng(t[2]))
            return
        if self._pool is None:
            self._pool = mp.Pool(self.workers, initializer=_init_worker,
                                 initargs=(self.dataset,))
        window = deque()
        ahead = self.prefetch * self.batch_size
        it = iter(tasks)
        for t in it:
            window.append(self._pool.apply_async(_load_task, (t,)))
            if len(window) >= ahead:
                break
        for t in it:
            yield window.popleft().get()
            window.append(self._pool.apply_async(_load_task, (t,)))
        while window:
            yield window.popleft().get()

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def _collate(items):
    images = np.stack([it[0] for it in items])
    return images, [it[1] for it in items], [it[2] for it in items]
