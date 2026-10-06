"""Notebooks d'entraînement et d'inférence (M14, docs/tasks/M14-notebooks.md).

    python -m tools.notebooks all          # tous les notebooks + notebooks/README.md (make notebooks)
    python -m tools.notebooks kitti        # un jeu ; aussi un rôle (infer, train) ou un nom
    python -m tools.notebooks --list
    python -m tools.notebooks --check      # échoue si un notebook versionné a changé ou porte
                                           # une exécution partielle ou en erreur (make ci)

Un notebook n'apporte aucun calcul : il lance les outils de `tools/` (commandes dans
`commandes.py`) et affiche leurs sorties. Les notebooks sont écrits sans sorties dans
`notebooks/<jeu>/<modèle>_<rôle>.ipynb` (une exécution complète sans erreur est gardée,
règle M14) ; leurs résultats vont dans
`build/notebooks/<jeu>/<modèle>/`, jamais dans `results/`. Ni Jupyter ni matplotlib ne
sont importés ici ni depuis `python/yolo/` (ADR 0001).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NB_DIR = ROOT / "notebooks"
OUT = "build/notebooks"  # relatif à la racine : les notebooks font os.chdir(ROOT)
ROLES = ("infer", "train", "sweep")  # sweep : balayage lot × sous-ensemble (T14.10)


@dataclass(frozen=True)
class Notebook:
    dataset: str
    model: str            # nom affiché : tiny-yolov2-voc, tiny-yolov3-kitti…
    role: str             # infer | train | sweep
    net: str              # nom de CFG_FILES ou chemin de cfg (relatif à la racine)
    weights: str          # poids évalués (infer) ou de départ (train), relatifs à la racine
    family: str           # classes de sortie : voc, coco ou le jeu (poids affinés)
    tasks: tuple          # tâches sources (T11.4…)
    channels: int = None  # canaux d'entrée de la cfg d'affinage (FLIR : 1, T11.7)
    size: str = None      # entrée LxH de la cfg d'affinage (défaut : 416 × 416)

    @property
    def name(self):
        return f"{self.model}_{self.role}"

    @property
    def path(self):
        return Path(self.dataset) / f"{self.name}.ipynb"

    @property
    def train_dir(self):
        """Runs du modèle : `train` à la racine, balayage dans `runs/` (tools/notebooks/runs.py)."""
        return f"{OUT}/{self.dataset}/{self.model}"

    @property
    def out_dir(self):
        return f"{self.train_dir}/eval" if self.role == "infer" and self.finetuned else self.train_dir

    @property
    def trains(self):
        return self.role in ("train", "sweep")

    @property
    def out_of_domain(self):
        return self.family != self.dataset

    @property
    def finetuned(self):
        return self.weights.endswith("final.weights")
