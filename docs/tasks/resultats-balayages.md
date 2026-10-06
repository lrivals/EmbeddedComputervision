# Résultats des balayages et inférences (VOC, VisDrone)

Premiers passages des notebooks `_sweep` (T14.10) et `_infer` (T14.3, T14.11) de
`notebooks/voc/` et `notebooks/visdrone/`, sur un runtime Colab à GPU
([colab-vscode.md](colab-vscode.md)), révision `1a967f4`. Les runs sont dans
`build/notebooks/<jeu>/<modèle>/runs/` et sur Drive (`<DRIVE_DIR>/runs/`,
[donnees-drive.md](donnees-drive.md)).

Les notebooks exécutés sont versionnés avec leurs sorties (règle de
[M14](M14-notebooks.md#règles)) ; toutes les tables et courbes ci-dessous s'y relisent :

| jeu | balayage | inférence |
|---|---|---|
| VOC | [tiny-yolov3-voc_sweep](../../notebooks/voc/tiny-yolov3-voc_sweep.ipynb) | [tiny-yolov3-voc](../../notebooks/voc/tiny-yolov3-voc_infer.ipynb), [tiny-yolov3-coco](../../notebooks/voc/tiny-yolov3-coco_infer.ipynb), [tiny-yolov2-voc](../../notebooks/voc/tiny-yolov2-voc_infer.ipynb) |
| VisDrone | [tiny-yolov3-visdrone_sweep](../../notebooks/visdrone/tiny-yolov3-visdrone_sweep.ipynb) | [tiny-yolov3-visdrone](../../notebooks/visdrone/tiny-yolov3-visdrone_infer.ipynb), [tiny-yolov3-coco](../../notebooks/visdrone/tiny-yolov3-coco_infer.ipynb), [tiny-yolov2-voc](../../notebooks/visdrone/tiny-yolov2-voc_infer.ipynb) |

Figures : `python -m tools.figures balayage` (T13.53) relit les tableaux de ce fichier et
réécrit les PNG de [figures/resultats/](figures/resultats/). Pour les voir se mettre à jour
pendant qu'on travaille : [notebooks/figures_live.ipynb](../../notebooks/figures_live.ipynb).

## Conditions

| | |
|---|---|
| palier | **R** : mAP sur les 50 premières images (`SUBSET = 50`) |
| évaluation | VOC2007 test ou visdrone val, 416×416 `stretch` (Pillow), conf 0,005, NMS 0,45, AP 11 points |
| entraînement | 600 itérations par run, LR 0,001 (non ajusté au lot), burn-in 500, multi-échelle 320-608, init COCO (`weights/yolov3-tiny.weights`), `--device gpu` |
| grille | `BATCHES = [8, 16, 32]` × `TRAIN_SUBSETS = [500, 0]` (500 premières images, ou tout le split) |

Ces mAP **classent** les runs ; elles ne sont **pas publiables** (règles de
[M12](M12-profils-pc.md#règles)) et ne vont pas dans `results/`. Avec 50 images, une
classe peut n'avoir qu'une ou deux instances : son AP saute entre 0 et 100. Ordre de
grandeur du biais : Tiny-YOLOv2 VOC fait 63,51 sur ces 50 images contre **56,30** sur le
split complet ([results/map_float.md](../../results/map_float.md)).

## Vue d'ensemble

![mAP par lot et sous-ensemble](figures/resultats/balayage_map.png)

Chaque panneau a sa propre échelle : VisDrone plafonne à 2,24, VOC à 36,27.

## VOC

### Balayage `tiny-yolov3-voc` (16 551 images d'entraînement, VOC07+12)

| run | lot | images | époques | perte finale | mAP (50 images) |
|---|---|---|---|---|---|
| **b32-sall** | 32 | tout | 1,16 | 14,66 | **36,27** |
| b16-sall | 16 | tout | 0,58 | 17,45 | 34,70 |
| b8-s500 | 8 | 500 | 9,60 | 18,00 | 30,14 |
| b16-s500 | 16 | 500 | 19,20 | 13,78 | 25,58 |
| b32-s500 | 32 | 500 | 38,40 | 11,95 | 24,01 |
| b8-sall | 8 | tout | 0,29 | 18,29 | 12,41 |

Durée : environ 10 min par run de lot 32 sur le GPU Colab (624 s pour b32-sall).

### Inférence sur les mêmes 50 images

| modèle | poids | classes évaluées | mAP |
|---|---|---|---|
| `tiny-yolov3-coco` | Darknet COCO (`yolov3-tiny.weights`) | 20 (via `MAPPINGS`) | **68,45** |
| `tiny-yolov2-voc` | Darknet VOC (`yolov2-tiny-voc.weights`) | 20 | 63,51 |
| `tiny-yolov3-voc` | run `b32-sall` | 20 | 36,27 |

AP du run `b32-sall` : aeroplane et bicycle 100, cat 75,6, bus 59,1, tvmonitor 54,5,
bird 50,6, person 47,2 ; bottle et sofa 0, boat 1,8, cow 2,1, pottedplant 1,3.

## VisDrone

### Balayage `tiny-yolov3-visdrone` (cfg `build/m11/cfg/tiny-yolov3-visdrone.cfg`, 10 classes, 6 471 images)

| run | lot | images | époques | perte finale | mAP (50 images) |
|---|---|---|---|---|---|
| **b32-sall** | 32 | tout | 2,97 | 185,96 | **2,24** |
| b16-s500 | 16 | 500 | 19,20 | 220,65 | 1,35 |
| b8-sall | 8 | tout | 0,74 | 208,68 | 1,24 |
| b16-sall | 16 | tout | 1,48 | 203,54 | 1,15 |
| b32-s500 | 32 | 500 | 38,40 | 200,33 | 0,78 |
| b8-s500 | 8 | 500 | 9,60 | 252,11 | 0,56 |

Seule la classe `car` décolle (AP 17,4 pour b32-sall) ; van 2,0, truck 1,7, les autres
sous 0,5. Pertes de 186 à 252 : aucun run n'a convergé.

### Inférence hors domaine (T11.2) sur les mêmes 50 images

| modèle | classes évaluées | mAP |
|---|---|---|
| `tiny-yolov3-coco` | 6 (person, bicycle, car, motorbike, bus, truck) | 6,95 |
| `tiny-yolov2-voc` | 5 (person, bicycle, car, motorbike, bus) | 4,85 |
| `tiny-yolov3-visdrone` `b32-sall` | 10 | 2,24 |

Les mAP hors domaine portent sur les classes ayant un équivalent (`MAPPINGS` :
pedestrian et people → person, van → car, motor → motorbike ; tricycle et
awning-tricycle ignorés) : elles ne se comparent pas directement à la mAP à 10 classes.
Par classe, `car` est à 21,5 pour COCO contre 17,4 pour le run affiné.

![Run affiné face aux poids publiés](figures/resultats/balayage_modeles.png)

## Meilleurs paramètres

**Lot 32 sur tout le split d'entraînement (`b32-sall`)**, sur les deux jeux. C'est le
run par défaut des notebooks d'inférence (`RUN = None` prend le plus récent, ici
`b32-sall`).

```
python tools/train.py --net tiny-yolov3-voc --init coco --iters 600 --batch 32 \
    --lr 0.001 --burn-in 500 --multiscale --workers 4 --device gpu \
    --out build/notebooks/voc/tiny-yolov3-voc/runs/b32-sall
python tools/train.py --net build/m11/cfg/tiny-yolov3-visdrone.cfg --dataset visdrone \
    --init coco --iters 600 --batch 32 --lr 0.001 --burn-in 500 --multiscale \
    --workers 4 --device gpu --out build/notebooks/visdrone/tiny-yolov3-visdrone/runs/b32-sall
```

Ce que montre la grille :

- **Tout le split plutôt que 500 images.** Sur VOC, les runs `s500` font 10 à 38 époques
  sur les mêmes 500 images et surapprennent : `b32-s500` a la perte finale la plus basse
  (11,95) mais une mAP de 24,01, contre 36,27 pour `b32-sall` (écart de 12 points à lot
  32, de 9 points à lot 16). Seul le lot 8 fait exception, voir le point suivant.
- **Un grand lot.** À 600 itérations, le lot fixe le nombre d'images vues. `b8-sall`
  (0,29 époque) est le pire run VOC : le burn-in occupe 500 des 600 itérations, le LR
  n'atteint sa valeur qu'à la fin et le gradient d'un lot de 8 est bruité.
- **Écarts faibles entre les deux meilleurs.** `b32-sall` et `b16-sall` (1,6 point sur
  VOC) ne se départagent pas sur 50 images ; sur VisDrone, tous les runs sauf `b32-sall`
  restent dans le bruit (0,6 à 1,4).

![Perte finale face à la mAP](figures/resultats/balayage_perte.png)

Sur VOC, les runs à 500 images (marques creuses) ont les pertes les plus basses et les mAP
les plus faibles : ils surapprennent.

## Limites et suite

Exécutions et améliorations planifiées : [M15](M15-campagne-entrainement.md).

- **Entraînement trop court.** L'affinage de 600 itérations part des poids COCO, qui
  font 68,45 sur VOC avec la table de correspondance, et tombe à 36,27 : les têtes à
  20 classes repartent de zéro et n'ont vu qu'une époque. Sur VisDrone (petits objets
  nombreux), l'écart est plus fort encore.
- **Pistes**, dans l'ordre :
  1. allonger `ITERS` sur la configuration retenue (lot 32, tout le split ; palier N) ;
     l'exemple de `tools/train.py` pour VOC est à 20 000 itérations ;
  2. raccourcir `BURN_IN` (100 à 200) ou ajuster le LR au lot pour les runs courts ;
  3. VisDrone : entrée plus grande (`SIZE`, ex. `608` ou non carrée) pour les petits
     objets, avec les ancres k-means recalculées à cette taille.
- **Confirmer le classement** sur le split complet : notebook `_infer` avec
  `COMPARE = True` et `SUBSET = 0` (palier N), puis volet entier (`INT8 = True`, T14.4)
  sur le run retenu.
- FLIR et KITTI : pas de résultat. FLIR s'arrête à la cellule d'environnement (jeu absent
  du Drive : `tools/get_datasets.sh push flir`), KITTI a été interrompu avant les runs.
  Exécutions partielles, donc non versionnées : les deux notebooks sont remis à vide.
