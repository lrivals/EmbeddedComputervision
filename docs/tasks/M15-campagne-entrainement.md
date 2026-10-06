# M15 — Campagne d'entraînement (optionnel)

Objectif : passer des premiers balayages de M14 à des modèles affinés dont la mAP tient
sur le split complet, jeu par jeu. Le jalon liste les exécutions à lancer et les
améliorations d'outillage qu'elles demandent. VOC et VisDrone ont des résultats ; les
autres jeux ont une tâche prête, à remplir quand leurs runs auront tourné.

Constat de départ ([resultats-balayages.md](resultats-balayages.md), palier R,
50 images, 600 itérations par run, rév. `1a967f4`) :
- **VOC** : meilleur run `b32-sall` (lot 32, tout le split), mAP 36,27. Les poids COCO de
  départ font 68,45 sur les mêmes images : 600 itérations (1,16 époque) ne suffisent pas à
  réapprendre les têtes réinitialisées.
- **VisDrone** : meilleur run `b32-sall`, mAP 2,24 ; pertes de 186 à 252, aucun run n'a
  convergé ; seule `car` décolle (AP 17,4).
- **KITTI** : meilleur run `b32-sall`, mAP 2,38, en 416×416 `stretch` ; seule `Car`
  décolle (AP 16,8), loin des poids COCO hors domaine (20,55 sur 4 classes, car 49,7).
- Le classement repose sur 50 images : l'écart entre les deux meilleurs runs VOC
  (1,6 point) est dans le bruit.
- Seuls le lot et le sous-ensemble sont balayés ; LR (0,001), burn-in (500 sur 600
  itérations) et taille d'entrée sont fixes.
- FLIR (jeu absent du Drive) n'a pas de résultat ; ExDark et CrowdHuman n'ont pas été
  lancés.

Coûts mesurés sur Colab : entraînement GPU ≈ 0,033 s/image (624 s pour 600 itérations
au lot 32, soit ≈ 1 s/itération) ; évaluation sur CPU ≈ 0,27 s/image, soit ≈ 22 min pour
VOC2007 test (4 952 images), ≈ 2,5 min pour VisDrone val (548 images) et ≈ 7 min pour
KITTI val (1 496 images).

**État** : 0 tâche faite sur 14.

## Conventions communes

### Protocole par jeu

1. **Balayage** palier R (`_sweep`, lot × sous-ensemble, T14.10).
2. **Confirmation** du classement sur le split complet (`_infer`, `COMPARE = True`,
   `SUBSET = 0`).
3. **Run long** sur la meilleure configuration (`_train` ou `_sweep` à une case).
4. **Axe propre au jeu** : taille d'entrée (VisDrone, KITTI), canaux (FLIR), etc.
5. **Volet entier** (`INT8 = True`, T14.4) sur le run retenu.

### Règles

- Les résultats de chaque étape vont dans [resultats-balayages.md](resultats-balayages.md),
  une section par jeu, avec la révision et les paramètres.
- Une mAP sur `SUBSET` images classe les runs, elle ne se publie pas. Seule une mAP sur le
  split complet (palier N) peut aller dans `results/` (règles de
  [M12](M12-profils-pc.md#règles)).
- Un écart entre deux runs ne compte que s'il dépasse le bruit mesuré en T15.2.
- Les notebooks restent versionnés sans sorties ([M14](M14-notebooks.md#règles)) ; un
  changement de paramètre par défaut passe par `tools/notebooks/gabarits.py` et
  `make notebooks`.

---

## A. Mesures fiables (VOC, VisDrone, KITTI)

### [ ] T15.1 — Classement sur le split complet
- **Spec** : §8 · **Dépend de** : T14.11 · **Taille** : S
- **Livrables** : table des 18 mAP (6 runs × 3 jeux) dans `resultats-balayages.md`
- **Acceptation** : chaque run de VOC, VisDrone et KITTI évalué sur tout le split ;
  meilleur run confirmé ou remplacé
- **Notes** : notebooks `_infer` de `tiny-yolov3-voc`, `tiny-yolov3-visdrone` et
  `tiny-yolov3-kitti` avec `COMPARE = True`, `SUBSET = 0`. Compter ≈ 2 h 15 pour VOC
  (6 × 22 min), ≈ 15 min pour VisDrone, ≈ 40 min pour KITTI.

### [ ] T15.2 — Bruit d'un run
- **Spec** : §7 · **Dépend de** : T15.1 · **Taille** : S
- **Livrables** : écart-type de la mAP sur 3 graines
- **Acceptation** : `b32-sall` et `b16-sall` de VOC relancés avec `--seed 1` et
  `--seed 2`, évalués sur le split complet ; écart-type rapporté
- **Notes** : ≈ 40 min de GPU pour les 4 runs. Le résultat sert de seuil de
  significativité pour toutes les comparaisons de M15.

### [ ] T15.3 — Durée GPU dans le plan du balayage
- **Spec** : — · **Dépend de** : — · **Taille** : S
- **Livrables** : `S_PER_IMAGE["gpu"]` dans `tools/notebooks/matrice.py` (≈ 0,033 s)
- **Acceptation** : la table du plan de `_sweep` affiche une durée en GPU au lieu de
  « non mesurée » ; `python -m tools.notebooks --check` passe
- **Notes** : `estimate` lit déjà `loss.csv` quand un run existe ; la constante ne sert
  qu'avant le premier run.

## B. VOC

### [ ] T15.4 — Run long `tiny-yolov3-voc`
- **Spec** : §7.1 · **Dépend de** : T15.1 · **Taille** : M
- **Livrables** : run `long` dans `build/notebooks/voc/tiny-yolov3-voc/` ; courbe mAP /
  itérations
- **Acceptation** : mAP sur VOC2007 test complet comparée à la référence Tiny-YOLOv2
  (56,30, [map_float.md](../../results/map_float.md)) et aux poids COCO (T2.9)
- **Notes** :
  - lot 32, tout le split, 6 000 itérations (≈ 11,6 époques, ≈ 1 h 45 de GPU),
    `--burn-in 200`, `--steps 4800,5400 --scales 0.1,0.1`, `SAVE_EVERY = 500` ;
  - mAP palier R de chaque checkpoint pour la courbe, split complet sur le dernier ;
  - si la courbe monte encore à 6 000, prolonger par `--resume` vers 20 000 (exemple de
    `tools/train.py`).

### [ ] T15.5 — Axes LR et burn-in dans `_sweep`
- **Spec** : §7 · **Dépend de** : T15.2 · **Taille** : M
- **Livrables** : paramètres `LRS` et `BURN_INS` du gabarit `_sweep`
  (`tools/notebooks/gabarits.py`), nom de run étendu dans
  `tools/notebooks/runs.py:run_name` (ex. `b32-sall-lr2e-3-bi100`)
- **Acceptation** :
  - par défaut, `LRS = [LR]` et `BURN_INS = [BURN_IN]` : grille et noms de run actuels
    inchangés (test) ;
  - `--check` passe ;
  - balayage VOC LR {5e-4, 1e-3, 2e-3} × burn-in {100, 500} au lot 32, tout le split,
    600 itérations (6 runs, ≈ 1 h de GPU), comparé dans la table.
- **Notes** : à 600 itérations, le burn-in de 500 occupe presque tout l'entraînement ;
  c'est l'hypothèse principale derrière le mauvais score de `b8-sall`.

### [ ] T15.6 — Lot 64
- **Spec** : §7 · **Dépend de** : T15.5 · **Taille** : S
- **Livrables** : runs lot 64 (LR ×2) et lot 32 à nombre d'images vues égal
- **Acceptation** : mAP des deux runs sur le split complet ; mémoire GPU de pointe notée
- **Notes** : vérifier la mémoire du GPU Colab (T4 : 16 Go) en multi-échelle à 608.

### [ ] T15.7 — Volet entier du run retenu
- **Spec** : §9 · **Dépend de** : T15.4 · **Taille** : S
- **Livrables** : mAP flottante et INT8 du run retenu
- **Acceptation** : écart flottant → INT8 sur le split complet, comparé à celui de
  Tiny-YOLOv2 VOC ([map_int8.md](../../results/map_int8.md))
- **Notes** : `_infer` avec `INT8 = True` ; calibration sur 500 images du split `calib`.

## C. VisDrone

### [ ] T15.8 — Run long `tiny-yolov3-visdrone` à 416
- **Spec** : §7.1 · **Dépend de** : T15.1 · **Taille** : M
- **Livrables** : run `long` dans `build/notebooks/visdrone/tiny-yolov3-visdrone/`
- **Acceptation** : mAP sur visdrone val complet ; perte finale nettement sous 186
- **Notes** : même schéma que T15.4 (6 000 itérations ≈ 30 époques, ≈ 1 h 45 de GPU).
  Comparer l'AP de `car` à celle des poids COCO hors domaine (21,5 sur 50 images).

### [ ] T15.9 — Entrée 608 et 832
- **Spec** : §3, §10.2 · **Dépend de** : T15.8 · **Taille** : L
- **Livrables** : cfg et ancres à 608 et 832 ; runs longs ; AP par taille d'objet
- **Acceptation** : mAP et AP petits / moyens / grands objets (`--metric coco`) à 416, 608
  et 832 ; relie l'acceptation de [T11.5](M11-jeux-de-donnees.md)
- **Notes** :
  - `SIZE = '608'` ou `'832'` dans `_sweep` ou `_train` : la cellule de préparation refait
    ancres et cfg à cette taille ;
  - premier essai sans réentraînement : profil `tools/m11.sh visdrone-size` (mêmes
    poids, entrée plus grande) ;
  - coût par itération ×2,1 à 608, ×4 à 832.

### [ ] T15.10 — Régions ignorées de VisDrone
- **Spec** : §8.3 · **Dépend de** : — · **Taille** : M
- **Livrables** : régions ignorées (catégorie 0 des annotations) lues par le chargeur
  VisDrone et exclues de l'évaluation
- **Acceptation** : une détection dont le centre tombe dans une région ignorée n'est
  comptée ni vraie ni fausse (test) ; mAP avant / après sur le run retenu
- **Notes** : « Reste » de T11.5. Aujourd'hui, ces détections comptent comme fausses
  alarmes et sous-estiment la mAP.

### [ ] T15.11 — Une tête ou deux
- **Spec** : §3 · **Dépend de** : T15.8 · **Taille** : M
- **Livrables** : run long Tiny-YOLOv2 à 5 ancres (`NET=tiny-yolov2 tools/m11.sh
  visdrone-prep | visdrone-train`, init VOC hors tête)
- **Acceptation** : AP par taille d'objet de Tiny-YOLOv2 (13×13) face à Tiny-YOLOv3
  (13×13 et 26×26), même nombre d'itérations

## D. Autres jeux (en attente de résultats)

Chaque tâche suit le protocole commun. Sa table se remplit au fil des runs (mAP 50 images
au balayage, split complet ensuite).

### [ ] T15.12 — KITTI
- **Spec** : §5.2, §10.2 · **Dépend de** : T11.4 · **Taille** : L
- **Livrables** : runs longs en 416×416 et à 640×192 ; balayage à 640×192
  (`SIZE = '640x192'` dans `_sweep`)
- **Acceptation** : mAP sur KITTI val (1 496 images) aux deux entrées ; relie
  [T11.4](M11-jeux-de-donnees.md)
- **Notes** :
  - balayage 416×416 `stretch` fait (rév. `1a967f4`, détails dans
    [resultats-balayages.md](resultats-balayages.md#kitti)) : même meilleur run que VOC et
    VisDrone. Le notebook `_sweep` est à relancer en entier (« Run All », runs sautés par
    `SKIP_DONE`) pour être versionné avec ses sorties ;
  - en `stretch` 416×416, une image 1242×375 perd les deux tiers de sa hauteur relative ;
    en letterbox elle n'occupe que ≈ 30 % de l'entrée. À 640×192, elle occupe toute
    l'entrée.

  | run | lot | images | entrée | mAP (50 images) | mAP (complet) |
  |---|---|---|---|---|---|
  | b32-sall | 32 | tout | 416×416 | 2,38 | |
  | b16-sall | 16 | tout | 416×416 | 2,23 | |
  | b8-sall | 8 | tout | 416×416 | 2,18 | |
  | b32-s500 | 32 | 500 | 416×416 | 2,02 | |
  | b16-s500 | 16 | 500 | 416×416 | 0,50 | |
  | b8-s500 | 8 | 500 | 416×416 | 0,49 | |
  | à remplir (640×192) | | | 640×192 | | |

### [ ] T15.13 — FLIR
- **Spec** : §10.2 · **Dépend de** : T11.7 · **Taille** : L
- **Livrables** : FLIR sur le Drive (`tools/get_datasets.sh push flir`) ; balayage ; runs
  longs à 1 et 3 canaux
- **Acceptation** : mAP (`--metric coco`, AP@[.5:.95] et AP50) à 1 canal face à
  3 canaux ; relie [T11.7](M11-jeux-de-donnees.md)
- **Notes** : `CHANNELS = 1` dans `_sweep` (cfg `tiny-yolov3-flir-c1.cfg`, L00 sommée sur
  les canaux RGB).

  | run | lot | images | canaux | mAP (50 images) | mAP (complet) |
  |---|---|---|---|---|---|
  | à remplir | | | | | |

### [ ] T15.14 — ExDark et CrowdHuman
- **Spec** : §8 · **Dépend de** : T11.3, T11.6 · **Taille** : M
- **Livrables** : mAP hors domaine sur le split complet (poids Tiny-YOLOv2 VOC et
  Tiny-YOLOv3 COCO)
- **Acceptation** :
  - ExDark : écart à la mAP VOC par classe commune ; décision d'affinage
    ([T11.3](M11-jeux-de-donnees.md)) ;
  - CrowdHuman : AP et rappel de `person`, débordements de la NMS
    ([T11.6](M11-jeux-de-donnees.md), profil `tools/m11.sh crowdhuman`).
- **Notes** : pas de notebook d'entraînement pour ces jeux ; un affinage ExDark
  demanderait de l'ajouter au registre (`tools/notebooks/matrice.py`).

  | jeu | modèle | classes évaluées | mAP (50 images) | mAP (complet) |
  |---|---|---|---|---|
  | à remplir | | | | |

## Hors périmètre

- Entraînement sur COCO train2017 (trop grand, voir [M14](M14-notebooks.md)).
- Nouvelles architectures (troisième tête, autre backbone) : M9.
- QAT et ADMM : T14.7 et [M9.2](M9.2-req-yolo.md), [M9.3](M9.3-quant-4bits.md).

## Ordre conseillé

T15.3 et T15.10 sont indépendants et courts. Ensuite, confirmer le classement et le bruit
avant de lancer des heures de GPU.

```mermaid
graph LR
  T151[T15.1] --> T152[T15.2] --> T155[T15.5] --> T156[T15.6]
  T151 --> T154[T15.4] --> T157[T15.7]
  T151 --> T158[T15.8] --> T159[T15.9]
  T158 --> T1511[T15.11]
  T1510[T15.10] -.-> T158
  T153[T15.3]
  T1512[T15.12]
  T1513[T15.13]
  T1514[T15.14]
```
