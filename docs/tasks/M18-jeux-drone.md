# M18 — Jeux drone et thermiques : AU-AIR, DroneVehicle, HIT-UAV, UAVDT (optionnel)

Objectif : brancher quatre jeux vus de drone dans la chaîne de [M11](M11-jeux-de-donnees.md),
comme VisDrone, FLIR, ExDark et CrowdHuman. Pour chacun : un chargeur dans `DATASETS`, une
fiche, les notebooks de [M14](M14-notebooks.md) et de [M16](M16-presentation-jeux.md)
(statistiques, inférence hors domaine, affinage), et l'archive sur le Drive. Ce document dit
surtout **quels prétraitements chaque jeu demande** avant d'entrer dans la chaîne, et
pourquoi.

Chaque jeu éprouve une partie de la chaîne que VisDrone ne couvre qu'en partie :

| Jeu | Ce qu'il éprouve |
|---|---|
| auair | trames 1920×1080 réduites à 416 : objets minuscules, déséquilibre des classes (Car ≈ 78 %) |
| dronevehicle | boîtes orientées ramenées à des boîtes droites, scènes denses de véhicules, nuit en RGB |
| hituav | thermique à 1 canal vu de 60 à 130 m (L00 à cin = 1, comme FLIR), objets très petits |
| uavdt | 1024×540 non carré, trames vidéo très redondantes, objets parmi les plus petits des jeux |

**État** : 8 tâches faites sur 11. Données prêtes sur le PC (`tools/get_datasets.sh check`),
chargeurs, prétraitements, fiches et 24 notebooks générés. Restent l'exécution des notebooks
`_stats` (T18.8) et l'envoi sur le Drive (T18.9). Les affinages se suivent en M15 (T18.10).

## Synthèse

Les comptes viennent des chargeurs (`tools/data_stats.py`, révision `16c9b6f`). Les valeurs
« vues par le réseau » sont prises à 416×416 en letterbox, avec les ancres COCO, avant les
ancres propres à chaque jeu (`tools/m11.sh <jeu>-prep`).

| Jeu | Images (train / test) | Objets | Classes | Résolution | Objets/image (moy.) | Petits < 32² à 416 | Licence |
|---|---|---|---|---|---|---|---|
| auair | 27 880 / 4 943 (val) | 131 977 | 8 | 1920×1080 | 4,0 | 79,8 % | CC BY-NC-SA 2.0 |
| dronevehicle | 12 118 / 2 608 (+ 2 599 val) | 278 078 | 2 | 640×512 (après recadrage) | 16,0 | 65,4 % | CC BY-NC-SA 4.0 |
| hituav | 2 008 / 571 (+ 287 val) | 24 751 | 4 | 640×512, 1 canal | 8,6 | 84,2 % | à vérifier |
| uavdt | 24 143 / 16 592 | 798 795 | 3 | 1024×540 | 19,6 | 96,9 % | recherche seulement |

Correspondances vers les poids pré-entraînés (`MAPPINGS`, hors domaine) :

| Jeu | → VOC | → COCO, en plus | Sans équivalent |
|---|---|---|---|
| auair | Human→person, Car et Van→car, Motorbike, Bicycle, Bus | Truck→truck | Trailer |
| dronevehicle | small-vehicle→car | large-vehicle→truck (bus et camions confondus) | — |
| hituav | Person, Car, Bicycle | — | OtherVehicle |
| uavdt | car, bus | truck | — |

## Prétraitements

Règle : le chargeur lit le format brut quand c'est rapide. On ne prétraite sur disque que
si l'image elle-même doit changer (DroneVehicle) ou si la lecture du format brut est trop
lente (UAVDT). Les dossiers bruts gardent leur nom d'origine dans `data/`.
`tools/get_datasets.sh kaggle` les relie (ou les prétraite) vers `data/<jeu>`, le seul
chemin que lit le code.

```bash
tools/get_datasets.sh kaggle    # liens auair et hituav ; prep_datasets.py dronevehicle uavdt
tools/get_datasets.sh check     # 4 jeux « prêt »
```

### AU-AIR — aucun prétraitement sur disque

- **Brut** : `data/AU-AIR dataset/AU-AIR dataset/{annotations.json,images/}`, relié en
  `data/auair`.
- **Format** : un seul JSON, une entrée par trame (`image_name`, `bbox` en
  `top/left/width/height` pixels, `class` indice de `categories`), avec l'altitude,
  l'attitude et la date de chaque trame.
- **Pièges du JSON**, tolérés par `load_auair` :
  - la clé de largeur s'écrit `"image_width:"` (deux-points compris) ;
  - 5 972 trames sont nommées `…_xx_…` au lieu de `…_x_…` (les fichiers existent sous ce
    nom).
- **Découpage** : le jeu n'en a pas d'officiel, et ses 32 823 trames viennent de 8 vidéos.
  Un tirage au hasard placerait des trames voisines dans train et dans val. Le découpage
  se fait donc **par vidéo** : `AUAIR_VAL` réunit 3 vidéos entières (4 943 trames, 15 %),
  le reste va en train. `val` sert de split d'évaluation.
- **Retraits** : 54 boîtes vides après rognage (`make_sample`).
- Val n'a que 5 Motorbike et 66 Bus : l'AP de ces classes y est très bruitée.

### DroneVehicle — recadrage et boîtes englobantes (`tools/prep_datasets.py dronevehicle`)

- **Brut** : `data/DroneVehicle Dataset/DroneVehiclesDatasetYOLO/{train,val,test}/{images,labels}`.
  C'est la version Roboflow « YOLOv11-OBB » (M. Mandal). Elle ne garde que la moitié RGB
  du jeu d'origine (pas l'infrarouge) et re-découpe les images du train d'origine (n°
  00001 à 17990) en 12 118 / 2 599 / 2 608. Les classes sont fusionnées en small-vehicle
  (car, van) et large-vehicle (bus, truck, freight car).
- **Cadre blanc** : chaque image fait 840×712, soit l'image 640×512 entourée d'un cadre
  blanc de 100 px (vérifié sur un échantillon). Laissé en place, ce cadre occuperait 45 %
  des pixels d'entrée. Le prétraitement le **recadre** et réécrit l'image en 640×512
  (JPEG qualité 95).
- **Boîtes orientées** : une ligne vaut `classe x1 y1 … x4 y4`, normalisée sur 840×712. La
  tête YOLO ne produit que des boîtes droites (« Hors périmètre » de M11). Chaque boîte
  devient donc sa **boîte englobante** : min/max des 4 coins, décalés de 100 px, rognés à
  l'image puis renormalisés. Une boîte englobante est plus grande que la boîte orientée
  pour un véhicule en biais, d'où des recouvrements plus forts entre voisins.
- **Sortie** : `data/dronevehicle/<split>/{images,labels}/<n°>.{jpg,txt}` en YOLO
  `classe cx cy w h`. Le nom perd le suffixe Roboflow `_jpg.rf.<hash>`. 2,2 Go,
  ≈ 20 s sur 16 cœurs, et le script est idempotent.
- `--keep-border` garde les images 840×712 telles quelles et ne convertit que les boîtes.
- Le jeu contient des images RGB de nuit, presque noires : une piste pour la calibration
  INT8, comme ExDark.

### HIT-UAV — aucun prétraitement sur disque

- **Brut** : `data/HIT-UAV: A High-altitude Infrared Thermal Dataset/hit-uav/{images,labels}/{train,val,test}`,
  relié en `data/hituav`. Le `dataset.yaml` d'origine pointe vers `/tmp/dataset` et n'est
  pas lu.
- **Format** : YOLO txt (`classe cx cy w h` normalisés), images JPEG 640×512 en mode `L`
  (un canal), lues par `parse_yolo_txt`.
- **DontCare** (classe 4, 148 boîtes) est retiré par le chargeur, comme dans KITTI.
  Restent Person, Car, Bicycle et OtherVehicle (148 objets, presque vide).
- **Un canal** : l'affinage se fait à `CH=1` (cfg `tiny-yolov3-hituav-c1.cfg`, L00 copiée
  des poids RGB par somme sur les canaux), comme FLIR (T11.7). Les poids pré-entraînés
  reçoivent l'image grise recopiée sur 3 canaux (`letterbox`, `image_mode`).
- Le nom de fichier code les conditions de prise de vue : `<période>_<altitude m>_<angle
  °>_…_<n°>`. L'altitude va de 60 à 130 m et l'angle de 30 à 90°. Le premier champ
  sépare deux périodes (917 images en 0, 1 949 en 1) : jour ou nuit, sens à confirmer sur
  la page du jeu.

### UAVDT — séquences M seules et JSON compact (`tools/prep_datasets.py uavdt`)

- **Brut** : `data/UAVDT Dataset/{train,test}/{img,ann,meta}`. C'est l'export Supervisely
  de DatasetNinja : un JSON par image, avec des boîtes `points.exterior` (coin bas droit
  inclus), des tags d'objet (occlusion, sortie du champ, identifiant de cible) et des tags
  d'image (jour, nuit, brouillard, altitude, vue).
- **Séquences S écartées** : le test de l'export mélange les 20 séquences M du benchmark
  DET (16 592 trames) et les séquences S du suivi d'un seul objet (37 084 trames, une seule
  boîte de classe `vehicle`). Celles-ci fausseraient la mAP et ne sont pas gardées. Le
  train ne contient que des séquences M (30 séquences, 24 143 trames).
- **Lecture** : relire 40 000 JSON à chaque chargement prendrait plusieurs minutes. Le
  prétraitement écrit `data/uavdt/annotations_{train,test}.json`, une liste compacte
  (image, taille, coins continus, classes, séquence, tags d'image, occlusion et sortie par
  objet). Il crée aussi `data/uavdt/images/{train,test}/` en **liens physiques** vers les
  images M, sans copie, ce qui laisse les séquences S hors de l'archive du Drive.
- **Régions ignorées** : le jeu d'origine en a (`*_gt_ignore.txt`), l'export n'en a pas.
  Une détection dans une zone ignorée compte donc comme fausse alarme, comme pour VisDrone
  avant T15.10.
- **Redondance** : les trames consécutives sont presque identiques (≈ 800 par séquence).
  Une époque sur 24 143 images vaut beaucoup moins que 24 143 images indépendantes, ce
  qu'il faut garder en tête pour le balayage lot × sous-ensemble de M15.
- **Écart train ↔ test** : les objets du test sont plus petits (variation totale de
  l'aire à l'entrée 0,32, la plus forte des quatre jeux).

## Conventions communes

Celles de M11 (chargeur, `MAPPINGS`), M14 (notebooks générés, `--check`) et M16 (fiche,
`stats`) s'appliquent sans changement. Ajouter un jeu revient à toucher ces fichiers :

| Fichier | Ajout |
|---|---|
| `python/yolo/data/datasets.py` | classes, chargeur, entrée `DATASETS`, `MAPPINGS` vers VOC et COCO |
| `tools/prep_datasets.py` | prétraitement sur disque, s'il y en a un |
| `tools/get_datasets.sh` | `MARKER`, `HOWTO`, lien ou prétraitement dans `m18()`, liste de `check` |
| `tools/data_stats.py` | `FICHES` ; `raw_annotations` si le chargeur retire des boîtes |
| `tools/notebooks/matrice.py` | `TRAINABLE`, `TRAIN_INPUT` (1 canal), `SPLIT_IMAGES`, `DATA_MARKERS`, `REGISTRATION`, `TASKS`, `STATS_TASKS`, `READS_HEADERS` |
| `tools/notebooks/gabarits.py` | question propre du notebook `stats` (`QUESTIONS`) |
| `tools/m11.sh` | profils `<jeu>-prep` et `<jeu>-train` |
| tests | `test_datasets.py`, `test_prep_datasets.py` |

puis `python -m tools.notebooks <jeu>` (ou `make notebooks`).

### Questions par jeu

| Jeu | Question propre (notebook `stats`) | Renvoi |
|---|---|---|
| auair | que reste-t-il des objets d'une trame 1920×1080 réduite à 416, classe par classe ? | T18.1 |
| dronevehicle | véhicules par image face aux 256 emplacements, et leur taille à 416 | T18.2 |
| hituav | distribution des niveaux thermiques, et ce qu'en garde l'INT8 de L00 | T18.3 |
| uavdt | `letterbox` ou `stretch` pour du 1024×540 : combien d'objets passent sous 8 px de haut ? | T18.4 |

---

## A. Données

### [x] T18.0 — Arborescences et `tools/get_datasets.sh`
- **Dépend de** : T11.0 · **Taille** : S
- **Livrables** : `MARKER` et `HOWTO` des quatre jeux ; fonction `m18()` appelée par
  `kaggle` (liens `auair` et `hituav`, `prep_datasets.py` pour `dronevehicle` et `uavdt`
  si leur témoin manque) ; les quatre jeux dans `check`.
- **Acceptation** : `tools/get_datasets.sh check` donne les quatre jeux « prêt » ;
  `pack` archive `data/<jeu>` liens suivis.
- **Notes** : pas de `KAGGLE_ID` tant que les identifiants des miroirs ne sont pas relevés.
  Sur Colab, la voie est donc l'archive du Drive (T18.9).

### [x] T18.1 — AU-AIR
- **Dépend de** : T18.0 · **Taille** : S
- **Livrables** : `AUAIR_CLASSES`, `load_auair`, `AUAIR_VAL` (découpage par vidéo) ;
  `test_auair`.
- **Acceptation** : 27 880 + 4 943 images, 131 977 objets (132 031 bruts, moins les 54
  boîtes vides relevées par `raw_annotations`).

### [x] T18.2 — DroneVehicle
- **Dépend de** : T18.0 · **Taille** : M
- **Livrables** : `tools/prep_datasets.py dronevehicle` (recadrage, OBB → boîte
  englobante) ; `parse_yolo_txt`, `load_dronevehicle` ; tests `test_obb_to_yolo_border`,
  `test_prep_dronevehicle`, `test_dronevehicle`.
- **Acceptation** : 17 325 images en 640×512, 278 078 objets (autant que de lignes brutes) ;
  boîtes contrôlées à l'œil sur des images de jour et de nuit.

### [x] T18.3 — HIT-UAV
- **Dépend de** : T18.0 · **Taille** : S
- **Livrables** : `HITUAV_CLASSES` (sans DontCare), `load_hituav` ; `test_yolo_txt_and_hituav`.
- **Acceptation** : 2 008 / 287 / 571 images, 24 751 objets (DontCare retiré) ; images
  lues en 1 canal par l'affinage `CH=1`.

### [x] T18.4 — UAVDT
- **Dépend de** : T18.0 · **Taille** : M
- **Livrables** : `tools/prep_datasets.py uavdt` (séquences M, JSON compact, liens
  physiques) ; `load_uavdt` ; `test_prep_uavdt`, `test_uavdt`.
- **Acceptation** : train 24 143 images, 422 911 objets ; test 16 592 images, 375 884
  objets ; 37 084 trames S écartées.

## B. Registre et notebooks

### [x] T18.5 — Fiches et correspondances
- **Dépend de** : T18.1 à T18.4 · **Taille** : S
- **Livrables** : quatre entrées `FICHES` de `tools/data_stats.py` ; `raw_annotations`
  pour auair (boîtes brutes) et hituav (DontCare) ; `MAPPINGS` vers VOC et COCO ;
  `test_m18_mappings`.
- **Reste** : licence de HIT-UAV à relever sur la page du jeu.

### [x] T18.6 — Matrice et profils
- **Dépend de** : T18.5 · **Taille** : S
- **Livrables** : `matrice.py` (les quatre jeux dans `TRAINABLE`, hituav à 1 canal,
  `SPLIT_IMAGES`, témoins, tâches) ; `QUESTIONS` de `gabarits.py` ; profils
  `<jeu>-prep` et `<jeu>-train` de `tools/m11.sh` (`CH=1` pour hituav).

### [x] T18.7 — Génération des notebooks
- **Dépend de** : T18.6 · **Taille** : S
- **Livrables** : `notebooks/{auair,dronevehicle,hituav,uavdt}/`, 6 notebooks par jeu :
  `<jeu>_stats`, `tiny-yolov2-voc_infer`, `tiny-yolov3-coco_infer`,
  `tiny-yolov3-<jeu>_{infer,train,sweep}` ; `notebooks/README.md` régénéré ;
  `analyse_balayages.ipynb` réexécuté (sa liste de balayages a changé).
- **Acceptation** : `python -m tools.notebooks all --check` passe pour ces notebooks.
  `data_stats.py` passe sur les quatre jeux (`--sample 8`), et `eval_voc.py` aussi (poids
  COCO, 20 images, en simple essai de la chaîne).

## C. Exécution

### [ ] T18.8 — Notebooks `stats` exécutés
- **Dépend de** : T18.7 · **Taille** : S
- **Livrables** : les quatre `<jeu>_stats.ipynb` exécutés et versionnés avec leurs sorties ;
  sections dans [stats-jeux.md](stats-jeux.md) (`tools/data_stats.py --report`).
- **Notes** : refaire après `tools/m11.sh <jeu>-prep`, pour que les collisions utilisent
  les ancres du jeu et non celles de COCO.

### [ ] T18.9 — Envoi sur le Drive
- **Dépend de** : T18.0 · **Taille** : S
- **Livrables** : `tools/get_datasets.sh push auair dronevehicle hituav uavdt` ; lignes
  du tableau de [donnees-drive.md](donnees-drive.md).
- **Notes** : ≈ 2,4 + 2,2 + 0,2 + 6,5 Go (`du -shL`). UAVDT est le plus long à envoyer.
  Vérifier la place (`rclone about gdrive:`).
- **Status (2026-10-07)**: a first `push` was stopped by hand during the AU-AIR upload.
  `data_archives/auair.tar` (2,477,015,040 bytes) is complete locally but not on the
  Drive. The other three archives are not built yet. The Drive has 43.7 GiB free. To
  resume, run the same `push` command (`pack` rebuilds `auair.tar`).

### [ ] T18.10 — Hors domaine et affinages
- **Dépend de** : T18.8, T18.9 · **Taille** : L
- **Livrables** : mAP des poids VOC et COCO sans affinage (notebooks `_infer`, comme
  T11.2) ; balayages et runs affinés (`_sweep`, `_train`), suivis en
  [M15](M15-campagne-entrainement.md).
- **Notes** :
  - HIT-UAV et UAVDT ont 84 % et 97 % d'objets plus petits que 32² à 416 : ce sont les
    meilleurs arguments chiffrés pour une entrée plus grande ou des tuiles (T15.9, T15.16).
  - Sur UAVDT, sous-échantillonner les trames, ou se fier au sous-ensemble du balayage,
    plutôt qu'aux époques.

## Hors périmètre

- Boîtes orientées de DroneVehicle : la tête n'en produit pas (comme DOTA en M11).
- Moitié infrarouge de DroneVehicle : absente de cette version. Si elle arrive, elle
  formera un jeu à part, à 1 canal comme HIT-UAV.
- Suivi d'objet (séquences S d'UAVDT, identifiants de cible).

## Ordre conseillé

T18.0 → T18.1 à T18.4 → T18.5 → T18.6 → T18.7 sont faites. Ensuite T18.9 (Drive, pour
Colab), puis T18.8, puis T18.10.
