# M2 — Cibles, perte, entraînement

Objectif : une perte YOLOv3 dont tous les gradients sont vérifiés, une boucle
d'entraînement capable de surapprendre une image, puis un affinage sur VOC.

### [x] T2.1 — IoU
- **Spec** : §5.3, §5.2 (`iou_wh`) · **Dépend de** : T0.2 · **Taille** : S
- **Livrables** : `python/yolo/infer/boxes.py` (conversions centre/coins, IoU vectorisée
  N×M, `iou_wh`) + tests
- **Acceptation** : cas limites (disjointes, incluses, identiques, IoU = 0 au contact)
- **Fait** : union nulle → IoU 0 ; `iou` (cxcywh), `iou_xyxy`, `iou_wh` en matrices (N, M).

### [x] T2.2 — Ancres par k-means
- **Spec** : §5.2 · **Dépend de** : T2.1, T0.6 · **Taille** : S
- **Livrables** : `python/yolo/data/anchors.py`, `tools/kmeans_anchors.py`
- **Acceptation** : k = 5 (v2) et k = 6 (v3) sur VOC ; IoU moyenne au plus proche
  centroïde rapportée et comparée aux ancres Darknet ; graine fixée
- **Notes** : distance $1 - \text{IoU}$, jamais euclidienne
- **Fait** (`make anchors` → `results/anchors.md`, 40 058 boîtes VOC07+12 trainval en px du
  letterbox 416) : k = 5 → IoU moyenne 0,611 (YOLOv2 rapporte 61,0 pour 5 clusters), ancres
  `yolov2-tiny-voc.cfg` 0,559 ; k = 6 → 0,631, ancres `yolov3-tiny.cfg` (COCO) 0,603.
  Initialisation **k-means++** avec d = 1 − IoU (hors base) : le tirage uniforme du §5.2 tombe
  facilement dans un minimum local (deux centroïdes dans le même amas).

### [x] T2.3 — Encodage des cibles et masques
- **Spec** : §5.1 · **Dépend de** : T2.1 · **Taille** : M
- **Livrables** : `python/yolo/data/targets.py` + tests
- **Acceptation** : pour un jeu de boîtes construit à la main : bonne cellule, bonne ancre
  (parmi les 6, toutes échelles), cibles $x^*, y^*, t_w^*, t_h^*$ exactes, masques
  obj / ignore / noobj corrects
- **Notes** : le masque *ignore* dépend des boîtes **prédites** : il se calcule dans la
  perte, à chaque itération, pas au chargement des données
- **Fait** : ancres converties en fraction par /416 quelle que soit la taille d'entrée
  (conventions.md) ; cellule bornée à S − 1 pour g = 1 ; collision même cellule/même ancre :
  le dernier objet l'emporte (Darknet, hors base). `heads(net)` décrit les têtes `yolo`/`region`.

### [x] T2.4 — Perte YOLOv3 et gradients
- **Spec** : §6.2 · **Dépend de** : T2.3, T1.6 · **Taille** : M
- **Livrables** : `python/yolo/train/loss.py` + tests
- **Acceptation** : gradcheck de la perte complète ≤ 1e-7 sur le mini-réseau du §11
  (2 images, 3 objets) ; gradients analytiques == tableau du §6.2
- **Notes** : coordonnées sur $\sigma(t_x)$ (variante hors base) ; poids
  $\omega = 2 - g_w g_h$ ; ancres *ignore* exclues de tous les termes
- **Fait** : `yolo_loss` rend la perte (somme sur le lot), les δ des têtes, les parts
  coord/obj/noobj/cls et les termes élémentaires (gradcheck terme à terme). Gradcheck de la
  perte complète ≤ 1e-7 sur le mini-réseau du §11 (entrée 16×16, 2 images, 3 objets, cellules
  obj/ignore/noobj toutes présentes, aucune IoU à moins de 1e-3 du seuil) ; gradients ==
  tableau du §6.2 à 1e-12. `ignore_thresh` = 0,5 par défaut (spec ; les `.cfg` ont 0,7),
  `lambda_coord` = 1.

### [x] T2.5 — Perte YOLOv2 (et variantes)
- **Spec** : §2.2, §6.1, §6.3 · **Dépend de** : T2.4 · **Taille** : M
- **Livrables** : softmax de classes pour Tiny-YOLOv2 dans `loss.py` ; GIoU/CIoU en option
- **Acceptation** : gradcheck ≤ 1e-7 pour chaque variante implémentée
- **Notes** : GIoU/DIoU/CIoU sont hors base (§6.3) ; à faire seulement si utile
- **Fait** : `class_mode="softmax"` (entropie croisée, softmax stabilisé), choisi
  automatiquement pour une couche `region` ; gradcheck ≤ 1e-7. GIoU/DIoU/CIoU non faits.

### [x] T2.6 — Pipeline de données
- **Spec** : §1, §7.1 (augmentation) · **Dépend de** : T0.6 · **Taille** : M
- **Livrables** : `python/yolo/data/{letterbox,augment,loader}.py`
- **Acceptation** : letterbox réversible (boîtes reprojetées à 1e-6 près) ; augmentation
  géométrique ±20 % et HSV ×1,5 appliquée identiquement aux boîtes ; visualisation de
  contrôle
- **Notes** : chargement parallèle (multiprocessing) pour ne pas affamer la boucle
- **Fait** : géométrie = une seule affinité (letterbox, puis échelle et translation
  ±20 % autour du centre, flip horizontal optionnel, hors base) appliquée à l'image (PIL) et
  aux coins des boîtes ; boîtes rognées, retirées sous 2 px. HSV en NumPy. Test : rectangle
  blanc synthétique == boîte transformée à 1,5 px. Objets `difficult` retirés à
  l'entraînement (Darknet). Chargeur : `multiprocessing.Pool`, préchargement borné, tirages
  par `default_rng([graine, époque, image])` → identique en séquentiel et en parallèle.
  Contrôle visuel : `python tools/show_augment.py` → `build/augment_samples/`.

### [x] T2.7 — Boucle d'entraînement
- **Spec** : §7.1 · **Dépend de** : T2.4, T2.6, T1.8 · **Taille** : M
- **Livrables** : `python/yolo/train/{optim,schedule,trainer}.py`, `tools/train.py`
- **Acceptation** : SGD momentum 0,9 + weight decay 5e-4 vérifiés sur un problème
  quadratique ; warm-up ; paliers ; multi-échelle toutes les 10 itérations (320 à 608) ;
  sauvegarde et reprise des checkpoints
- **Notes** : pas de weight decay sur γ, β et biais (convention usuelle, hors base)
- **Fait** : perte et gradients divisés par N dans le trainer ; montée
  lr·((it+1)/burn_in)⁴ (Darknet, hors base) ; taille multi-échelle = fonction pure de
  (graine, it // 10), donc reprise de checkpoint bit à bit (testée). Transfert COCO → VOC :
  `copy_matching` (tout sauf les têtes 15 et 22). Mesure : `tools/train.py --init coco`,
  lot 8 à 416, float32, 4 workers : ≈ 3,6 s/itération (≈ 0,45 s/image).

### [x] T2.8 — Surapprentissage d'une image
- **Spec** : §11 · **Dépend de** : T2.7 · **Taille** : S
- **Livrables** : `python/tests/test_overfit.py` (marqué `slow`)
- **Acceptation** : la perte tend vers 0 ; après décodage (T3.1-T3.2), les boîtes prédites
  coïncident avec la vérité (IoU > 0,9)
- **Fait** (`make test-slow`, ≈ 1 min) : Tiny-YOLOv3-VOC, init He, image synthétique
  256×256 à 3 objets, 200 itérations SGD (lr 1e-3, montée 50) : perte 893 → 0,11 ; meilleure
  boîte de chaque classe : score > 0,99, IoU > 0,997.

### [ ] T2.9 — Affinage sur VOC
- **Spec** : §7.1 · **Dépend de** : T2.8, T1.9, T3.3 · **Taille** : L
- **Livrables** : poids affinés dans `weights/`, courbe de perte, rapport dans `results/`
- **Acceptation** : mAP VOC2007 test d'un Tiny-YOLOv3 affiné depuis les poids COCO,
  rapportée et comparée à la référence Darknet
- **Notes** : **risque principal du projet** : la lenteur de NumPy. Mesurer le temps par
  itération dès T1.2 ; si nécessaire, sous-ensemble de VOC ou moins de classes.
  M3 et M4 ne dépendent pas de cette tâche (raccourci T1.9)
- **Prêt** : `tools/train.py --net tiny-yolov3-voc --init coco` (checkpoints, `loss.csv`,
  `final.weights`). Au rythme mesuré en T2.7 (≈ 0,45 s/image à 416), une époque VOC07+12
  (16 551 images) prend ≈ 2 h. À surveiller : la perte noobj initiale est très élevée avec
  des têtes initialisées à la He (≈ 2 600 par image), elle s'effondre en quelques itérations.
