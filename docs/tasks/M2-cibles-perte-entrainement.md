# M2 — Cibles, perte, entraînement

Objectif : une perte YOLOv3 dont tous les gradients sont vérifiés, une boucle
d'entraînement capable de surapprendre une image, puis un affinage sur VOC.

### [ ] T2.1 — IoU
- **Spec** : §5.3, §5.2 (`iou_wh`) · **Dépend de** : T0.2 · **Taille** : S
- **Livrables** : `python/yolo/infer/boxes.py` (conversions centre/coins, IoU vectorisée
  N×M, `iou_wh`) + tests
- **Acceptation** : cas limites (disjointes, incluses, identiques, IoU = 0 au contact)

### [ ] T2.2 — Ancres par k-means
- **Spec** : §5.2 · **Dépend de** : T2.1, T0.6 · **Taille** : S
- **Livrables** : `python/yolo/data/anchors.py`, `tools/kmeans_anchors.py`
- **Acceptation** : k = 5 (v2) et k = 6 (v3) sur VOC ; IoU moyenne au plus proche
  centroïde rapportée et comparée aux ancres Darknet ; graine fixée
- **Notes** : distance $1 - \text{IoU}$, jamais euclidienne

### [ ] T2.3 — Encodage des cibles et masques
- **Spec** : §5.1 · **Dépend de** : T2.1 · **Taille** : M
- **Livrables** : `python/yolo/data/targets.py` + tests
- **Acceptation** : pour un jeu de boîtes construit à la main : bonne cellule, bonne ancre
  (parmi les 6, toutes échelles), cibles $x^*, y^*, t_w^*, t_h^*$ exactes, masques
  obj / ignore / noobj corrects
- **Notes** : le masque *ignore* dépend des boîtes **prédites** : il se calcule dans la
  perte, à chaque itération, pas au chargement des données

### [ ] T2.4 — Perte YOLOv3 et gradients
- **Spec** : §6.2 · **Dépend de** : T2.3, T1.6 · **Taille** : M
- **Livrables** : `python/yolo/train/loss.py` + tests
- **Acceptation** : gradcheck de la perte complète ≤ 1e-7 sur le mini-réseau du §11
  (2 images, 3 objets) ; gradients analytiques == tableau du §6.2
- **Notes** : coordonnées sur $\sigma(t_x)$ (variante hors base) ; poids
  $\omega = 2 - g_w g_h$ ; ancres *ignore* exclues de tous les termes

### [ ] T2.5 — Perte YOLOv2 (et variantes)
- **Spec** : §2.2, §6.1, §6.3 · **Dépend de** : T2.4 · **Taille** : M
- **Livrables** : softmax de classes pour Tiny-YOLOv2 dans `loss.py` ; GIoU/CIoU en option
- **Acceptation** : gradcheck ≤ 1e-7 pour chaque variante implémentée
- **Notes** : GIoU/DIoU/CIoU sont hors base (§6.3) ; à faire seulement si utile

### [ ] T2.6 — Pipeline de données
- **Spec** : §1, §7.1 (augmentation) · **Dépend de** : T0.6 · **Taille** : M
- **Livrables** : `python/yolo/data/{letterbox,augment,loader}.py`
- **Acceptation** : letterbox réversible (boîtes reprojetées à 1e-6 près) ; augmentation
  géométrique ±20 % et HSV ×1,5 appliquée identiquement aux boîtes ; visualisation de
  contrôle
- **Notes** : chargement parallèle (multiprocessing) pour ne pas affamer la boucle

### [ ] T2.7 — Boucle d'entraînement
- **Spec** : §7.1 · **Dépend de** : T2.4, T2.6, T1.8 · **Taille** : M
- **Livrables** : `python/yolo/train/{optim,schedule,trainer}.py`, `tools/train.py`
- **Acceptation** : SGD momentum 0,9 + weight decay 5e-4 vérifiés sur un problème
  quadratique ; warm-up ; paliers ; multi-échelle toutes les 10 itérations (320 à 608) ;
  sauvegarde et reprise des checkpoints
- **Notes** : pas de weight decay sur γ, β et biais (convention usuelle, hors base)

### [ ] T2.8 — Surapprentissage d'une image
- **Spec** : §11 · **Dépend de** : T2.7 · **Taille** : S
- **Livrables** : `python/tests/test_overfit.py` (marqué `slow`)
- **Acceptation** : la perte tend vers 0 ; après décodage (T3.1-T3.2), les boîtes prédites
  coïncident avec la vérité (IoU > 0,9)

### [ ] T2.9 — Affinage sur VOC
- **Spec** : §7.1 · **Dépend de** : T2.8, T1.9, T3.3 · **Taille** : L
- **Livrables** : poids affinés dans `weights/`, courbe de perte, rapport dans `results/`
- **Acceptation** : mAP VOC2007 test d'un Tiny-YOLOv3 affiné depuis les poids COCO,
  rapportée et comparée à la référence Darknet
- **Notes** : **risque principal du projet** : la lenteur de NumPy. Mesurer le temps par
  itération dès T1.2 ; si nécessaire, sous-ensemble de VOC ou moins de classes.
  M3 et M4 ne dépendent pas de cette tâche (raccourci T1.9)
