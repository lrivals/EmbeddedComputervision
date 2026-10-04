# M3 — Inférence et évaluation

Objectif : de la sortie brute du réseau aux boîtes finales, et une mAP VOC qui servira de
référence à toutes les étapes suivantes.

### [ ] T3.1 — Décodage des têtes
- **Spec** : §8.1 · **Dépend de** : T1.8, T2.1 · **Taille** : S
- **Livrables** : `python/yolo/infer/decode.py` + tests
- **Acceptation** : décodage(encodage(boîte)) == boîte à 1e-9 près (cohérence avec T2.3) ;
  score v3 $\sigma(t_o)\sigma(t_c)$ et v2 $\sigma(t_o)\,\text{softmax}_c$ ; boîtes ramenées
  aux coordonnées de l'image d'origine (letterbox inverse)

### [ ] T3.2 — Seuil et NMS
- **Spec** : §8.2 · **Dépend de** : T3.1 · **Taille** : S
- **Livrables** : `python/yolo/infer/nms.py` + tests
- **Acceptation** : NMS par classe ; résultats identiques à une implémentation naïve O(n²)
  sur 1 000 jeux aléatoires
- **Notes** : seuils par défaut `conf = 0,25`, `iou = 0,45` (hors base), en paramètres

### [ ] T3.3 — mAP PASCAL VOC
- **Spec** : §8.3 · **Dépend de** : T3.2, T0.6 · **Taille** : M
- **Livrables** : `python/yolo/infer/metrics.py`, `tools/eval_voc.py`
- **Acceptation** : AP 11 points ; objets `difficult` ignorés ; une vérité appariée une
  seule fois ; résultats identiques au script officiel VOC sur un jeu de détections de
  test fourni

### [ ] T3.4 — Démo et mAP flottante de référence
- **Spec** : §10.5 étape 1 · **Dépend de** : T3.3, T1.9 · **Taille** : S
- **Livrables** : `tools/detect.py` (image → image annotée), `results/map_float.md`
- **Acceptation** : mAP VOC2007 test de Tiny-YOLOv2 VOC (poids Darknet) proche de la
  valeur publiée par Darknet ; écart expliqué s'il dépasse 1 point
- **Notes** : **jalon de validation** : cette mAP est la référence des étapes entière
  (M4) et matérielle (M8)
