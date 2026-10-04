# M4 — Quantification et modèle entier bit-exact

Objectif : un modèle Python qui ne calcule qu'en entiers, exactement comme le fera le
matériel, avec une perte de mAP mesurée, puis l'export qui sert de contrat aux étages C++
et HLS.

### [ ] T4.1 — Fusion de la BN
- **Spec** : §9.1 · **Dépend de** : T1.8 · **Taille** : S
- **Livrables** : `python/yolo/quant/fuse_bn.py` + tests
- **Acceptation** : réseau fusionné == réseau avec BN (mode inférence) à 1e-12 en float64 ;
  mAP inchangée

### [ ] T4.2 — Quantification des poids et calibration
- **Spec** : §9.2 · **Dépend de** : T4.1, T3.3 · **Taille** : M
- **Livrables** : `python/yolo/quant/{quantize,calibrate}.py`
- **Acceptation** : poids int8 symétriques **par canal de sortie** ; échelles
  d'activation par couche calibrées sur ~500 images ; comparaison des percentiles 90, 95,
  99 et du max, choix documenté
- **Notes** : comme 2026-fata, choisir l'écrêtage qui minimise l'erreur FP32/INT8

### [ ] T4.3 — Couches entières
- **Spec** : §9.3 · **Dépend de** : T4.2 · **Taille** : L
- **Livrables** : `python/yolo/quant/int_layers.py`, `python/yolo/quant/int_model.py` + tests
- **Acceptation** : conv int8 avec accumulateur int32 (débordement testé : 30 bits au pire
  pour la couche 13) ; requantification $(acc \cdot M_0 + 2^{n-1}) \gg n$ ; leaky
  $(y \cdot 13) \gg 7$ ; maxpool s2/s1 et upsample en entier ; écart à la couche
  flottante ≤ 2 % de la dynamique
- **Notes** :
  - **route** : les deux entrées de la concaténation (couches 19 et 8) doivent partager
    **la même échelle**, sinon la concaténation « gratuite » du §10.3 est impossible.
    Imposer l'échelle commune lors de la calibration
  - arrondi et décalage arithmétique sur entiers signés : définir précisément le
    comportement pour les négatifs, le C++ et le HLS devront le reproduire
  - tout en `np.int64`, sans aucun flottant dans la passe avant

### [ ] T4.4 — Sigmoïde et exponentielle par LUT
- **Spec** : §9.4 · **Dépend de** : T4.3 · **Taille** : S
- **Livrables** : `python/yolo/quant/lut.py`, décodage entier dans `decode.py`
- **Acceptation** : LUT de 256 entrées, erreur max sur σ ≤ 0,0075 (pas 1/16) ; seuillage
  sur le logit $t_o > \ln\frac{\theta}{1-\theta}$ identique au seuillage après sigmoïde

### [ ] T4.5 — mAP du modèle entier
- **Spec** : §10.5 étape 2 · **Dépend de** : T4.4, T3.4 · **Taille** : S
- **Livrables** : `results/map_int8.md`
- **Acceptation** : mAP entière et écart au flottant rapportés, couche la plus sensible
  identifiée (quantification d'une seule couche à la fois)
- **Notes** : cible indicative : moins d'un point de perte (2025-kim : -0,2 % en INT16)

### [ ] T4.6 — QAT (si nécessaire)
- **Spec** : §9.2 (PTQ ou QAT) · **Dépend de** : T4.5, T2.7 · **Taille** : L
- **Livrables** : `python/yolo/quant/fake_quant.py` (STE), mode QAT du trainer
- **Acceptation** : gradcheck du fake-quant avec STE ; écart de mAP réduit sous la cible
- **Notes** : à déclencher seulement si T4.5 dépasse la cible ; lr = 1e-4 (2026-fata)

### [ ] T4.7 — Export du modèle entier
- **Spec** : §10.5 étapes 2-3 · **Dépend de** : T4.3, T0.4 · **Taille** : M
- **Livrables** : `python/yolo/io/export.py`, `tools/export_model.py`, sortie dans `model/`
- **Acceptation** : manifest valide (schéma T0.4) ; blobs relus et réinjectés dans le
  modèle entier Python == sortie d'origine ; dumps `L00.npy`… pour 3 images de test
