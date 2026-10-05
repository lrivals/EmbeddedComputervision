# M4 — Quantification et modèle entier bit-exact

Objectif : un modèle Python qui ne calcule qu'en entiers, exactement comme le fera le
matériel, avec une perte de mAP mesurée, puis l'export qui sert de contrat aux étages C++
et HLS.

### [x] T4.1 — Fusion de la BN
- **Spec** : §9.1 · **Dépend de** : T1.8 · **Taille** : S
- **Livrables** : `python/yolo/quant/fuse_bn.py` + tests
- **Acceptation** : réseau fusionné == réseau avec BN (mode inférence) à 1e-12 en float64 ;
  mAP inchangée
- **Fait** : `fuse_bn` (§9.1) et `fuse_network` (même graphe, convs sans BN avec biais) ;
  égalité à 1e-12 (relative au max) sur v2 et v3 à BN aléatoires, mêmes détections avec les
  poids Darknet. `Network.forward(all_outputs=True)` rend toutes les cartes (calibration).

### [x] T4.2 — Quantification des poids et calibration
- **Spec** : §9.2 · **Dépend de** : T4.1, T3.3 · **Taille** : M
- **Livrables** : `python/yolo/quant/{quantize,calibrate}.py`
- **Acceptation** : poids int8 symétriques **par canal de sortie** ; échelles
  d'activation par couche calibrées sur ~500 images ; comparaison des percentiles 90, 95,
  99 et du max, choix documenté
- **Notes** : comme 2026-fata, choisir l'écrêtage qui minimise l'erreur FP32/INT8
- **Fait** : `quantize.py` (arrondi ⌊v+½⌋, poids par canal s_w = max|W_f|/127, biais int32,
  M0 < 2³¹ avec n ≤ 31 maximal par couche, entrée à 1/127) ; `calibrate.py` + 
  `tools/calibrate.py` (`make calibrate`) : 500 images VOC2007 **trainval** tirées au hasard,
  4 096 valeurs échantillonnées par image et par couche, candidats p90/p95/p99/p99,9/p99,99/max,
  critère MSE FP32/fake-INT8 : p99,99 retenu partout (p90-p99 écrêtent trop, le max gaspille
  la plage) — `results/calibration_<net>.md`. Route : échelle commune imposée aux sources
  (v3 : {L08, L18}, la plus grande). **Têtes** : échelle fixe 1/8 et non 1/16 (voir T4.4).

### [x] T4.3 — Couches entières
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
- **Fait** : `int_layers.py` (`rshift_round`, `conv_int`, `leaky_int`…) et `int_model.py`
  (`QuantModel`, `IntNetwork`, `fake_quant_forward`) ; contrat écrit dans
  `docs/conventions.md` (« Arithmétique entière »). Écart à la couche flottante ≤ 1,5 % de
  la dynamique (max) avec les poids Darknet ; couche 13 : borne Σ|q_w|·127 + |q_b| = 2^25,7
  (pire cas générique 9 216 × 127² < 2³⁰) ; −128 jamais produit. Moteur `f64` (produit
  d'entiers par BLAS, exact car |acc| < 2⁵³) bit à bit égal au moteur `int64`, 4× plus rapide.
  **Écart à la spec** : leaky arrondie `(13y + 64) >> 7` au lieu de `(13y) >> 7` ; le
  plancher biaise chaque sortie négative de −½ pas et le biais se cumule (écart moyen aux
  têtes ×1,7 sur une image).

### [x] T4.4 — Sigmoïde et exponentielle par LUT
- **Spec** : §9.4 · **Dépend de** : T4.3 · **Taille** : S
- **Livrables** : `python/yolo/quant/lut.py`, décodage entier dans `decode.py`
- **Acceptation** : LUT de 256 entrées, erreur max sur σ ≤ 0,0075 (pas 1/16) ; seuillage
  sur le logit $t_o > \ln\frac{\theta}{1-\theta}$ identique au seuillage après sigmoïde
- **Fait** : `lut.py` (σ et e^{t} indexées par `q + 128`, e^{d·s} pour le softmax v2 en
  trois étages, valeurs Q16 ; e^{t} en Q`exp_frac` pour tenir sur 32 bits),
  `decode_head_int`/`decode_int` (seuil sur l'entier t_o puis tables, boîtes en flottant
  côté hôte) et `postprocess_int`. Seuil sur le logit identique à σ > θ et à LUT > θ pour
  tout q. Erreur de la table aux points de grille ≤ 2⁻¹⁷.
  **Écart à la spec** : de bout en bout, l'erreur max vaut σ'(0)·s/2 = 1/128 ≈ 0,0078 au
  pas 1/16 (et non 0,0075). Surtout, le pas 1/16 sature les logits de Tiny-YOLOv2
  (p99,99 de |t| ≈ 19) : 2,5 % des sorties écrêtées, softmax aplati, mAP 49,7 contre 57,2
  en flottant (500 images, letterbox). Les têtes sont donc au pas **1/8** (erreur σ ≤ 1/64 ≈ 0,016,
  écrêtage 0,03 %), mAP intacte ; une échelle calibrée (p99,99, 0,149) fait 56,6.

### [x] T4.5 — mAP du modèle entier
- **Spec** : §10.5 étape 2 · **Dépend de** : T4.4, T3.4 · **Taille** : S
- **Livrables** : `results/map_int8.md`
- **Acceptation** : mAP entière et écart au flottant rapportés, couche la plus sensible
  identifiée (quantification d'une seule couche à la fois)
- **Notes** : cible indicative : moins d'un point de perte (2025-kim : -0,2 % en INT16)
- **Fait** : `tools/eval_quant.py` (`make eval-int` ; variantes `float`, `int`, `fq:all`,
  `fq:<id>`, en parallèle) → `results/map_int8.md`. VOC2007 test complet, prétraitement
  de référence (stretch) : flottant **56,30**, entier **55,66**, soit **−0,64 point** ;
  letterbox : 54,18 → 53,68 (−0,50). Couche la plus sensible seule : L10
  (−0,61 sur 1 000 images, au niveau du bruit ; aucune ne domine).

### [x] T4.6 — QAT (si nécessaire) — sans objet
- **Spec** : §9.2 (PTQ ou QAT) · **Dépend de** : T4.5, T2.7 · **Taille** : L
- **Livrables** : `python/yolo/quant/fake_quant.py` (STE), mode QAT du trainer
- **Acceptation** : gradcheck du fake-quant avec STE ; écart de mAP réduit sous la cible
- **Notes** : à déclencher seulement si T4.5 dépasse la cible ; lr = 1e-4 (2026-fata)
- **Décision** : non déclenché, la perte PTQ (−0,64 point en stretch, −0,50 en
  letterbox) est sous la cible de 1 point. À
  reprendre si un réseau affiné (T2.9) ou une largeur de bits plus faible la dépasse.

### [x] T4.7 — Export du modèle entier
- **Spec** : §10.5 étapes 2-3 · **Dépend de** : T4.3, T0.4 · **Taille** : M
- **Livrables** : `python/yolo/io/export.py`, `tools/export_model.py`, sortie dans `model/`
- **Acceptation** : manifest valide (schéma T0.4) ; blobs relus et réinjectés dans le
  modèle entier Python == sortie d'origine ; dumps `L00.npy`… pour 3 images de test
- **Fait** : `export.py` (`layout` générique : ping-pong A/B, `S` pour une carte lue deux
  fois, `R` pour les routes, `H<grille>` pour les têtes ; il reproduit exactement
  `model/example_manifest.json`, dont le générateur est devenu une surcouche ;
  `export_model`, `load_model`, `load_luts`), `tools/export_model.py` (`make export`) →
  `model/tiny-yolov2-voc/` et `model/tiny-yolov3-coco/` (poids COCO : pas de poids
  Tiny-YOLOv3 VOC avant T2.9). Schéma étendu de façon compatible : couche `region`, ancres
  non entières (34,56 px), `luts.bin`, `lut_offset`/`exp_frac` par tête. Dumps
  `dumps/<id>/{input,L00…}.npy` + `detections.json` pour VOC2007 test 000001-000003.
