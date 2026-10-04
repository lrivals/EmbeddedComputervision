# M9 — Extensions (optionnel)

Pistes à ouvrir une fois M8 terminé. Chacune devient un jalon à part entière, découpé dans
le même format.

### [ ] T9.1 — Post-traitement matériel
- **Spec** : §10.3 (2024-zhang) · **Dépend de** : T8.1
- Décodage par LUT et NMS sans tri dans le FPGA. Acceptation : boîtes == golden ; référence
  publiée : 845 candidates traitées en 0,38 ms.

### [ ] T9.2 — Quantification REQ-YOLO (ADMM, puissances de 2)
- **Spec** : §9.2 (2019-ding) · **Dépend de** : T4.6
- Poids 6 bits « mixed powers-of-two » : multiplication = 2 décalages + 1 addition.
  Acceptation : mAP et économie de DSP comparées au modèle INT8.

### [ ] T9.3 — Quantification 4 bits
- **Spec** : §9.2, §9.3 (2024-yan) · **Dépend de** : T4.6
- Écrêtage appris contraint à une puissance de 2 ; première couche décomposée en deux
  convolutions 4 bits.

### [ ] T9.4 — Architecture streaming
- **Spec** : §10.1 (SATAY, 2018-venieris) · **Dépend de** : T8.3
- Un bloc par couche, line buffers $(K-1) \times W \times C$, paramètres sur puce. Comparer
  latence et ressources au moteur unique.

### [ ] T9.5 — Apprentissage en ligne sur FPGA
- **Spec** : §4 (EF-Train), §12 (gaps) · **Dépend de** : T8.3
- Piste de recherche ouverte : affiner la tête YOLO sur la carte (BN en pleine précision,
  §9.1). À croiser avec l'axe continual learning.
