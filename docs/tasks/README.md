# Plan de travail

Le projet suit les six étapes de la feuille de route du §10.5 de la
[spécification](../../yolo-embarque-de-zero.md), découpées en jalons.

| Jalon | Objet | Sortie vérifiable |
|---|---|---|
| [M0](M0-infrastructure.md) | Infrastructure | `make test` passe, comptage des MACs conforme au §3 |
| [M1](M1-briques-numpy.md) | Briques NumPy (§4) | Tiny-YOLOv2/v3 en passes avant et arrière, gradcheck vert |
| [M2](M2-cibles-perte-entrainement.md) | Cibles, perte, entraînement (§5-§7) | une image surapprise ; fine-tuning VOC |
| [M3](M3-inference-evaluation.md) | Inférence et évaluation (§8) | **mAP flottante de référence** |
| [M4](M4-quantification.md) | Quantification, modèle entier (§9) | mAP entière + export `model/` |
| [M5](M5-golden-cpp.md) | Golden model C++ (§10.2) | 0 écart avec les dumps Python |
| [M6](M6-accelerateur-hls.md) | Accélérateur HLS (§10.2-§10.3) | C-sim et co-sim == golden |
| [M7](M7-integration-soc.md) | Intégration SoC | sortie sur carte == golden |
| [M8](M8-mesures.md) | Mesures (§10.4) | `results/benchmarks.csv`, mAP aux 3 stades |
| [M9](M9-extensions.md) | Extensions (optionnel) | index des pistes |
| [M9.1](M9.1-postproc-materiel.md) | Post-traitement matériel | boîtes du FPGA == golden entier |
| [M9.2](M9.2-req-yolo.md) | REQ-YOLO (ADMM, puissances de 2) | mAP et DSP face à INT8 |
| [M9.3](M9.3-quant-4bits.md) | Quantification 4 bits | mAP 4 bits face à INT8 |
| [M9.4](M9.4-streaming.md) | Architecture streaming | tête == golden, latence face au moteur unique |
| [M10](M10-ameliorations.md) | Améliorations (optionnel) | Tiny-YOLOv2 ≤ 45 ms en C-sim, == golden |
| [M11](M11-jeux-de-donnees.md) | Autres jeux de données (optionnel) | mAP flottante / entière / FPGA hors VOC |
| [M12](M12-profils-pc.md) | Profils de test sur PC (avant la carte) | chaque évaluation a un profil R/M/N, un critère et une décision |

## Dépendances

```mermaid
graph LR
  M0 --> M1 --> M2 --> M3
  M1 -- "T1.9 poids Darknet" --> M3
  M3 --> M4 --> M5 --> M6 --> M7 --> M8
  M5 -- "T5.6 roofline" --> T60["T6.0 choix carte"] --> M6
  M8 --> M9
  M9 --> M91[M9.1] & M93[M9.3] & M94[M9.4]
  M93 --> M92[M9.2]
  M8 --> M10[M10]
  M91 --> M10
  M8 --> M11[M11]
  M8 --> M12[M12]
  M9 --> M12
```

- **Raccourci T1.9 → M3** : avec les poids Darknet pré-entraînés, la mAP de référence est
  disponible sans attendre un entraînement complet en NumPy. M2 peut avancer en parallèle
  de M3.
- **Choix de la carte (T6.0)** : seule décision bloquante avant le matériel ; elle dépend de
  l'outil roofline (T5.6). M0 à M5 sont indépendants de la carte.

## Format d'une tâche

```markdown
### T1.2 — Titre
- **Spec** : §4.1 · **Dépend de** : T1.1 · **Taille** : S | M | L
- **Livrables** : fichiers créés
- **Acceptation** : critère mesurable
- **Notes** : pièges, choix hors base
```

Tailles indicatives : **S** ≤ ½ journée, **M** 1-3 jours, **L** une semaine ou plus.

## Suivi

Cocher `[x]` dans le titre de la tâche quand le critère d'acceptation est atteint, et
reporter ici l'avancement par jalon.

| Jalon | Tâches | Faites |
|---|---|---|
| M0 | 6 | 6 |
| M1 | 9 | 8 |
| M2 | 9 | 8 |
| M3 | 4 | 4 |
| M4 | 7 | 7 (T4.6 sans objet) |
| M5 | 6 | 6 |
| M6 | 6 | 1 (T6.0-T6.3 en C-sim ; synthèse et co-sim en attente de Vitis) |
| M7 | 4 | 0 (T7.1-T7.4 écrits et vérifiés sur PC, backend sim ; carte et Vivado en attente) |
| M8 | 3 | 0 (T8.1-T8.3 outillés ; projection C-sim et mAP FPGA en C-sim = entier (55,66, 4 952 images identiques) ; mesures carte en attente) |
| M9 | 1 (T9.5) | 0 |
| M9.1 | 3 | 2 (T9.1.3 vérifié en sim ; carte en attente) |
| M9.2 | 4 | 0 |
| M9.3 | 5 | 0 |
| M9.4 | 3 | 3 (estimations ; synthèse en attente) |
| M10 | 13 (+ renvoi T10.12) | 0 |
| M12 | 11 | 0 |
| M11 | 8 (+ renvoi T11.8) | 0 |
