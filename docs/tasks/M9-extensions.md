# M9 — Extensions (optionnel)

Pistes ouvertes une fois M8 terminé. T9.1 à T9.4 sont devenues des jalons à part entière ;
T9.5 reste une piste. Les optimisations d'ingénierie (ports larges, pipeline ARM, CI…)
sont rassemblées dans [M10](M10-ameliorations.md).

| Piste | Jalon | Spec | Dépend de |
|---|---|---|---|
| T9.1 — Post-traitement matériel | [M9.1](M9.1-postproc-materiel.md) | §10.3 (2024-zhang) | T8.1 |
| T9.2 — Quantification REQ-YOLO (ADMM, puissances de 2) | [M9.2](M9.2-req-yolo.md) | §9.2 (2019-ding) | T4.6 → T9.3.3 |
| T9.3 — Quantification 4 bits | [M9.3](M9.3-quant-4bits.md) | §9.2, §9.3 (2024-yan) | T4.6 |
| T9.4 — Architecture streaming | [M9.4](M9.4-streaming.md) | §10.1 (SATAY, 2018-venieris) | T8.3 |

M9.3 passe avant M9.2 : il apporte le fake-quant, le mode QAT du trainer et la largeur de
bits paramétrée dont l'ADMM se sert.

### [ ] T9.5 — Apprentissage en ligne sur FPGA
- **Spec** : §4 (EF-Train), §12 (gaps) · **Dépend de** : T8.3
- Piste de recherche ouverte : affiner la tête YOLO sur la carte (BN en pleine précision,
  §9.1). À croiser avec l'axe continual learning.
