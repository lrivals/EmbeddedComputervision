# ADR 0001 — Modèle de référence en Python + NumPy seul

- Statut : **accepté** (2026-10-04)

## Contexte

La spec vise une réécriture « sans bibliothèque d'apprentissage » : chaque passe avant et
arrière doit être écrite à la main et vérifiée (§4, §11).

## Décision

`python/yolo/` dépend uniquement de NumPy. Les gradients sont validés par différences finies
(T1.1), pas par comparaison avec PyTorch.

## Conséquences

- Contrôle total des formules, réutilisable telles quelles pour le modèle entier.
- Entraînement lent : la convolution passe par im2col (T1.2). La mAP de référence peut être
  obtenue avec des poids Darknet pré-entraînés (T1.9), sans entraînement complet.
