# Conventions

## Tenseurs

- Format **NCHW** partout : lot, canaux, hauteur, largeur (spec §0).
- Poids de convolution : `(F, C, k, k)`.
- Tête YOLO : sortie `(N, A·(5+C), S, S)`, réorganisée en `(N, A, 5+C, S, S)` ; ordre des
  5 premiers canaux : `t_x, t_y, t_w, t_h, t_o`, puis les classes.
- Python : `float64` pour les gradchecks, `float32` pour l'entraînement ; entier : `int8`
  pour les activations et les poids, `int32` pour les accumulateurs et les biais.

## Boîtes et ancres

- Vérités terrain normalisées dans `[0, 1]` : `(cx, cy, w, h)` relatifs à l'image après
  letterbox.
- Ancres stockées **en pixels pour une entrée de 416** (comme les `.cfg` Darknet) et
  converties en fraction de l'image au moment de l'utilisation.
- Ordre des ancres croissant par aire ; Tiny-YOLOv3 : indices 3-5 → tête 13×13,
  0-2 → tête 26×26.

## Nommage

- Couches numérotées comme dans les tableaux du §3 (`L00` … `L22`), identiques au `.cfg`
  Darknet. Les dumps et le manifest utilisent ces numéros.
- Tâches : `T<jalon>.<n>` ; branche git suggérée : `t1.2-conv`.

## Tolérances de test

| Comparaison | Tolérance |
|---|---|
| gradcheck (différences centrées, h = 1e-6, float64) | erreur relative ≤ 1e-7 |
| implémentation optimisée vs naïve (float64) | ≤ 1e-12 |
| fusion BN vs conv + BN | ≤ 1e-12 |
| entier Python vs golden C++ vs HLS vs carte | **égalité exacte** |
| mAP entière vs flottante | écart à documenter ; cible ≤ 1 point (sinon QAT, T4.6) |

## Code

- Python : fonctions `forward(x) -> (y, cache)` et `backward(dy, cache) -> (dx, grads)`.
  Pas de dépendance hors NumPy dans `python/yolo/` (Pillow toléré dans `data/` pour lire les
  images).
- C++ : C++17, en-têtes dans `cpp/golden/include/golden/` ; le code partagé avec HLS ne
  fait pas d'allocation dynamique.
- Chaque formule implémentée cite sa section de la spec en commentaire (`# §6.2`).
