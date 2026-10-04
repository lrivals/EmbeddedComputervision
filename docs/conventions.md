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

- Couches numérotées comme dans les tableaux du §3 (`L00` … `L23`), identiques au `.cfg`
  Darknet. Les dumps et le manifest utilisent ces numéros.
- Tâches : `T<jalon>.<n>` ; branche git suggérée : `t1.2-conv`.

## Format d'échange (`model/<nom>/`)

Schéma : [manifest.schema.json](manifest.schema.json) ; exemple complet :
`model/example_manifest.json`, généré par `tools/make_example_manifest.py` (valeurs fictives).
Validation : `pytest python/tests/test_manifest.py`.

- **Blobs** little-endian, chaque tableau commence sur une frontière de **64 octets** ;
  `w_offset`, `b_offset`, `m0_offset` sont en **octets** depuis le début du fichier.
  - `weights.bin` : int8, ordre `(F, C, k, k)` contigu, poids après fusion BN (§9.1).
  - `bias.bin` : int32, un par canal de sortie, $q_b = \text{round}(b'/(s_x s_w))$ (§9.3).
  - `requant.bin` : int32, $M_0$ par canal de sortie ; `shift` = $n$ de la couche (§9.3).
- **Tampons** (`buffers`) : zones DDR nommées, taille en octets. Une activation est un tenseur
  `(C, H, W)` int8 contigu à `{buf, offset}` (offset en octets) ; le lot vaut 1.
  Concaténer sur les canaux revient donc à écrire les sources bout à bout.
- **Échelles** : symétriques, `valeur réelle = q × scale`. Une conv a `in_scale` et
  `out_scale` ; maxpool, upsample et route conservent l'échelle de leur entrée.
- **Couches** : les 24 couches du §3.2 sont toutes listées, avec leur numéro (`id`).
  - Maxpool fusionné : la conv porte `fused_pool = {layer, k, s}` et écrit directement la carte
    poolée (son `out_shape` est celui après pooling) ; le maxpool reste listé avec
    `fused_into`. Stride 1 : complétion par réplication à droite et en bas (§4.4).
  - `prepool_out` : seconde sortie d'une conv à pool fusionné, carte **avant** pooling, pour une
    route (couche 8 → route 20).
  - Route `layout: "contiguous"` : les sources sont déjà bout à bout dans le même tampon, dans
    l'ordre de `from` ; `out` pointe sur la première, la route ne copie rien (§10.3). Les
    sources doivent partager la **même échelle** (contrainte pour la quantification, T4).
  - Upsample : copie ×2 vers `out` (ici le début du tampon de la route 20).
  - `yolo` : `mask` = indices dans `anchors` ; lit la tête à `in`.
- **Allocation de l'exemple** : ping-pong A/B ; `S` garde la couche 13 (lue par 14 et par
  la route 17) ; `R` reçoit 19 puis 8 ; `H13`/`H26` reçoivent les têtes. Aucune écriture ne
  recouvre un tenseur encore à lire (vérifié par le test).

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
