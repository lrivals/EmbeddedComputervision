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
- Conversion en fraction : **toujours /416** (`yolo.data.targets.ANCHOR_REF`), y compris en
  multi-échelle : une ancre couvre la même part de l'image à toutes les tailles d'entrée.
- Perte : somme sur le lot dans `yolo_loss`, divisée par N dans le trainer ; seuil *ignore*
  0,5 par défaut (§2.3 ; les `.cfg` Darknet utilisent 0,7).

## Inférence et évaluation

- Seuils : démo `conf = 0,25`, mAP `conf = 0,005` (toute la courbe P/R) ; NMS par classe
  `iou = 0,45`, suppression si IoU > seuil (§8.2).
- mAP : AP 11 points VOC2007, pixels VOC (largeur `x2 − x1 + 1`), apparié si IoU ≥ 0,5
  (`VOCevaldet.m`), `difficult` ignorés (§8.3).
- Référence flottante (T3.4) : Tiny-YOLOv2 VOC, poids Darknet, **redimensionnement direct**
  416×416 (`--resize stretch`, sans letterbox) : mAP 56,30. M4 et M8 se comparent à elle
  avec le même prétraitement.

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
  - Upsample : `out` réserve la place d'une copie ×2 (début du tampon de la route 20). Le
    golden C++ (M5) ne fait **aucune copie** : la conv suivante lit la source par division
    d'adresse `(r >> 1, c >> 1)` ; son entrée est une vue à segments (`golden::InView`) et la
    route 20 se lit en deux segments, L18 vue ×2 puis le prépool de L08 à `R + 128·26·26`.
    `R:[0, 128·26·26)` n'est jamais écrit.
  - `yolo` : `mask` = indices dans `anchors` ; lit la tête à `in`.
- `region` (YOLOv2) : `num` ancres, classes par softmax ; les ancres v2 ne sont pas entières
  (34,56 px).
- `luts.bin` (export réel) : pour chaque tête, à `lut_offset`, trois tables uint32 de 256
  entrées : σ(q·s) en Q16 et e^{q·s} en Q`exp_frac` (≤ 16, pour tenir sur 32 bits)
  indexées par `q + 128`, puis e^{d·s} en Q16 indexée par `d + 255`,
  d = q_c − q_max ∈ [−255, 0] (softmax v2), avec s = `scale` de la tête.
- Export réel : `model/<net>/` (`tools/export_model.py`), dumps `dumps/<id>/input.npy`,
  `Lxx.npy` (sortie int8 de chaque couche) et `detections.json` pour 3 images de VOC2007 test.
- **Allocation** (`yolo.io.export.layout`, commune à l'exemple et à l'export) : ping-pong
  A/B ; `S` garde la couche 13 (lue par 14 et par la route 17) ; `R` reçoit 19 puis 8 ;
  `H13`/`H26` reçoivent les têtes. Aucune écriture ne
  recouvre un tenseur encore à lire (vérifié par le test).

## Arithmétique entière (contrat Python ↔ C++ ↔ HLS)

Référence : `python/yolo/quant/int_layers.py` ; tout entier signé, décalages **arithmétiques**.

- Entrée : `q = ⌊x·127 + ½⌋` (x ∈ [0, 1], échelle 1/127), fait par l'hôte.
- Accumulateur : `acc = Σ q_w·q_x + q_b` sur 32 bits signés (int8 × int8, biais int32).
- Requantification : `y = (acc·M0 + 2ⁿ⁻¹) >> n`, produit sur **64 bits**, `M0 < 2³¹` par
  canal, `n ≤ 31` par couche (`shift`). Arrondi au plus proche, demis vers +∞
  (−2,5 → −2, −0,5 → 0).
- Leaky : `y > 0 ? y : (13·y + 64) >> 7` (pente 13/128, **arrondie** de la même façon ;
  le plancher `(13·y) >> 7` du §9.3 biaise les négatifs de −½ pas par couche).
- Saturation : `clip(y, −qmax, qmax)` après la leaky, `qmax = 127` sauf champ `qmax` de la
  conv dans le manifest (activations à b bits : 2^{b−1} − 1, T9.3) ; −128 n'est jamais
  produit.
- Maxpool, upsample, route : exacts (max et copies) ; maxpool stride 1 par réplication.
- Têtes : `M0`/`shift` comme les autres convs, sans leaky.

## Tolérances de test

| Comparaison | Tolérance |
|---|---|
| gradcheck (différences centrées, h = 1e-6, float64) | erreur relative ≤ 1e-7 (norme max du tenseur, `yolo.testing.gradcheck`) |
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
- HLS (`hls/kernels/`) : le noyau compile en C++14 (défaut Vitis), sans allocation ni
  bibliothèque non synthétisable ; il réutilise l'arithmétique de `golden/conv.hpp`
  (`requantize`, `leaky_int`, `clip8`). Tuiles en macros `ACC_TM`, `ACC_TN`, `ACC_TR`,
  `ACC_TC` (défauts KV260). Boucles nommées (`mac:`, `load_in:`…) pour lire leur II dans
  les rapports. Le top est à portée globale (`set_top yolo_conv`). Testbenchs et driver :
  C++17 ; code de sortie 0 = aucun écart.
- Logiciel ARM (`sw/`) : C++17, bibliothèque standard + POSIX ; tout accès matériel passe
  par `driver::Device` (backends `sim` et `uio`), jamais par des pointeurs bruts ailleurs.
  Les offsets de registres de `sw/driver/regmap.hpp` doivent égaler ceux de l'en-tête
  généré `xyolo_conv_hw.h` (`make check-regmap`) : en cas d'écart, l'en-tête fait foi.
- Mesures (M8) : les périmètres de [results/protocole.md](../results/protocole.md) font foi.
  FPS = 1 000 / latence moyenne de bout en bout (séquentiel) ; GOPS = 2 × MACs (convs) /
  temps accélérateur ; p99 au rang le plus proche ; puissance toujours avec son périmètre
  (puce estimée Vivado, SOM mesuré, carte entière). Une valeur non mesurée est marquée
  « projection » ou laissée vide, jamais estimée sans le dire.
- Chaque formule implémentée cite sa section de la spec en commentaire (`# §6.2`).
