# Post-traitement matériel — Tiny-YOLOv2 VOC (M9.1)

Décodage et NMS en entiers seulement, dans un noyau HLS `yolo_post` séparé du moteur de
convolution, comme 2024-zhang [2024-zhang#013.1, #014.0, #014.2].

- Référence Python : `python/yolo/infer/hw_postproc.py`.
- Miroir C++ : `cpp/golden/include/golden/hw_postproc.hpp`.
- Noyau : `hls/kernels/postproc.cpp`.
- Intégration : `sw/driver`, avec l'option `--hw-post` de `yolo_app` et `yolo_bench`.

## Arithmétique

| Étape | Matériel (`yolo_post`) | Post-traitement ARM (M7) |
|---|---|---|
| seuil d'objectness | entier t_o ≥ ⌊logit(θ)/s⌋ + 1 (§9.4) | idem |
| σ, e^t, softmax | tables Q16 de `luts.bin` ; softmax en 3 étages, **une réciproque ⌊2³²/Σe⌋ par cellule** | tables, puis division en double |
| boîtes | coins en pixels **Q4** : décalages (pas 416/S = 2^k), ancres Q8 | (cx, cy, w, h) en double |
| scores | Q16, seuil score > ⌊θ·2¹⁶⌋ | double |
| IoU > 0,45 | **29·inter > 9·(a₁ + a₂)**, entiers 64 bits | division en double |
| NMS | **sans tri**, 256 emplacements, ordre du flux | tri stable puis glouton |

La NMS sans tri fonctionne ainsi :

- une candidate recouverte par une sélectionnée de score supérieur ou égal est rejetée ;
- sinon, elle prend la place de la première sélectionnée qu'elle recouvre et invalide les
  autres.

Elle ne donne pas toujours le résultat de la NMS triée. Exemple : dans une chaîne a < b < c
où seuls a∩b et b∩c dépassent le seuil, elle ne garde que c, alors que la NMS triée garde
c et a. Ce cas est testé dans `test_hw_postproc.py`.

## Vérification

- **Python == C++ à l'octet**, via `golden_run hwpp` : sur les dumps des deux réseaux, aux
  seuils 0,25 et 0,005, et sur 6 jeux de têtes aléatoires
  (`python/tests/test_hw_postproc.py`).
- **Noyau == golden** en C-sim g++ (`tb_post`, `make csim-gcc`) : 1 014 cas, dont les dumps
  aux seuils 0,25 et 0,005, le pire cas où toutes les cellules passent, et 1 000 jeux de
  têtes aléatoires (800 pour v2, 200 pour v3), avec débordement compris.
- **Driver** (backend sim, `run_compare --hw-post`, ctest `run_compare_hw_post`) : registres
  `yolo_post` émulés, tables écrites en DDR. Sur les têtes produites par l'accélérateur, les
  boîtes sont égales à celles du golden, pour les deux réseaux.
- Boîtes identiques au post-traitement flottant sur les dumps, à 2/416 près (< 1 pixel Q4)
  pour les coordonnées et à 2⁻¹⁵ près pour les scores.

## mAP (VOC2007 test, 4 952 images, stretch, seuil 0,005)

| Post-traitement | mAP | Écart |
|---|---|---|
| modèle entier, décodage LUT + NMS triée sur l'hôte (T4.5) | 55,66 | — |
| **matériel : tout entier, NMS sans tri, 256 emplacements** | **55,56** | −0,10 |
| matériel, emplacements illimités | 55,56 | −0,10 |

La sélection de 256 emplacements ne déborde que 50 fois sur les 4 952 images au seuil 0,005
de la mAP, et sans effet sur la mAP. L'écart de −0,10 point vient donc de la NMS sans tri et
des coordonnées Q4.

Commande : `tools/eval_quant.py --variants int,int-hwpp --resize stretch [--hw-cap N]`.

## Cycles (estimation C-sim, II = 1, latences ignorées)

| Cas (Tiny-YOLOv2, 845 cellules) | Candidates | Cycles | à 200 MHz |
|---|---|---|---|
| image 000001, seuil 0,25 | 4 | 1 449 | 7,2 µs |
| image 000002, seuil 0,25 | 3 | 1 364 | 6,8 µs |
| image 000002, seuil 0,005 (mAP) | 67 | 9 981 | 50 µs |
| pire cas : 845 cellules au-dessus du seuil, scores aléatoires | 1 181 | 342 375 | 1,7 ms |

Lecture des cycles :

- **Balayage** : 845 cycles, un t_o par cycle.
- **Décodage** : 4 + C + 2C cycles par cellule survivante (lectures, max, somme et scores
  du softmax).
- **NMS** : une comparaison par emplacement occupé et par candidate.

Comparaison à 2024-zhang, qui traite 845 candidates en 0,38 ms, soit 76 000 cycles à
200 MHz [2024-zhang#020.3] :

- Sur une image réelle, le noyau est 50 fois plus rapide, car le seuil sur l'entier t_o
  élimine presque toutes les cellules avant les tables.
- Dans le pire cas (256 sélectionnées), la NMS coûte 4,5 fois plus. Il faudrait alors
  comparer P emplacements par cycle. Avec P = 8, environ 34 000 cycles de NMS, au prix de
  8 tests d'IoU en parallèle.

Sur le PC, le post-traitement ARM de M7 prend entre 0,011 et 0,037 ms par image (x86, seuil
0,25). Il reste à mesurer sur l'A53 de la KV260.

## Reste

- Synthèse de `yolo_post` : LUT/DSP/BRAM, timing à 200 MHz.
- Vérification des offsets de `sw/driver/post_regmap.hpp` contre l'en-tête Vitis.
- Ajout de l'IP au block design (`hw/boards/kv260/build.tcl`, second nœud UIO).
- Mesure sur carte de `yolo_bench --hw-post` face au post-traitement ARM.
