# ADR 0003 — Choix de la carte FPGA

- Statut : **accepté** (2026-10-05) — installation de Vivado/Vitis encore à vérifier

## Contexte

L'architecture « moteur unique couche par couche » (§10.1) convient à un SoC avec ARM
(Zynq-7000, Zynq UltraScale+/Kria). Le choix fixe les ressources (DSP, BRAM, bande passante
DDR) et donc les tuiles Tm, Tn, Tr, Tc.

## Critères

| Critère | Mesure |
|---|---|
| Faisabilité | tuiles trouvées par `tools/roofline.py` (T5.6) tenant dans la BRAM |
| Performance estimée | ms/image au point roofline retenu |
| Processeur | ARM dur disponible pour le driver et le post-traitement |
| Outils | version de Vivado/Vitis supportée, licence gratuite |
| Coût et disponibilité | prix, livraison |
| Comparabilité | cartes utilisées par la base (Zynq 7035, Zybo Z7-20, KV260) |

## Candidats

Kria KV260, Zynq-7000 (Zybo Z7-20, PYNQ-Z2), Ultra96-V2.

| Critère | KV260 | Ultra96-V2 | Zybo Z7-20 / PYNQ-Z2 |
|---|---|---|---|
| Faisabilité (roofline, ≤ 80 %) | Tm, Tn, Tr, Tc = 32, 24, 13, 13 ; 896 DSP, 112 BRAM18 | 8, 32, 13, 13 ; 288 DSP, 80 BRAM18 | 8, 16, 13, 13 ; 160 DSP, 48 BRAM18 |
| Tiny-YOLOv2 / v3 (ms, roofline) | **31,9 / 27,0** | 86,2 / 71,7 | 198,4 / 159,4 |
| Processeur | 4 × Cortex-A53 | 4 × Cortex-A53 | 2 × Cortex-A9 |
| Outils | Vitis/Vivado, licence gratuite (xck26 dans l'édition Standard) | idem | idem |
| Coût | ~250 € | ~250-300 € | ~300 € / ~150 € |
| Comparabilité | 2026-fata (DPU Vitis AI, pas à la main) | — | 2025-kim (Zybo Z7-20, INT16) |

Source des chiffres : `results/roofline.md` (`make roofline`), ressources de
`hw/boards/*.yaml`.

## Décision

**Kria KV260** (`xck26-sfvc784-2LV-c`, 200 MHz), tuiles **Tm = 32, Tn = 24, Tr = 13,
Tc = 13** : `hls/configs/kv260.tcl`.

- C'est la seule carte dont le point roofline laisse une marge réelle : 6× plus rapide que
  les Zynq-7020 et 2,7× plus que l'Ultra96-V2, pour un prix équivalent.
- La BRAM n'est pas le facteur limitant (112 BRAM18 sur 288, URAM inutilisées) : on peut
  agrandir Tr, Tc ou passer au line buffer sans changer de carte.
- L'ARM A53 suffit pour le driver et le post-traitement de M7 (décodage + NMS sur l'hôte,
  §10.3).
- Comparabilité : 2026-fata mesure YOLOv3-tiny sur la même carte avec un DPU préconfiguré,
  ce qui donne un point de comparaison direct « accélérateur à la main vs DPU ».

## Conséquences

- Les tuiles sont des paramètres de compilation (`-DACC_TM=… -DACC_TN=…`) : changer de carte
  revient à écrire `hls/configs/<carte>.tcl`. Le testbench est aussi passé avec la config
  Zynq-7020 (8, 16, 13, 13) pour vérifier les bords de tuile partiels.
- Vitis HLS n'est pas encore installé sur la machine de développement : la C-sim tourne
  avec g++ et les en-têtes `ap_int` open source (`hls/CMakeLists.txt`). La synthèse, la
  co-simulation et l'export (T6.5) attendent l'installation de Vitis 2023.2 ou ultérieur
  (version validée sur KV260 par AMD).
- Pile logicielle de la carte (M7) : Ubuntu Kria 22.04, `xmutil` pour le bitstream et
  l'overlay, UIO pour les registres et l'interruption, u-dma-buf pour la mémoire contiguë.
  Préférée à XRT (flux plateforme Vitis plus lourd) et à PYNQ (driver en Python) : le driver
  reste en C++ pur et le même code tourne sur PC avec un backend simulé.
