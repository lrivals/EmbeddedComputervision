# Protocole de mesure (T8.1, T8.2)

Ce protocole fixe ce que recouvre chaque chiffre de [`benchmarks.csv`](benchmarks.csv),
[`mesures.md`](mesures.md) et [`map_stades.md`](map_stades.md), pour que la comparaison avec
la base (§10.4) se fasse à périmètre connu. Les commandes de la carte supposent l'étape 3 de
[`hw/boards/kv260/README.md`](../hw/boards/kv260/README.md) faite (overlay chargé, 0 écart
avec `run_compare`).

## Système mesuré

| | |
|---|---|
| Carte | Kria KV260 (SOM K26, XCK26-SFVC784-2LV-C), Ubuntu Kria 22.04 |
| Accélérateur | IP `yolo_conv` (Vitis HLS), moteur unique couche par couche, Tm = 32, Tn = 24, Tr = Tc = 13 ([ADR 0003](../docs/adr/0003-choix-carte.md)) |
| Horloge PL | `pl_clk0` = 200 MHz (vérifier `timing.rpt` : WNS ≥ 0) |
| Réseau | Tiny-YOLOv2 VOC 416×416, INT8 (poids par canal, activations par couche) ; Tiny-YOLOv3 COCO en second |
| MACs | 3,486 GMAC (v2), 2,782 GMAC (v3), convolutions seulement (`tools/perf_model.py`, `tools/count_macs.py`) |
| Logiciel | `sw/app/yolo_bench`, build natif `make sw-board` (`-O2`, Release), backend `uio`, interruption (pas `--poll`) |
| Lot | 1 image ; images traitées l'une après l'autre, sans recouvrement des étages |

## Étages chronométrés

`std::chrono::steady_clock` sur l'ARM, une valeur par image et par étage
(`yolo_bench --times`) :

| Étage | Contenu | Hors périmètre |
|---|---|---|
| `pre` | `--images` : lecture du fichier, décodage JPEG (stb_image), redimensionnement `stretch` (interpolation Darknet), quantification int8 | caméra, acquisition |
| `load` | mise à zéro de l'arène et copie de l'entrée dans le u-dma-buf (non caché, `O_SYNC`) | chargement des poids (une fois, au démarrage) |
| `acc` | pour chaque conv : écriture des registres AXI-Lite, `ap_start`, attente de l'interruption ; somme sur les 9 (v2) ou 13 (v3) convs | — |
| `post` | têtes lues en place dans l'arène, décodage par tables, seuil, NMS par classe | dessin des boîtes |

Avec `--hw-post`, le CSV a cinq colonnes de plus pour `yolo_post` : `overflow` (candidates
perdues par la sélection) et, sur le backend sim seulement, `post_cycles`, `nms_cycles`,
`candidates` et `survivors` (estimation de la C-sim, T11.6).

- **Latence** = `pre + load + acc + post` ; moyenne et **p99** (rang le plus proche :
  990e valeur triée sur 1 000) par étage et de bout en bout.
- **FPS** = 1 000 / latence moyenne (débit séquentiel ; un recouvrement `pre` / `acc` sur deux
  cœurs ARM est une piste de T8.3, pas une mesure).
- **GOPS** = 2 × MACs / `acc` moyen (accélérateur seul, comme 2024-zhang).
- **Efficacité** = cycles théoriques / cycles mesurés = (MACs / (Tm·Tn)) / (`acc` × 200 MHz).
  Par couche : `run_compare --csv` (temps par conv) comparé au modèle de cycles.

## Conditions

- **Images** : les 1 000 premières de VOC2007 test (ordre de `ImageSets/Main/test.txt`),
  JPEG copiés sur la carte (`--images`) ; **5 passes de chauffe** sur la première image,
  non comptées (`--warmup 5`).
- CPU : gouverneur `performance` (`sudo cpupower frequency-set -g performance`), processus
  épinglé sur un cœur (`taskset -c 3`), aucune autre charge (pas de bureau graphique).
- Seuil de détection : 0,005 (celui de la mAP, `--conf 0.005`) : le post-traitement est
  mesuré dans le cas le plus chargé. La démo (0,25) est plus rapide.
- Fichiers de sortie sur la carte en `tmpfs` ou carte SD ; l'écriture JSONL est hors étages.

## Puissance : deux périmètres

| Périmètre | Source | Ce qui est inclus |
|---|---|---|
| **Puce, estimée** | `build/vivado/kv260/power.rpt` (`report_power` après implémentation, activité par défaut) | PL + PS du XCK26 ; on reporte aussi la part PS8 |
| **SOM, mesuré** | INA260 du SOM K26 lu par hwmon (`yolo_bench --power`), comme `platformstats` | rail 5 V du SOM : XCK26, DDR4, eMMC, régulateurs ; **pas** la carte porteuse (ventilateur, USB, Ethernet, carte SD) |

```bash
grep . /sys/class/hwmon/hwmon*/name            # repérer l'INA260 du SOM (ina260_u14)
P=/sys/class/hwmon/hwmonN/power1_input           # µW
```

`yolo_bench --power $P --idle-s 10` mesure 10 s **au repos** (overlay chargé, accélérateur
inactif) puis pendant toute la boucle ; on reporte la puissance en charge, le repos et la
différence. Une mesure à la prise (wattmètre sur l'alimentation 12 V) peut compléter :
périmètre « carte entière », à indiquer comme tel.

Comparaison avec la base (§10.4) : 2024-zhang distingue puce estimée (7,8 W) et carte
mesurée (12,3 W) ; les autres travaux sont *unspec.* : ne comparer que des périmètres
identiques.

## Ressources

`build/vivado/kv260/utilization.rpt` (après implémentation) : `CLB LUTs`, `DSPs`,
`Block RAM Tile` (BRAM36) ; design complet (IP + interconnexions + reset), pas l'IP seule.

## Commandes sur la carte

```bash
make sw-board
N=$(ls /sys/class/uio | head -1)                 # celui dont name == yolo_conv
head -n 1000 data/VOCdevkit/VOC2007/ImageSets/Main/test.txt \
  | sed 's|^|data/VOCdevkit/VOC2007/JPEGImages/|; s|$|.jpg|' > /tmp/voc1000.txt
mkdir -p build/m8/tiny-yolov2-voc/board

# T8.1 : latence, puissance (1 000 images JPEG, prétraitement ARM)
sudo taskset -c 3 build/sw-board/yolo_bench --uio /dev/$N --model model/tiny-yolov2-voc \
  --images /tmp/voc1000.txt --warmup 5 --power $P --idle-s 10 \
  --times build/m8/tiny-yolov2-voc/board/times_0.csv | tee build/m8/tiny-yolov2-voc/board/bench.txt
# temps par couche (3 images de référence) et contrôle 0 écart
sudo build/sw-board/run_compare --uio /dev/$N --net tiny-yolov2-voc \
  --csv build/m8/tiny-yolov2-voc/board/layers.csv

# T8.2 : stade FPGA de la mAP (entrées int8 de make m8-inputs, 2,6 Go, copiées sur la carte)
sudo build/sw-board/yolo_bench --uio /dev/$N --model model/tiny-yolov2-voc \
  --inputs build/m8/tiny-yolov2-voc/inputs.bin --ids build/m8/tiny-yolov2-voc/ids.txt \
  --dets build/m8/tiny-yolov2-voc/board/dets_0.jsonl
```

Puis, sur le PC (fichiers de `board/` rapatriés) :

```bash
make map-stades
python tools/bench_report.py --net tiny-yolov2-voc \
  --layer-csv build/m8/tiny-yolov2-voc/board/layers.csv --vivado build/vivado/kv260 \
  --board-power <W en charge> --idle-power <W au repos>
```

## Sans la carte (état actuel)

- Stade FPGA de la mAP : même `yolo_bench`, backend `sim` (registres émulés devant le noyau
  C-sim), `make bench-sim` ; valide le driver, le post-traitement et le bit-exact sur les
  4 952 images, pas le temps.
- Performance : projection du modèle de cycles (`tools/perf_model.py`, égal cycle pour cycle
  aux compteurs C-sim), marquée « projection C-sim » dans `benchmarks.csv`. C'est une borne
  basse du temps accélérateur : II = 1, profondeurs de pipeline, latence DDR et pilotage ARM
  ignorés.
