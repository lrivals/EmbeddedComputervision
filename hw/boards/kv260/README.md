# KV260 — du bitstream à la comparaison sur carte (M7)

Pile : Ubuntu Kria 22.04 sur la KV260, overlay chargé par `xmutil`, registres du noyau par
`/dev/uioN` (`generic-uio`), mémoire contiguë par `/dev/udmabuf0`
([u-dma-buf](https://github.com/ikwzm/udmabuf)).

## 1. Sur le PC de développement (Vitis/Vivado 2023.2 ou ultérieur)

```bash
make hls-export BOARD=kv260    # IP → build/hls/ip/yolo_conv_kv260.zip (T6.5)
make check-regmap              # offsets de sw/driver/regmap.hpp == xyolo_conv_hw.h généré
make vivado-build              # hw/boards/kv260/build.tcl → build/vivado/kv260/{yolo.bit, yolo.xsa}
make fpga-firmware             # bootgen + dtc → build/vivado/kv260/firmware/yolo/
```

`vivado-build` échoue si le timing n'est pas tenu à 200 MHz (WNS, TNS ou WHS négatif) ;
`timing.rpt`, `utilization.rpt` et `power.rpt` sont dans `build/vivado/kv260/` (M8).
Les fichiers de carte Kria viennent du board store (`xhub`) ; le script indique la commande
s'ils manquent.

Si `check-regmap` signale un écart, corriger les offsets de `sw/driver/regmap.hpp`
(l'en-tête généré fait foi) puis relancer `make sw-sim`.

## 2. Sur la KV260 (une fois)

```bash
# Module u-dma-buf (CMA) et pilote UIO pour les nœuds « generic-uio »
sudo apt install build-essential cmake git device-tree-compiler
git clone https://github.com/ikwzm/udmabuf && cd udmabuf && make && sudo insmod u-dma-buf.ko
sudo modprobe uio_pdrv_genirq of_id=generic-uio

# Firmware
sudo mkdir -p /lib/firmware/xilinx/yolo
sudo cp yolo.bit.bin yolo.dtbo shell.json /lib/firmware/xilinx/yolo/
```

## 3. Chargement et exécution

```bash
sudo xmutil unloadapp
sudo xmutil loadapp yolo
ls /dev/uio* /dev/udmabuf0          # le nœud yolo_conv : cat /sys/class/uio/uio*/name

make sw-board                        # build natif, backend uio (sans hls/)
sudo build/sw-board/run_compare --uio /dev/uioN --layer 0     # T7.2 : une couche isolée
sudo build/sw-board/run_compare --uio /dev/uioN --csv times.csv   # T7.4 : 0 écart attendu
sudo build/sw-board/yolo_app --uio /dev/uioN --model model/tiny-yolov2-voc \
     --image 000004.jpg --draw boites.jpg --repeat 100
```

`--poll` remplace l'interruption par un sondage d'`ap_done` (utile pour isoler un problème
d'IRQ). `model/` (export T4.7 et dumps) doit être copié sur la carte.

## Points à vérifier au premier essai

- Numéro de l'UIO : plusieurs nœuds `generic-uio` peuvent exister ; prendre celui dont
  `/sys/class/uio/uioN/name` vaut `yolo_conv`.
- Interruption : `cat /proc/interrupts | grep yolo` doit s'incrémenter à chaque couche.
- Cohérence : le u-dma-buf est ouvert en `O_SYNC` (non caché) ; les ports HP ne sont pas
  cohérents. Si les temps ARM (comparaison, post-traitement) gênent les mesures de M8,
  passer à un tampon caché + `sync_for_cpu/device` (sysfs de u-dma-buf).
- Taille du u-dma-buf (32 Mo, `pl.dtsi`) : le driver vérifie qu'elle couvre arène + poids
  + paramètres ; la CMA du noyau Kria doit être au moins aussi grande.
