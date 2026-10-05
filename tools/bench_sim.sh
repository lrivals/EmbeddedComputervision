#!/usr/bin/env bash
# Stade « FPGA » de la mAP sur PC (T8.2) : yolo_bench, backend sim (noyau C-sim), sur toutes
# les entrées de make_inputs.py.
#   tools/bench_sim.sh build/m8/tiny-yolov2-voc model/tiny-yolov2-voc [JOBS]
# Les images sont distribuées par paquets de CHUNK (défaut 10) aux JOBS processus au fil de
# l'eau (xargs -P) : les cœurs rapides (P) prennent plus de paquets que les lents (E). Un
# paquet dont toutes les images figurent déjà dans <dir>/sim/dets_*.jsonl est sauté (reprise
# après interruption). Sorties : <dir>/sim/dets_c<début>.jsonl, times_c<début>.csv ;
# tools/map_stades.py fusionne tous les dets_*.jsonl.
# BENCH=build/sw-fast/yolo_bench : noyau compilé avec -DACC_NO_APINT (entiers natifs au lieu
# d'ap_int, même arithmétique, ≈ 2× plus rapide).
set -euo pipefail
dir=$1 model=$2 jobs=${3:-$(nproc)}
bench=${BENCH:-build/sw/yolo_bench}
chunk=${CHUNK:-10}
mkdir -p "$dir/sim"

# Débuts des paquets encore incomplets (lignes JSONL tronquées par une interruption ignorées).
todo=$(python3 - "$dir" "$chunk" <<'EOF'
import json, sys
from pathlib import Path
d, chunk = Path(sys.argv[1]), int(sys.argv[2])
done = set()
for p in (d / "sim").glob("dets_*.jsonl"):
    for line in p.read_text().splitlines():
        try:
            done.add(json.loads(line)["image"])
        except (ValueError, KeyError):
            pass
ids = (d / "ids.txt").read_text().split()
print(" ".join(str(s) for s in range(0, len(ids), chunk)
               if not set(ids[s:s + chunk]) <= done))
EOF
)
n=$(echo "$todo" | wc -w)
echo "$n paquets de $chunk images à calculer, $jobs processus ($bench)"
for s in $todo; do echo "$s"; done | xargs -P "$jobs" -I{} sh -c \
  "'$bench' --model '$model' --inputs '$dir/inputs.bin' --ids '$dir/ids.txt' \
     --start {} --count $chunk --warmup 0 --dets '$dir/sim/dets_c{}.jsonl' \
     --times '$dir/sim/times_c{}.csv' > '$dir/sim/log_c{}.txt' 2>&1 && echo 'paquet {} fait'"
echo "→ $dir/sim/"
