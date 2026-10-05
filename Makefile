.PHONY: help test test-py test-slow test-cpp golden-check roofline lint count-macs bench-conv get-weights anchors detect eval-float calibrate eval-int export csim csim-gcc hls-cycles hls-synth hls-cosim hls-export hls-synth-post hls-export-post hls-synth-stream hls-cosim-stream hls-export-stream stream-rom hls-report check-regmap vivado-build fpga-firmware sw-sim sw-board perf-model m8-inputs m8-int bench-sim map-stades bench-report ci-model ci clean

help:
	@echo "test-py     tests du modèle NumPy (pytest)"
	@echo "test-slow   tests longs : surapprentissage d'une image (T2.8)"
	@echo "test-cpp    build + tests du golden model C++"
	@echo "golden-check golden C++ sur les dumps de model/ + rapport par couche (T5.4)"
	@echo "roofline    tuiles et performance par carte hw/boards → results/roofline.md (T5.6)"
	@echo "lint        ruff sur python/ et tools/"
	@echo "count-macs  paramètres et MACs des réseaux Tiny (T0.5)"
	@echo "bench-conv  temps de la conv 13×13×1024→1024 (T1.2)"
	@echo "get-weights poids Darknet pré-entraînés dans weights/ (T1.9)"
	@echo "anchors     ancres k-means sur VOC → results/anchors.md (T2.2)"
	@echo "detect      démo Tiny-YOLOv2 VOC sur une image → build/detect/ (T3.4)"
	@echo "eval-float  mAP VOC2007 test de Tiny-YOLOv2 VOC, poids Darknet (T3.4)"
	@echo "calibrate   échelles INT8 sur 500 images VOC trainval → build/quant/ (T4.2)"
	@echo "eval-int    mAP VOC2007 test du modèle entier et du flottant (T4.5)"
	@echo "export      modèle entier + dumps → model/<net>/ (T4.7)"
	@echo "csim-gcc    C-sim du noyau HLS avec g++ (sans Vitis) : tb_conv + tb_net (T6.1-T6.4)"
	@echo "hls-cycles  estimation de cycles par couche en C-sim → build/hls/cycles_conv.csv"
	@echo "csim        C-simulation HLS dans Vitis (BOARD=$(BOARD))"
	@echo "hls-synth   synthèse Vitis HLS (T6.5)"
	@echo "hls-cosim   co-simulation RTL sur une image (T6.5, long)"
	@echo "hls-export  IP pour Vivado → build/hls/ip/ (T6.5)"
	@echo "hls-report  results/hls_report.md (T6.5)"
	@echo "hls-synth-post / hls-export-post  noyau yolo_post (T9.1, Vitis)"
	@echo "hls-synth-stream / hls-cosim-stream / hls-export-stream  yolo_stream, poids en ROM (T10.8-T10.9, Vitis)"
	@echo "check-regmap offsets de sw/driver/regmap.hpp == xyolo_conv_hw.h généré (T7.2)"
	@echo "vivado-build block design + bitstream + .xsa → build/vivado/$(BOARD)/ (T7.1)"
	@echo "fpga-firmware yolo.bit.bin + yolo.dtbo pour xmutil (T7.1)"
	@echo "sw-sim      driver ARM sur PC (backend sim) : run_compare + yolo_app (T7.2-T7.4)"
	@echo "sw-board    build natif sur la KV260 (backend uio)"
	@echo "perf-model  modèle de cycles == C-sim, pistes d'optimisation chiffrées (T8.1, T8.3)"
	@echo "m8-inputs   entrées int8 de VOC2007 test (stretch) → build/m8/<net>/ (T8.2)"
	@echo "m8-int      mAP flottante + entière, détections entières par image (T8.2)"
	@echo "bench-sim   stade FPGA en C-sim sur PC, paquets parallèles (T8.2, ~40 min sur 22 threads)"
	@echo "map-stades  results/map_stades.md : mAP aux trois stades, égalité entier == FPGA"
	@echo "bench-report results/benchmarks.csv (ce travail) : mesures carte ou projection"
	@echo "ci-model    export synthétique → model/ (sans poids Darknet ni VOC, T10.13)"
	@echo "ci          lint, golden, C-sim, sw, perf-model, pytest ; export obligatoire (T10.13-14)"

test: test-py test-cpp

# CI (T10.13, T10.14) : make ci-model ci. Les tests Python passent en dernier, une fois
# golden_run, tb_stream et cycles_conv.csv construits ; YOLO_REQUIRE_MODEL change en échec
# tout saut faute d'export.
ci-model:
	python tools/make_ci_model.py

ci: export YOLO_REQUIRE_MODEL = 1
ci:
	$(MAKE) lint
	$(MAKE) test-cpp
	$(MAKE) golden-check
	$(MAKE) csim-gcc
	$(MAKE) perf-model
	$(MAKE) sw-sim
	$(MAKE) test-py

test-py:
	cd python && python -m pytest -q

test-slow:
	cd python && python -m pytest -q -m slow

test-cpp:
	cmake -S cpp/golden -B build/golden -DGOLDEN_TESTS=ON
	cmake --build build/golden -j
	cd build/golden && ctest --output-on-failure

DUMP_IMAGES = 000001 000002 000003

golden-check:
	cmake -S cpp/golden -B build/golden
	cmake --build build/golden -j --target golden_run
	for n in $(QNETS); do for i in $(DUMP_IMAGES); do \
	  build/golden/golden_run run model/$$n model/$$n/dumps/$$i/input.npy build/golden/out/$$n/$$i && \
	  python tools/compare_dumps.py model/$$n/dumps/$$i build/golden/out/$$n/$$i || exit 1; \
	done; done

roofline:
	python tools/roofline.py

lint:
	ruff check python tools

count-macs:
	python tools/count_macs.py --net all

bench-conv:
	python tools/bench_conv.py

get-weights:
	bash tools/get_weights.sh

anchors:
	python tools/kmeans_anchors.py

detect:
	python tools/detect.py data/VOCdevkit/VOC2007/JPEGImages/000004.jpg

eval-float:
	python tools/eval_voc.py --net tiny-yolov2-voc --weights weights/yolov2-tiny-voc.weights --resize stretch

QNETS = tiny-yolov2-voc tiny-yolov3-coco

calibrate:
	for n in $(QNETS); do python tools/calibrate.py --net $$n || exit 1; done

eval-int:
	python tools/eval_quant.py --net tiny-yolov2-voc --variants float,int --resize stretch

export:
	for n in $(QNETS); do python tools/export_model.py --net $$n || exit 1; done

BOARD ?= kv260

csim-gcc:
	cmake -S hls -B build/hls
	cmake --build build/hls -j
	cd build/hls && ctest --output-on-failure

hls-cycles:
	cmake -S hls -B build/hls
	cmake --build build/hls -j --target tb_conv
	build/hls/tb_conv --image 000001 --csv build/hls/cycles_conv.csv

csim:
	cd hls && vitis_hls -f scripts/csim.tcl -tclargs $(BOARD)

hls-synth:
	cd hls && vitis_hls -f scripts/synth.tcl -tclargs $(BOARD)

hls-cosim:
	cd hls && vitis_hls -f scripts/cosim.tcl -tclargs $(BOARD)

hls-export:
	cd hls && vitis_hls -f scripts/export.tcl -tclargs $(BOARD)

hls-synth-post:
	cd hls && vitis_hls -f scripts/synth_post.tcl -tclargs $(BOARD)

hls-export-post:
	cd hls && vitis_hls -f scripts/export_post.tcl -tclargs $(BOARD)

stream-rom:
	python tools/gen_stream_rom.py --board $(BOARD) --out build/hls/stream_rom

hls-synth-stream: stream-rom
	cd hls && vitis_hls -f scripts/synth_stream.tcl -tclargs $(BOARD)

hls-cosim-stream: stream-rom
	cd hls && vitis_hls -f scripts/cosim_stream.tcl -tclargs $(BOARD)

hls-export-stream: stream-rom
	cd hls && vitis_hls -f scripts/export_stream.tcl -tclargs $(BOARD)

hls-report:
	python tools/hls_report.py --board $(BOARD)

check-regmap:
	python tools/check_regmap.py --board $(BOARD)

VIVADO_JOBS ?= 8

ENGINE ?= conv

vivado-build: check-regmap
	vivado -mode batch -nojournal -log build/vivado-$(BOARD).log \
	  -source hw/boards/$(BOARD)/build.tcl -tclargs $(VIVADO_JOBS) $(ENGINE)

fpga-firmware:
	ENGINE=$(ENGINE) hw/boards/$(BOARD)/firmware.sh

sw-sim:
	cmake -S sw -B build/sw -DSW_BACKEND=sim
	cmake --build build/sw -j
	cd build/sw && ctest --output-on-failure

sw-board:
	cmake -S sw -B build/sw-board -DSW_BACKEND=uio
	cmake --build build/sw-board -j

M8_NET ?= tiny-yolov2-voc
M8_DIR = build/m8/$(M8_NET)
M8_JOBS ?= $(shell nproc)

perf-model: hls-cycles
	python tools/perf_model.py --check build/hls/cycles_conv.csv
	python tools/perf_model.py

m8-inputs:
	python tools/make_inputs.py --net $(M8_NET)

m8-int:
	python tools/eval_quant.py --net $(M8_NET) --variants float,int --resize stretch \
	  --save-dets $(M8_DIR)/int.jsonl --out $(M8_DIR)/eval_float_int.json

# Noyau C-sim en entiers natifs (-DACC_NO_APINT, même arithmétique qu'ap_int, ≈ 7× plus
# rapide en charge) ; paquets distribués au fil de l'eau, reprise après interruption.
bench-sim:
	CXXFLAGS="-DACC_NO_APINT -march=native" cmake -S sw -B build/sw-fast -DSW_BACKEND=sim
	cmake --build build/sw-fast -j --target yolo_bench
	BENCH=build/sw-fast/yolo_bench tools/bench_sim.sh $(M8_DIR) model/$(M8_NET) $(M8_JOBS)

map-stades:
	python tools/map_stades.py --net $(M8_NET)

bench-report:
	python tools/bench_report.py

clean:
	rm -rf build
