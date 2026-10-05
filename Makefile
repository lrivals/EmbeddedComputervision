.PHONY: help test test-py test-slow test-cpp golden-check roofline lint count-macs bench-conv get-weights anchors detect eval-float calibrate eval-int export csim clean

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
	@echo "csim        C-simulation HLS (T6.1, nécessite Vitis HLS)"

test: test-py test-cpp

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

csim:
	cd hls && vitis_hls -f scripts/csim.tcl

clean:
	rm -rf build
