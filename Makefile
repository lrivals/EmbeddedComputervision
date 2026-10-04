.PHONY: help test test-py test-slow test-cpp lint count-macs bench-conv get-weights anchors csim clean

help:
	@echo "test-py     tests du modèle NumPy (pytest)"
	@echo "test-slow   tests longs : surapprentissage d'une image (T2.8)"
	@echo "test-cpp    build + tests du golden model C++"
	@echo "lint        ruff sur python/ et tools/"
	@echo "count-macs  paramètres et MACs des réseaux Tiny (T0.5)"
	@echo "bench-conv  temps de la conv 13×13×1024→1024 (T1.2)"
	@echo "get-weights poids Darknet pré-entraînés dans weights/ (T1.9)"
	@echo "anchors     ancres k-means sur VOC → results/anchors.md (T2.2)"
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

csim:
	cd hls && vitis_hls -f scripts/csim.tcl

clean:
	rm -rf build
