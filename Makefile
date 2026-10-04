.PHONY: help test test-py test-cpp lint count-macs csim clean

help:
	@echo "test-py     tests du modèle NumPy (pytest)"
	@echo "test-cpp    build + tests du golden model C++"
	@echo "lint        ruff sur python/ et tools/"
	@echo "count-macs  paramètres et MACs des réseaux Tiny (T0.5)"
	@echo "csim        C-simulation HLS (T6.1, nécessite Vitis HLS)"

test: test-py test-cpp

test-py:
	cd python && python -m pytest -q

test-cpp:
	cmake -S cpp/golden -B build/golden -DGOLDEN_TESTS=ON
	cmake --build build/golden -j
	cd build/golden && ctest --output-on-failure

lint:
	ruff check python tools

count-macs:
	python tools/count_macs.py

csim:
	cd hls && vitis_hls -f scripts/csim.tcl

clean:
	rm -rf build
