.PHONY: build debug bench bench-build bench-run clean

build:
	cmake -B build-release -S . -GNinja -DCMAKE_BUILD_TYPE=Release
	cmake --build build-release -j

debug:
	cmake -B build-debug -S . -GNinja -DCMAKE_BUILD_TYPE=Debug
	cmake --build build-debug -j

bench: bench-build bench-run

bench-build:
	bash benchmarks/bench_build.sh

bench-run:
	python3 benchmarks/bench_h2h.py

clean:
	rm -rf build-release build-debug
