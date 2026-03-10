#!/bin/bash
# Self-contained build script for head-to-head benchmarks:
#   tokenizerpp (pure C++23)  vs  original tokenizers (Rust via C++ bindings)
#
# This script:
#   1. Downloads benchmark data (big.txt, gpt2-tokenizer.json) if missing
#   2. Clones the reference tokenizers repo (tg/cpp branch) into .vendor/ if missing
#   3. Builds the Rust C FFI library + C++ bindings benchmark binary
#   4. Builds tokenizerpp in release mode + its benchmark binary
#
# Usage: ./bench_build.sh          (from the benchmarks/ directory)
#    or: make bench                 (from the tokenizerspp/ root)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
VENDOR_DIR="$SCRIPT_DIR/.vendor"
TOKENIZERS_DIR="$VENDOR_DIR/tokenizers"

# ── 1. Download benchmark data ─────────────────────────────────────────────

echo "=== Checking benchmark data ==="

if [ ! -f "$SCRIPT_DIR/big.txt" ]; then
    echo ">>> Downloading big.txt (6.2 MB)..."
    curl -fSL -o "$SCRIPT_DIR/big.txt" https://norvig.com/big.txt
    echo "    ✓ big.txt downloaded"
else
    echo "    ✓ big.txt already present"
fi

if [ ! -f "$SCRIPT_DIR/gpt2-tokenizer.json" ]; then
    echo ">>> Downloading GPT-2 tokenizer.json from HuggingFace..."
    curl -fSL -o "$SCRIPT_DIR/gpt2-tokenizer.json" \
        https://huggingface.co/gpt2/resolve/main/tokenizer.json
    echo "    ✓ gpt2-tokenizer.json downloaded"
else
    echo "    ✓ gpt2-tokenizer.json already present"
fi
echo

# ── 2. Clone reference tokenizers repo ─────────────────────────────────────

echo "=== Checking reference tokenizers repo ==="

if [ ! -d "$TOKENIZERS_DIR" ]; then
    echo ">>> Cloning thammegowda/tokenizers (tg/cpp branch) into .vendor/..."
    mkdir -p "$VENDOR_DIR"
    git clone --branch tg/cpp --depth 1 \
        https://github.com/thammegowda/tokenizers.git \
        "$TOKENIZERS_DIR"
    echo "    ✓ tokenizers repo cloned"
else
    echo "    ✓ tokenizers repo already present at .vendor/tokenizers"
    echo "    (delete .vendor/tokenizers to force a fresh clone)"
fi
echo

# ── 3. Build the reference tokenizers (Rust C FFI + C++ bindings bench) ────

echo "=== Building reference tokenizers ==="

# Build the Rust C FFI library (shared + static)
echo ">>> Building Rust C FFI library (bindings/c)..."
cd "$TOKENIZERS_DIR/bindings/c"
cargo build --release
echo "    ✓ libtokenizers_c built"
echo

RUST_LIB_DIR="$TOKENIZERS_DIR/bindings/c/target/release"

# Build the Rust benchmark binary
echo ">>> Building Rust benchmark binary..."
cd "$TOKENIZERS_DIR/tokenizers"
cargo build --release --features http --example encode_batch 2>/dev/null || true

TOKENIZERS_LIB=$(find target/release/deps -name "libtokenizers-*.rlib" 2>/dev/null | head -n1)
if [ -n "$TOKENIZERS_LIB" ]; then
    rustc --edition 2018 -L target/release/deps -L target/release \
        --extern tokenizers="$TOKENIZERS_LIB" \
        "$SCRIPT_DIR/bench_rust.rs" \
        -o "$SCRIPT_DIR/bench_rust.out" \
        -C opt-level=3 \
        -C strip=symbols
    echo "    ✓ Rust benchmark binary built (stripped)"
else
    echo "    ⚠ Skipping Rust benchmark binary (could not find rlib)"
fi
echo

# Build the C++ bindings benchmark binary
# Statically links Rust tokenizers lib; dynamically links system libs (matching Rust binary)
echo ">>> Building C++ bindings benchmark binary..."
g++ -std=c++17 -O3 -s \
    -I"$TOKENIZERS_DIR/bindings/cpp/include" \
    "$SCRIPT_DIR/bench_cpp_bindings.cpp" \
    -o "$SCRIPT_DIR/bench_cpp_bindings.out" \
    "$RUST_LIB_DIR/libtokenizers_c.a" \
    -lpthread -ldl -lm
echo "    ✓ C++ bindings benchmark binary built"
echo

# ── 4. Install Python tokenizers ───────────────────────────────────────────

echo "=== Checking Python tokenizers ==="
if python3 -c "import tokenizers" 2>/dev/null; then
    echo "    ✓ tokenizers Python package already installed"
else
    echo ">>> Installing tokenizers Python package..."
    pip install tokenizers --quiet
    echo "    ✓ tokenizers Python package installed"
fi
echo

# ── 5. Build tokenizerpp ───────────────────────────────────────────────────

echo "=== Building tokenizerpp ==="

cd "$ROOT_DIR"
cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -DTOKENIZERPP_BUILD_BENCHMARKS=ON -DTOKENIZERPP_BUILD_TESTS=OFF
cmake --build build-release -j
echo "    ✓ tokenizerpp built (including bench_cpp.out)"
echo

# ── 6. Show binary sizes ───────────────────────────────────────────────

echo "=== Binary sizes ==="
for bin in "$SCRIPT_DIR/bench_rust.out" "$SCRIPT_DIR/bench_cpp_bindings.out" "$SCRIPT_DIR/bench_cpp.out"; do
    if [ -f "$bin" ]; then
        printf "    %-40s %s\n" "$(basename "$bin")" "$(du -h "$bin" | cut -f1)"
    fi
done
echo

echo "=== All builds completed successfully ==="
echo
echo "Run benchmarks with:  python3 $SCRIPT_DIR/bench_h2h.py"
echo "            or with:  make bench   (from $ROOT_DIR)"
