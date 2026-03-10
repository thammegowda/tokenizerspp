#!/usr/bin/env python3
"""Benchmark for HuggingFace tokenizers Python bindings.
Output format: key:value pairs for bench_h2h.py compatibility.
Usage: bench_python.py <tokenizer.json> <input.txt>
"""

import sys
import time


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <tokenizer.json> <input.txt>", file=sys.stderr)
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    input_path = sys.argv[2]

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("Error: 'tokenizers' package not installed. Run: pip install tokenizers", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    load_start = time.perf_counter()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    load_ms = (time.perf_counter() - load_start) * 1000

    # Read input
    with open(input_path, "r") as f:
        text = f.read()
    num_chars = len(text)

    # Benchmark encode
    encode_start = time.perf_counter()
    encoding = tokenizer.encode(text, add_special_tokens=False)
    encode_ms = (time.perf_counter() - encode_start) * 1000

    ids = encoding.ids
    num_tokens = len(ids)
    encode_tps = num_tokens / (encode_ms / 1000)

    # Benchmark decode
    decode_start = time.perf_counter()
    _ = tokenizer.decode(ids, skip_special_tokens=False)
    decode_ms = (time.perf_counter() - decode_start) * 1000

    decode_tps = num_tokens / (decode_ms / 1000)

    print(f"load_time_ms:{load_ms:.0f}")
    print(f"encode_time_ms:{encode_ms:.0f}")
    print(f"decode_time_ms:{decode_ms:.0f}")
    print(f"num_tokens:{num_tokens}")
    print(f"num_chars:{num_chars}")
    print(f"tokens_per_sec:{encode_tps:.2f}")
    print(f"decode_tokens_per_sec:{decode_tps:.2f}")


if __name__ == "__main__":
    main()
