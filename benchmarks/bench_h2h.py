#!/usr/bin/env python3
"""
Head-to-head benchmark: Rust tokenizers vs tokenizerpp (pure C++23)
Runs each variant multiple times and generates a comparison report.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Optional

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent

# Configuration
NUM_RUNS = 3
INPUT_FILE = SCRIPT_DIR / "big.txt"
TOKENIZER_FILE = SCRIPT_DIR / "gpt2-tokenizer.json"

# Allow overrides
if len(sys.argv) > 1:
    INPUT_FILE = Path(sys.argv[1])
if len(sys.argv) > 2:
    TOKENIZER_FILE = Path(sys.argv[2])
if len(sys.argv) > 3:
    NUM_RUNS = int(sys.argv[3])

VARIANTS = {
    "rust": {
        "command": [str(SCRIPT_DIR / "bench_rust.out"), str(TOKENIZER_FILE), str(INPUT_FILE)],
        "binary": str(SCRIPT_DIR / "bench_rust.out"),
        "name": "Rust (reference)",
        "optional": True,
    },
    "python": {
        "command": ["python3", str(SCRIPT_DIR / "bench_python.py"), str(TOKENIZER_FILE), str(INPUT_FILE)],
        "binary": None,
        "name": "Python",
        "optional": True,
    },
    "cpp-bindings": {
        "command": [str(SCRIPT_DIR / "bench_cpp_bindings.out"), str(TOKENIZER_FILE), str(INPUT_FILE)],
        "binary": str(SCRIPT_DIR / "bench_cpp_bindings.out"),
        "name": "C++ bindings (Rust FFI)",
    },
    "tokenizerpp": {
        "command": [str(SCRIPT_DIR / "bench_cpp.out"),
                    str(TOKENIZER_FILE), str(INPUT_FILE)],
        "binary": str(SCRIPT_DIR / "bench_cpp.out"),
        "name": "tokenizerpp (C++23)",
    },
}


def parse_output(output: str) -> Dict[str, float]:
    result = {}
    for line in output.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def run_benchmark(variant_key: str, config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    env = os.environ.copy()
    if "env" in config:
        env.update(config["env"])
    try:
        result = subprocess.run(
            config["command"], capture_output=True, text=True, check=True, env=env,
            timeout=600  # 10 min timeout
        )
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e.stderr[:200]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>600s)", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  Binary not found: {config['command'][0]}", file=sys.stderr)
        return None


def fmt_num(n: float) -> str:
    return f"{n:,.0f}"


def fmt_size(nbytes: int) -> str:
    """Format byte count as human-readable size (matches ls -lh)."""
    import math
    for unit in ["", "K", "M", "G"]:
        if nbytes < 1024:
            if not unit:
                return f"{nbytes}B"
            val = math.ceil(nbytes * 10) / 10
            return f"{val:.1f}{unit}" if val < 10 else f"{val:.0f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}T"


def get_binary_size(path: Optional[str]) -> Optional[int]:
    """Return file size in bytes, or None if not found."""
    if not path:
        return None
    p = Path(path)
    return p.stat().st_size if p.exists() else None


def main():
    print("=" * 70)
    print("  Head-to-Head Benchmark: tokenizers (Rust) vs tokenizerpp (C++23)")
    print("=" * 70)
    print(f"  Input:     {INPUT_FILE} ({INPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Tokenizer: {TOKENIZER_FILE.name}")
    print(f"  Runs:      {NUM_RUNS}")
    print()

    all_stats = {}
    for variant_key, config in VARIANTS.items():
        name = config["name"]
        # Check if binary/command exists for optional variants
        if config.get("optional"):
            import shutil
            cmd = config["command"][0]
            if not Path(cmd).exists() and not shutil.which(cmd):
                print(f">>> {name}: skipped (not found)")
                print()
                continue
        print(f">>> {name}")

        runs = []
        for i in range(1, NUM_RUNS + 1):
            print(f"    Run {i}/{NUM_RUNS}...", end=" ", flush=True)
            result = run_benchmark(variant_key, config)
            if result is None:
                print("✗")
                break
            runs.append(result)
            encode_ms = result.get("encode_time_ms", 0)
            decode_ms = result.get("decode_time_ms", 0)
            enc_tps = result.get("tokens_per_sec", 0)
            dec_tps = result.get("decode_tokens_per_sec", 0)
            ntok = result.get("num_tokens", 0)
            print(f"\u2713  enc {encode_ms:.0f}ms ({fmt_num(enc_tps)} tok/s)  dec {decode_ms:.0f}ms ({fmt_num(dec_tps)} tok/s)  [{fmt_num(ntok)} tokens]")

        if runs:
            encode_times = [r["encode_time_ms"] for r in runs]
            decode_times = [r["decode_time_ms"] for r in runs]
            tokens_per_sec = [r["tokens_per_sec"] for r in runs]
            decode_tps = [r["decode_tokens_per_sec"] for r in runs]
            load_times = [r["load_time_ms"] for r in runs]

            s = {
                "name": name,
                "load_ms": {"mean": mean(load_times), "stdev": stdev(load_times) if len(load_times) > 1 else 0},
                "encode_ms": {"mean": mean(encode_times), "stdev": stdev(encode_times) if len(encode_times) > 1 else 0},
                "decode_ms": {"mean": mean(decode_times), "stdev": stdev(decode_times) if len(decode_times) > 1 else 0},
                "tokens_per_sec": {"mean": mean(tokens_per_sec), "stdev": stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0},
                "decode_tps": {"mean": mean(decode_tps), "stdev": stdev(decode_tps) if len(decode_tps) > 1 else 0},
                "num_tokens": int(runs[0]["num_tokens"]),
                "num_chars": int(runs[0]["num_chars"]),
            }
            all_stats[variant_key] = s
        print()

    # Print comparison table
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"{'Variant':<25} {'Encode (ms)':<16} {'Enc tok/s':<16} {'Decode (ms)':<16} {'Dec tok/s':<16} {'Binary':<8}")
    print("-" * 97)

    # Find reference variant (prefer rust, fall back to cpp-bindings)
    ref_key = None
    for candidate in ["rust", "cpp-bindings"]:
        if candidate in all_stats:
            ref_key = candidate
            break

    for key, s in all_stats.items():
        enc = s["encode_ms"]
        enc_tps = s["tokens_per_sec"]
        dec = s["decode_ms"]
        dec_tps = s["decode_tps"]
        enc_str = f"{enc['mean']:.0f} \u00b1 {enc['stdev']:.0f}"
        enc_tps_str = f"{fmt_num(enc_tps['mean'])}"
        dec_str = f"{dec['mean']:.0f} \u00b1 {dec['stdev']:.0f}"
        dec_tps_str = f"{fmt_num(dec_tps['mean'])}"
        bin_size = get_binary_size(VARIANTS[key]["binary"])
        size_str = fmt_size(bin_size) if bin_size else "N/A"
        print(f"{s['name']:<25} {enc_str:<16} {enc_tps_str:<16} {dec_str:<16} {dec_tps_str:<16} {size_str:<8}")

    print()

    # Compare tokenizerpp against reference
    if ref_key and "tokenizerpp" in all_stats:
        ref = all_stats[ref_key]
        cpp = all_stats["tokenizerpp"]
        ref_name = ref["name"]

        enc_ratio = cpp["tokens_per_sec"]["mean"] / ref["tokens_per_sec"]["mean"] * 100
        dec_ratio = cpp["decode_tps"]["mean"] / ref["decode_tps"]["mean"] * 100
        print(f"  Encode speed: tokenizerpp = {enc_ratio:.1f}% of {ref_name}")
        print(f"  Decode speed: tokenizerpp = {dec_ratio:.1f}% of {ref_name}")

        # Token count comparison
        ref_tok = ref["num_tokens"]
        cpp_tok = cpp["num_tokens"]
        if ref_tok == cpp_tok:
            print(f"  Token count:  \u2713 MATCH ({fmt_num(ref_tok)} tokens)")
        else:
            print(f"  Token count:  \u2717 MISMATCH ({ref_name}={fmt_num(ref_tok)}, tokenizerpp={fmt_num(cpp_tok)})")

    print()

    # Save results
    out_file = SCRIPT_DIR / "benchmark_h2h_results.json"
    with open(out_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Results saved to {out_file}")


if __name__ == "__main__":
    main()
