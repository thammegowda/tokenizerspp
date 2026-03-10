# tokenizerpp

A pure C++23 rewrite of HuggingFace's [tokenizers](https://github.com/huggingface/tokenizers) library. Designed for fast, self-contained tokenization without Rust dependencies — ideal for edge deployments and C++-native projects.

## Highlights

- **1.9x faster encoding**, **2.3x faster decoding** than Rust ([benchmarks](benchmarks/results-2026-03-10.md))
- **Exact token compatibility** — produces identical output to the Rust implementation
- **2 MB binary** (stripped, vs 4.3 MB for Rust)
- **Zero runtime dependencies** beyond the libc/libC++ standard libraries
- Loads any HuggingFace `tokenizer.json` out of the box

## Quick Start

### Build

```bash
make build        # Release build
make debug        # Debug build
make bench        # Build + run benchmarks against Rust reference
```

Requires CMake 3.20+ and a C++23-capable compiler (GCC 13+, Clang 17+).

### Usage

```cpp
#include <tokenizers/tokenizer.h>

// Load from file
auto result = tokenizers::Tokenizer::from_file("tokenizer.json");
auto& tokenizer = *result;

// Encode
auto encoding = tokenizer.encode("Hello, world!", true);
auto& ids = encoding->get_ids();           // std::vector<uint32_t>
auto& attention = encoding->get_attention_mask();

// Decode
auto text = tokenizer.decode(ids, false);  // "Hello, world!"

// Batch operations
std::vector<std::string> texts = {"Hello", "World"};
auto batch = tokenizer.encode_batch(texts, true);

// Encode a pair (e.g., for QA)
auto pair_enc = tokenizer.encode_pair("question", "context", true);
```

### Chat Templates

```cpp
#include <tokenizers/tokenizer.h>

auto tokenizer = tokenizers::Tokenizer::from_file("tokenizer.json");

std::vector<tokenizers::ChatMessage> messages = {
    {"system", "You are a helpful assistant."},
    {"user", "What is 2+2?"},
};

// Apply chat template → formatted string
auto formatted = tokenizer->apply_chat_template(messages, true);

// Or encode directly → token IDs
auto encoding = tokenizer->encode_chat(messages, true, true);
```

### Serialization

```cpp
// Load from JSON string
auto tok = tokenizers::Tokenizer::from_string(json_blob);

// Save to file
tokenizer.save("output.json", true);  // pretty-printed

// Export to JSON string
auto json = tokenizer.to_string(false);
```

### Error Handling

All fallible operations return `Result<T>` (alias for `std::expected<T, Error>`):

```cpp
auto result = tokenizer.encode("text", true);
if (!result) {
    std::cerr << result.error().message() << "\n";
    return;
}
auto& encoding = *result;
```

## Supported Components

### Models

| Model | Description |
|-------|-------------|
| **BPE** | Byte-Pair Encoding (GPT-2, RoBERTa, etc.) |
| **WordPiece** | BERT-style subword tokenization |
| **Unigram** | SentencePiece Viterbi algorithm (T5, ALBERT) |
| **WordLevel** | Simple dictionary lookup |

### Normalizers

BertNormalizer, NFC, NFD, NFKC, NFKD, NMT, Lowercase, Strip, StripAccents, Replace, Prepend, ByteLevel, Precompiled, Sequence

### Pre-tokenizers

BERT, ByteLevel, Whitespace, WhitespaceSplit, CharDelimiter, Digits, Metaspace, Punctuation, Split, UnicodeScripts, FixedLength, Sequence

### Decoders

WordPiece, BPE, ByteLevel, ByteFallback, CTC, Fuse, Metaspace, Strip, Sequence

### Post-processors

BERT, RoBERTa, Template, Sequence

### Encoding Output

Each `Encoding` contains:
- `ids` — token IDs
- `type_ids` — segment IDs (for sentence pairs)
- `tokens` — string tokens
- `offsets` — byte offsets into original text
- `attention_mask`
- `special_tokens_mask`
- `words` — word indices (alignment tracking)
- `overflowing` — overflow encodings from truncation with stride

## Benchmarks

Dataset: [big.txt](https://norvig.com/big.txt) (6.2 MB) with GPT-2 tokenizer. All binaries stripped, same dynamic linking.

| Variant | Encode (ms) | Enc tok/s | Decode (ms) | Dec tok/s | Binary |
|---------|------------|-----------|-------------|-----------|--------|
| Rust (reference) | 3,871 ± 88 | 413,902 | 460 ± 23 | 3,479,954 | 4.3 MB |
| Python | 5,395 ± 101 | 297,013 | 521 ± 56 | 3,097,710 | N/A |
| C++ bindings (Rust FFI) | 3,678 ± 56 | 435,628 | 330 ± 41 | 4,905,813 | 4.8 MB |
| **tokenizerpp** | **2,074 ± 35** | **772,591** | **199 ± 2** | **8,051,071** | **2.0 MB** |

See [benchmarks/results-2026-03-10.md](benchmarks/results-2026-03-10.md) for full details.

```bash
make bench   # reproduce
```

## What's Not Implemented

These features exist in the Rust tokenizers library but are **not** in this C++ rewrite:

| Feature | Notes |
|---------|-------|
| **Training** | No BPE/WordPiece/Unigram/WordLevel trainers. Can only load pre-trained models. |
| **`from_pretrained()` HTTP loading** | No downloading models by name from HuggingFace Hub. Load from local files only. |
| **Parallel batch processing** | No `rayon`-style multithreading. Batch encode/decode runs sequentially. |
| **Pre-tokenized input** | Cannot pass pre-split text to `encode()`. Only raw strings supported. |
| **Individual model save** | `model.save(folder, prefix)` not implemented. Use tokenizer-level `save()` instead. |

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11.3 | JSON parsing |
| [uni-algo](https://github.com/uni-algo/uni-algo) | 1.2.0 | Unicode normalization |
| [PCRE2](https://github.com/PCRE2Project/pcre2) | 10.44 | Regex (lookahead/lookbehind) |

All fetched automatically via CMake FetchContent.

## Project Structure

```
tokenizerspp/
├── include/tokenizers/   # Public headers (23 files)
├── src/
│   ├── core/             # Tokenizer, encoding, serialization, patterns
│   ├── models/           # BPE, WordPiece, Unigram, WordLevel
│   ├── normalizers/      # All normalizer implementations
│   ├── pre_tokenizers/   # All pre-tokenizer implementations
│   ├── decoders/         # All decoder implementations
│   ├── processors/       # All post-processor implementations
│   ├── utils/            # Regex, padding, truncation
│   └── chat_template/    # Jinja2 template engine
├── tests/                # Google Test suite
├── benchmarks/           # Head-to-head benchmarks vs Rust
├── CMakeLists.txt
└── Makefile
```

## License

Apache 2.0 — same as the original [HuggingFace tokenizers](https://github.com/huggingface/tokenizers).
