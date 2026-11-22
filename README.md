# tokenizerpp

A pure C++23 rewrite of HuggingFace's [tokenizers](https://github.com/huggingface/tokenizers) library. Designed for fast, self-contained tokenization without Rust dependencies — ideal for edge deployments and C++-native projects.

## Highlights

- **1.9x faster encoding**, **2.3x faster decoding** than Rust ([benchmarks](benchmarks/results-2026-03-10.md))
- **Hugging Face-compatible tokenization** — see [tested model coverage](#tested-model-coverage)
- **< 1 MB binary** (`0.94 MiB` stripped, vs `4.3 MB` for Rust)
- **Zero runtime dependencies** beyond the libc/libC++ standard libraries
- Loads supported Hugging Face `tokenizer.json` files

## Quick Start

### Build

```bash
make build        # Release build
make debug        # Debug build with tests
make test         # run tests (after make debug)
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

### Agentic Chat

Use structured messages for tool calls/results, reasoning, and multimodal
content. The checkpoint's template decides which roles and content formats are
valid; the library does not require conversations to end with a user message.

```cpp
auto tokenizer = tokenizers::Tokenizer::from_directory("./checkpoint");
if (!tokenizer) { /* report tokenizer.error().message() */ return; }

using json = nlohmann::ordered_json;
auto messages = json::parse(R"([
  {"role":"user","content":"What is the weather in Paris?"},
  {"role":"assistant","content":null,"tool_calls":[
    {"id":"call_1","type":"function","function":{
      "name":"weather","arguments":{"city":"Paris"}
    }}
  ]},
  {"role":"tool","tool_call_id":"call_1","name":"weather","content":"20 C"}
])");
auto options = json::parse(R"({
  "tools":[{"type":"function","function":{
    "name":"weather","description":"Get weather",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
  }}],
  "reasoning_effort":"low"
})");

auto formatted = tokenizer->apply_chat_template_json(messages, true, "default", options);
auto encoding = tokenizer->encode_chat_json(messages, true, "default", options);
// Check both Results before use. encode_chat_json does not add special tokens
// again: the chat template already supplies them.
```

- Loaders read sibling `tokenizer_config.json` and `chat_template.jinja`; the
  standalone template overrides the embedded default. Invalid existing config
  files return errors rather than silently disabling chat support.
- Parse inputs as `ordered_json` to preserve tool-argument and schema key order.
  Existing `nlohmann::json` values still convert, but their lost insertion order
  cannot be recovered. Chat API signatures now accept `ordered_json`; rebuild
  consumers when updating the library.
- Arguments are not canonicalized automatically. Inkling 2 requires parsed
  argument objects, not JSON strings. Other templates may accept strings.
- Runtime options such as `tools`, `enable_thinking`, and `reasoning_effort`
  pass through to the model's template; the library doesn't force defaults.
- This is prompt formatting/tokenization, not a tool executor or a parser for
  generated tool calls. Generation blocks render their body but do not produce
  assistant-token masks.

The engine supports generation/do blocks, list concatenation and append/extend,
ordered dictionary iteration, `dictsort` keywords, and Hugging Face `tojson`
options (`sort_keys`, `separators`, `ensure_ascii`, integer `indent`). The default
JSON output now preserves insertion order and uses spaces after commas/colons;
request compact separators explicitly when required. Unsupported filters and
calls return errors rather than silently passing through.

#### Tested model coverage

| Models | Coverage |
|---|---|
| [Inkling-2-Mini](tests/chat-templates/inkling2.jinja), [Qwen3.5](tests/chat-templates/qwen3_5.jinja) | Checked-in chat templates, including tool calls |
| [Gemma 3](tests/chat-templates/gemma3.jinja), [Gemma 4](tests/chat-templates/gemma4.jinja), [TranslateGemma](tests/chat-templates/translate-gemma.jinja) | Checked-in chat templates |
| Phi-3, Qwen2 | Inline chat-template tests in [test_chat_template.cpp](tests/test_chat_template.cpp) |
| Llama 2, Mistral | Simplified chat-template patterns in [test_chat_template.cpp](tests/test_chat_template.cpp) |
| GPT-2 | [Benchmark](benchmarks/results-2026-03-10.md) against Rust; compares token counts, not token IDs |

The default test suite uses synthetic tokenizer vocabularies and the templates
above; it does not include full pretrained `tokenizer.json` files or establish
token-ID parity for these models. Template tests do not test model inference or
tool execution.

#### Reference parity

Builds with tests enabled also provide `tokenizerpp_render_chat`. Compare local
templates with Hugging Face's Jinja environment (Python `transformers` and
`jinja2` are validation-only dependencies):

```bash
python tools/check_chat_parity.py \
  --renderer build-debug/tokenizerpp_render_chat \
  --template tests/chat-templates/inkling2.jinja \
  --template tests/chat-templates/qwen3_5.jinja \
  --output /tmp/chat-parity.json
```

Additional one-off template comparisons (not all fixtures are checked in) covered
Inkling-2-Mini revision
`b85c3cfa615fc413cc9af3379f491b963f3b586a` (vendored as
`tests/chat-templates/inkling2.jinja`, Apache-2.0), Qwen3-8B, the Qwen3.5 and Gemma4
fixtures, and the tool-capable Llama 3.1 template from
`unsloth/Meta-Llama-3.1-8B-Instruct`: **120/120 cases matched** with
Transformers 5.23.0, including 10 matching reference-error cases. Checks cover
tool declarations, single/parallel tool round trips, reasoning, null content,
and both generation-prompt settings. This is tested template compatibility,
not full Jinja support or model-quality certification.

### Serialization

```cpp
// Load from JSON string
auto tok = tokenizers::Tokenizer::from_string(json_blob);

// Save to file
tokenizer.save("output.json", true);  // pretty-printed

// Export to JSON string
auto json = tokenizer.to_string(false);
```

### NLCodec Conversion

Convert an NLCodec BPE model into a self-contained `tokenizer.json`:

```bash
python3 tools/nlcodec_to_tokenizer_json.py nlcodec.model tokenizer.json
```

The converter preserves token IDs, greedy segmentation, per-character unknown
tokens, and duplicate token IDs. Output is compact UTF-8 by default; pass
`--pretty` for indented JSON.

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
| [uni-algo](https://github.com/uni-algo/uni-algo) | 1.2.0 | Unicode normalization, properties & scripts |

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
