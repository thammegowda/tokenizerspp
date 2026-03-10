#pragma once
/// @file tokenizers/decoders.h
/// Concrete decoder implementations.

#include "tokenizers/decoder.h"
#include <string>
#include <vector>

namespace tokenizers {
namespace decoders {

/// WordPiece decoder: removes ## prefix and joins.
class WordPieceDecoder : public Decoder {
public:
    std::string prefix = "##";
    bool cleanup = true;

    WordPieceDecoder() = default;
    WordPieceDecoder(std::string prefix, bool cleanup)
        : prefix(std::move(prefix)), cleanup(cleanup) {}

    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// Sequence decoder: chains multiple decoders.
class SequenceDecoder : public Decoder {
public:
    std::vector<DecoderPtr> decoders;

    explicit SequenceDecoder(std::vector<DecoderPtr> d) : decoders(std::move(d)) {}

    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// GPT-2 byte-level decoder: reverses bytes_char mapping.
class ByteLevelDecoder : public Decoder {
public:
    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// BPE decoder: replaces end-of-word suffix with space.
class BPEDecoder : public Decoder {
public:
    std::string suffix = "</w>";

    BPEDecoder() = default;
    explicit BPEDecoder(std::string suffix) : suffix(std::move(suffix)) {}

    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// ByteFallback decoder: converts <0xNN> tokens back to bytes.
class ByteFallbackDecoder : public Decoder {
public:
    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// CTC decoder: deduplicates, removes pad tokens, replaces word delimiter.
class CTCDecoder : public Decoder {
public:
    std::string pad_token = "<pad>";
    std::string word_delimiter_token = "|";
    bool cleanup = true;

    CTCDecoder() = default;
    CTCDecoder(std::string pad, std::string delim, bool cleanup)
        : pad_token(std::move(pad)), word_delimiter_token(std::move(delim)),
          cleanup(cleanup) {}

    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// Fuse decoder: joins all tokens into a single string.
class FuseDecoder : public Decoder {
public:
    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// Strip decoder: strips occurrences of a character from token ends.
class StripDecoder : public Decoder {
public:
    char32_t content = ' ';
    size_t start = 0;
    size_t stop = 0;

    StripDecoder() = default;
    StripDecoder(char32_t content, size_t start, size_t stop)
        : content(content), start(start), stop(stop) {}

    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

/// Metaspace decoder: reverses Metaspace pre-tokenizer (replaces ▁ with space).
class MetaspaceDecoder : public Decoder {
public:
    std::string replacement = "\xE2\x96\x81"; // ▁ U+2581
    bool prepend_scheme_always = true;

    MetaspaceDecoder() = default;
    MetaspaceDecoder(std::string replacement, bool prepend_scheme_always)
        : replacement(std::move(replacement)),
          prepend_scheme_always(prepend_scheme_always) {}

    Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const override;
};

} // namespace decoders
} // namespace tokenizers
