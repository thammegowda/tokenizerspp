#pragma once
/// @file tokenizers/pre_tokenizers.h
/// All concrete pre-tokenizer implementations.

#include "tokenizers/pre_tokenizer.h"
#include "tokenizers/pre_tokenized_string.h"
#include "tokenizers/normalizer.h"
#include "tokenizers/pattern.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {
namespace pre_tokenizers {

/// BERT pre-tokenizer: splits on whitespace, then isolates punctuation.
class BertPreTokenizer : public PreTokenizer {
public:
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Splits on whitespace, keeps punctuation attached to words.
class WhitespaceSplit : public PreTokenizer {
public:
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Regex-based whitespace pre-tokenizer: \w+|[^\w\s]+
class Whitespace : public PreTokenizer {
public:
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Splits on a single delimiter character.
class CharDelimiterSplit : public PreTokenizer {
public:
    char32_t delimiter;
    explicit CharDelimiterSplit(char32_t d) : delimiter(d) {}
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Splits on digit/non-digit boundaries.
class Digits : public PreTokenizer {
public:
    bool individual_digits = false;
    Digits() = default;
    explicit Digits(bool individual) : individual_digits(individual) {}
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Punctuation pre-tokenizer: isolates punctuation characters.
class Punctuation : public PreTokenizer {
public:
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// GPT-2 byte-level BPE pre-tokenizer.
class ByteLevel : public PreTokenizer {
public:
    bool add_prefix_space = true;
    bool trim_offsets = true;
    bool use_regex = true;

    ByteLevel() = default;
    ByteLevel(bool add_prefix_space, bool trim_offsets, bool use_regex);
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;

    /// Byte → Unicode char mapping (GPT-2 style) — flat 256-entry array.
    static const std::array<char32_t, 256>& bytes_char_array();
    /// Reverse: Unicode char → byte — flat array (max codepoint 323).
    static const std::array<uint8_t, 324>& char_bytes_array();
    /// Validity array for reverse mapping.
    static const std::array<bool, 324>& char_bytes_valid();
};

/// Sequence of pre-tokenizers.
class SequencePreTokenizer : public PreTokenizer {
public:
    std::vector<PreTokenizerPtr> pre_tokenizers;
    explicit SequencePreTokenizer(std::vector<PreTokenizerPtr> pts)
        : pre_tokenizers(std::move(pts)) {}
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Metaspace pre-tokenizer (SentencePiece-style: replaces spaces with ▁).
class Metaspace : public PreTokenizer {
public:
    enum class PrependScheme { First, Never, Always };

    std::string replacement = "\xE2\x96\x81"; // ▁ U+2581
    char32_t replacement_char = U'\u2581';
    PrependScheme prepend_scheme = PrependScheme::Always;
    bool split = true;

    Metaspace() = default;
    Metaspace(char32_t replacement, PrependScheme prepend_scheme, bool split);
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Split pre-tokenizer: splits on a pattern (string or regex).
class SplitPreTokenizer : public PreTokenizer {
public:
    std::unique_ptr<Pattern> pattern;
    SplitDelimiterBehavior behavior;
    bool invert = false;

    SplitPreTokenizer(std::unique_ptr<Pattern> pattern,
                      SplitDelimiterBehavior behavior, bool invert = false);
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Splits text on Unicode script boundaries.
class UnicodeScripts : public PreTokenizer {
public:
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

/// Splits input into fixed-length byte chunks.
class FixedLength : public PreTokenizer {
public:
    size_t length = 0;
    explicit FixedLength(size_t length) : length(length) {}
    Result<void> pre_tokenize(PreTokenizedString& pretokenized) const override;
};

} // namespace pre_tokenizers
} // namespace tokenizers
