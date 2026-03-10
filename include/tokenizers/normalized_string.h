#pragma once
/// @file tokenizers/normalized_string.h
/// NormalizedString: tracks original + normalized text with alignment.
/// Mirrors tokenizers/src/tokenizer/normalizer.rs (NormalizedString struct).

#include "tokenizers/common.h"
#include "tokenizers/error.h"
#include "tokenizers/normalizer.h"
#include "tokenizers/pattern.h"
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tokenizers {

/// Whether offsets are relative to the original or normalized string.
enum class OffsetReferential {
    Original,
    Normalized,
};

/// A range that can refer to Original or Normalized offsets.
struct Range {
    OffsetReferential referential;
    size_t start;
    size_t end;

    static Range original(size_t start, size_t end) {
        return {OffsetReferential::Original, start, end};
    }
    static Range normalized(size_t start, size_t end) {
        return {OffsetReferential::Normalized, start, end};
    }
};

/// A NormalizedString tracks both the original string and a "normalized" version,
/// maintaining byte-level alignment between them so offsets can be converted.
class NormalizedString {
public:
    NormalizedString() = default;
    explicit NormalizedString(std::string input);

    // Accessors
    [[nodiscard]] std::string_view get() const { return normalized_; }
    [[nodiscard]] std::string_view get_original() const { return original_; }
    [[nodiscard]] Offsets offsets_original() const {
        return {original_shift_, original_shift_ + original_.size()};
    }
    [[nodiscard]] size_t len() const { return normalized_.size(); }
    [[nodiscard]] size_t len_original() const { return original_.size(); }
    [[nodiscard]] bool is_empty() const { return normalized_.empty(); }

    /// Convert offsets from one referential to the other.
    [[nodiscard]] std::optional<std::pair<size_t, size_t>>
    convert_offsets(Range range) const;

    /// Get a substring of the normalized string using a Range.
    [[nodiscard]] std::optional<std::string_view> get_range(Range range) const;

    /// Get a substring of the original string using a Range.
    [[nodiscard]] std::optional<std::string_view> get_range_original(Range range) const;

    /// Create a slice (sub-NormalizedString) from a Range.
    [[nodiscard]] std::optional<NormalizedString> slice(Range range) const;

    // === Transformation methods ===

    /// Low-level transform: apply character-level changes with alignment tracking.
    /// Each pair is (new_char, change) where change is:
    ///   0  = replacing the current char
    ///   1  = inserting a new char
    ///  -N  = replacing current char and removing N following chars
    /// `initial_offset` = number of removed chars at the very beginning.
    void transform(const std::vector<std::pair<char32_t, int>>& dest,
                   size_t initial_offset);

    /// transform_range: like transform but restricted to a range.
    void transform_range(Range range,
                         const std::vector<std::pair<char32_t, int>>& dest,
                         size_t initial_offset);

    /// Optimized byte-level encoding: maps each byte through a lookup table.
    /// Replaces the normalized string with the byte-level encoded version,
    /// building alignments directly in one pass.
    void byte_level_encode(const std::array<char32_t, 256>& b2c);

    // === High-level normalization operations ===
    NormalizedString& nfc();
    NormalizedString& nfkc();
    NormalizedString& nfd();
    NormalizedString& nfkd();
    NormalizedString& lowercase();
    NormalizedString& uppercase();
    NormalizedString& strip();
    NormalizedString& lstrip();
    NormalizedString& rstrip();
    NormalizedString& prepend(std::string_view s);
    NormalizedString& append(std::string_view s);
    NormalizedString& map(std::function<char32_t(char32_t)> func);
    NormalizedString& filter(std::function<bool(char32_t)> keep);
    Result<void> replace(const Pattern& pattern, std::string_view content);
    size_t clear();

    /// Split the normalized string by a pattern.
    [[nodiscard]] Result<std::vector<NormalizedString>>
    split(const Pattern& pattern, SplitDelimiterBehavior behavior) const;

    bool operator==(const NormalizedString&) const = default;

    // Test accessors — expose internals for verification
    [[nodiscard]] const std::vector<std::pair<size_t, size_t>>& alignments() const {
        materialize_alignments();
        return alignments_;
    }
    [[nodiscard]] size_t original_shift() const { return original_shift_; }

private:
    std::string original_;
    std::string normalized_;
    mutable std::vector<std::pair<size_t, size_t>> alignments_;  // per-byte of normalized → original
    size_t original_shift_ = 0;
    mutable bool identity_alignments_ = false;  // if true, alignments_ is empty and identity

    // Materialize identity alignments into the real vector when needed
    void materialize_alignments() const;

    // Internal: validate and expand a Range to concrete byte indices
    std::optional<std::pair<size_t, size_t>> validate_range(Range range) const;

    void lrstrip(bool left, bool right);
};

} // namespace tokenizers
