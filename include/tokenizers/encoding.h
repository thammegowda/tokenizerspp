#pragma once
/// @file tokenizers/encoding.h
/// Encoding struct: output of the tokenizer pipeline.
/// Mirrors tokenizers/src/tokenizer/encoding.rs

#include "tokenizers/common.h"
#include "tokenizers/token.h"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizers {

/// The output of a Tokenizer.
class Encoding {
public:
    Encoding() = default;

    Encoding(std::vector<TokenId> ids,
             std::vector<TokenId> type_ids,
             std::vector<std::string> tokens,
             std::vector<std::optional<TokenId>> words,
             std::vector<Offsets> offsets,
             std::vector<TokenId> special_tokens_mask,
             std::vector<TokenId> attention_mask,
             std::vector<Encoding> overflowing,
             std::unordered_map<size_t, std::pair<size_t, size_t>> sequence_ranges);

    static Encoding with_capacity(size_t len);
    static Encoding from_tokens(const std::vector<Token>& tokens, TokenId type_id);

    // Accessors
    [[nodiscard]] bool is_empty() const { return ids_.empty(); }
    [[nodiscard]] size_t len() const { return ids_.size(); }
    [[nodiscard]] size_t n_sequences() const;

    [[nodiscard]] const std::vector<TokenId>& get_ids() const { return ids_; }
    [[nodiscard]] const std::vector<TokenId>& get_type_ids() const { return type_ids_; }
    [[nodiscard]] const std::vector<std::string>& get_tokens() const { return tokens_; }
    [[nodiscard]] const std::vector<std::optional<TokenId>>& get_words() const { return words_; }
    [[nodiscard]] const std::vector<Offsets>& get_offsets() const { return offsets_; }
    [[nodiscard]] const std::vector<TokenId>& get_special_tokens_mask() const { return special_tokens_mask_; }
    [[nodiscard]] const std::vector<TokenId>& get_attention_mask() const { return attention_mask_; }
    [[nodiscard]] const std::vector<Encoding>& get_overflowing() const { return overflowing_; }
    [[nodiscard]] std::vector<Encoding>& get_overflowing_mut() { return overflowing_; }

    // Mutators
    void set_sequence_id(size_t id);
    void set_type_ids(std::vector<TokenId> type_ids) { type_ids_ = std::move(type_ids); }

    /// Merge another encoding into this one
    void merge_with(Encoding other, bool growing_offsets);

    /// Merge multiple encodings
    static Encoding merge(std::vector<Encoding> encodings, bool growing_offsets);

    /// Truncate to max_length, with stride for overflow
    void truncate(size_t max_length, size_t stride, bool from_right = true);

    /// Pad to target length
    void pad(size_t target_length, TokenId pad_id, TokenId pad_type_id,
             const std::string& pad_token, bool pad_right = true);

    bool operator==(const Encoding&) const = default;

private:
    std::vector<TokenId> ids_;
    std::vector<TokenId> type_ids_;
    std::vector<std::string> tokens_;
    std::vector<std::optional<TokenId>> words_;
    std::vector<Offsets> offsets_;
    std::vector<TokenId> special_tokens_mask_;
    std::vector<TokenId> attention_mask_;
    std::vector<Encoding> overflowing_;
    std::unordered_map<size_t, std::pair<size_t, size_t>> sequence_ranges_;
};

} // namespace tokenizers
