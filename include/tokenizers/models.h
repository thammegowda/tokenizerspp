#pragma once
/// @file tokenizers/models.h
/// Concrete model implementations.

#include "tokenizers/model.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizers {
namespace models {

/// WordPiece model (used by BERT).
class WordPiece : public Model {
public:
    WordPiece() = default;

    WordPiece(std::unordered_map<std::string, TokenId> vocab,
              std::string unk_token = "[UNK]",
              std::string continuing_subword_prefix = "##",
              size_t max_input_chars_per_word = 100);

    Result<std::vector<Token>> tokenize(std::string_view sequence) const override;
    std::optional<TokenId> token_to_id(std::string_view token) const override;
    std::optional<std::string> id_to_token(TokenId id) const override;
    std::unordered_map<std::string, TokenId> get_vocab() const override;
    size_t get_vocab_size() const override;

    const std::string& get_unk_token() const { return unk_token_; }
    const std::string& get_continuing_subword_prefix() const { return continuing_subword_prefix_; }
    size_t get_max_input_chars_per_word() const { return max_input_chars_per_word_; }

private:
    std::unordered_map<std::string, TokenId> vocab_;
    std::unordered_map<TokenId, std::string> vocab_r_;
    std::string unk_token_ = "[UNK]";
    std::string continuing_subword_prefix_ = "##";
    size_t max_input_chars_per_word_ = 100;
};

/// Pair of token IDs used as merge key.
using Pair = std::pair<TokenId, TokenId>;

/// Hash for Pair.
struct PairHash {
    size_t operator()(const Pair& p) const {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(p.first) << 32) | p.second);
    }
};

/// MergeMap: pair → (rank, new_id).
using MergeMap = std::unordered_map<Pair, std::pair<TokenId, TokenId>, PairHash>;

/// BPE model (used by GPT-2, RoBERTa, etc.).
class BPE : public Model {
public:
    BPE() = default;
    BPE(std::unordered_map<std::string, TokenId> vocab,
        MergeMap merges,
        std::optional<std::string> unk_token = std::nullopt,
        std::optional<std::string> continuing_subword_prefix = std::nullopt,
        std::optional<std::string> end_of_word_suffix = std::nullopt,
        bool fuse_unk = false,
        bool byte_fallback = false,
        bool ignore_merges = false);

    Result<std::vector<Token>> tokenize(std::string_view sequence) const override;
    std::optional<TokenId> token_to_id(std::string_view token) const override;
    std::optional<std::string> id_to_token(TokenId id) const override;
    std::unordered_map<std::string, TokenId> get_vocab() const override;
    size_t get_vocab_size() const override;

    const std::optional<std::string>& get_unk_token() const { return unk_token_; }
    const std::optional<std::string>& get_continuing_subword_prefix() const { return continuing_subword_prefix_; }
    const std::optional<std::string>& get_end_of_word_suffix() const { return end_of_word_suffix_; }
    bool get_fuse_unk() const { return fuse_unk_; }
    bool get_byte_fallback() const { return byte_fallback_; }
    bool get_ignore_merges() const { return ignore_merges_; }
    const MergeMap& get_merges() const { return merges_; }

private:
    std::unordered_map<std::string, TokenId> vocab_;
    std::unordered_map<TokenId, std::string> vocab_r_;
    MergeMap merges_;
    std::optional<std::string> unk_token_;
    std::optional<std::string> continuing_subword_prefix_;
    std::optional<std::string> end_of_word_suffix_;
    bool fuse_unk_ = false;
    bool byte_fallback_ = false;
    bool ignore_merges_ = false;

    // Transparent hasher for string_view lookups without allocation
    struct StringHash {
        using is_transparent = void;
        size_t operator()(std::string_view sv) const { return std::hash<std::string_view>{}(sv); }
        size_t operator()(const std::string& s) const { return std::hash<std::string_view>{}(s); }
    };
    struct StringEqual {
        using is_transparent = void;
        bool operator()(std::string_view a, std::string_view b) const { return a == b; }
    };

    // Word-level cache: string → merged token list (thread-safe via mutable)
    static constexpr size_t MAX_CACHE_WORD_LEN = 128;
    mutable std::unordered_map<std::string, std::vector<Token>, StringHash, StringEqual> cache_;

    Result<std::vector<Token>> merge_word(std::string_view sequence) const;
    Result<std::vector<Token>> merge_word_uncached(std::string_view sequence) const;
};

/// WordLevel model: simple whole-word lookup.
class WordLevel : public Model {
public:
    WordLevel() = default;
    WordLevel(std::unordered_map<std::string, TokenId> vocab,
              std::string unk_token = "[UNK]");

    Result<std::vector<Token>> tokenize(std::string_view sequence) const override;
    std::optional<TokenId> token_to_id(std::string_view token) const override;
    std::optional<std::string> id_to_token(TokenId id) const override;
    std::unordered_map<std::string, TokenId> get_vocab() const override;
    size_t get_vocab_size() const override;

    const std::string& get_unk_token() const { return unk_token_; }

private:
    std::unordered_map<std::string, TokenId> vocab_;
    std::unordered_map<TokenId, std::string> vocab_r_;
    std::string unk_token_ = "[UNK]";
};

/// Unigram model (SentencePiece-style).
class Unigram : public Model {
public:
    /// vocab is a list of (token, score) pairs. unk_id is the index for unknown token.
    Unigram(std::vector<std::pair<std::string, double>> vocab,
            std::optional<size_t> unk_id, bool byte_fallback = false);

    Result<std::vector<Token>> tokenize(std::string_view sequence) const override;
    std::optional<TokenId> token_to_id(std::string_view token) const override;
    std::optional<std::string> id_to_token(TokenId id) const override;
    std::unordered_map<std::string, TokenId> get_vocab() const override;
    size_t get_vocab_size() const override;

    const std::vector<std::pair<std::string, double>>& get_vocab_scores() const;
    std::optional<size_t> get_unk_id() const;
    bool get_byte_fallback() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace models
} // namespace tokenizers
