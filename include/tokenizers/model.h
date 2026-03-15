#pragma once
/// @file tokenizers/model.h
/// Model trait: the core tokenization algorithm (BPE, WordPiece, Unigram, WordLevel).

#include "tokenizers/common.h"
#include "tokenizers/error.h"
#include "tokenizers/token.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tokenizers {

/// Base class for tokenization models.
class Model {
public:
    virtual ~Model() = default;

    /// Tokenize a single (pre-tokenized) sequence into Tokens.
    /// Offsets in the returned tokens are relative to the input sequence.
    [[nodiscard]] virtual Result<std::vector<Token>>
    tokenize(std::string_view sequence) const = 0;

    /// Look up the ID for a token string. Returns nullopt if not found.
    [[nodiscard]] virtual std::optional<TokenId>
    token_to_id(std::string_view token) const = 0;

    /// Look up the token string for an ID. Returns nullopt if not found.
    [[nodiscard]] virtual std::optional<std::string>
    id_to_token(TokenId id) const = 0;

    /// Get the full vocabulary mapping (token → id).
    [[nodiscard]] virtual std::unordered_map<std::string, TokenId>
    get_vocab() const = 0;

    /// Get the vocabulary size.
    [[nodiscard]] virtual size_t get_vocab_size() const = 0;
};

using ModelPtr = std::unique_ptr<Model>;

} // namespace tokenizers
