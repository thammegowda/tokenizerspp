#pragma once
/// @file tokenizers/pre_tokenized_string.h
/// PreTokenizedString: manages splitting → normalizing → tokenizing.
/// Mirrors tokenizers/src/tokenizer/pre_tokenizer.rs

#include "tokenizers/common.h"
#include "tokenizers/encoding.h"
#include "tokenizers/error.h"
#include "tokenizers/normalized_string.h"
#include "tokenizers/token.h"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tokenizers {

/// Offset type for encoding construction.
enum class OffsetType {
    Byte,
    Char,
    None,
};

/// A single split within a PreTokenizedString.
struct Split {
    NormalizedString normalized;
    std::optional<std::vector<Token>> tokens;

    explicit Split(NormalizedString n) : normalized(std::move(n)) {}
    Split(NormalizedString n, std::optional<std::vector<Token>> t)
        : normalized(std::move(n)), tokens(std::move(t)) {}
};

/// Manages the split/normalize/tokenize pipeline over a string.
class PreTokenizedString {
public:
    explicit PreTokenizedString(const std::string& input);
    explicit PreTokenizedString(NormalizedString input);

    /// Split each un-tokenized part using split_fn.
    /// split_fn receives (split_index, NormalizedString) and returns splits.
    using SplitFn = std::function<Result<std::vector<NormalizedString>>(size_t, NormalizedString)>;
    Result<void> split(SplitFn split_fn);

    /// Normalize all un-tokenized splits.
    using NormalizeFn = std::function<Result<void>(NormalizedString&)>;
    Result<void> normalize(NormalizeFn normalize_fn);

    /// Tokenize all un-tokenized splits.
    using TokenizeFn = std::function<Result<std::vector<Token>>(const NormalizedString&)>;
    Result<void> tokenize(TokenizeFn tokenize_fn);

    /// Convert to an Encoding.
    Result<Encoding> into_encoding(std::optional<TokenId> word_idx,
                                   TokenId type_id,
                                   OffsetType offset_type) const;

    /// Get splits info for inspection.
    struct SplitRef {
        std::string_view text;
        Offsets offsets;
        const std::optional<std::vector<Token>>* tokens;
    };
    [[nodiscard]] std::vector<SplitRef> get_splits(
        OffsetReferential offset_ref, OffsetType offset_type) const;

private:
    std::string original_;
    std::vector<Split> splits_;
};

} // namespace tokenizers
