#pragma once
/// @file tokenizers/post_processor.h
/// PostProcessor trait: adds special tokens after tokenization.

#include "tokenizers/common.h"
#include "tokenizers/encoding.h"
#include "tokenizers/error.h"
#include <memory>
#include <optional>
#include <vector>

namespace tokenizers {

/// Base class for post-processors.
class PostProcessor {
public:
    virtual ~PostProcessor() = default;

    /// Returns the number of tokens added during processing.
    [[nodiscard]] virtual size_t added_tokens(bool is_pair) const = 0;

    /// Process encodings (1 or 2) and return merged result(s).
    [[nodiscard]] virtual Result<std::vector<Encoding>>
    process_encodings(std::vector<Encoding> encodings, bool add_special_tokens) const = 0;

    /// Convenience: process a single encoding (or pair) into one merged Encoding.
    [[nodiscard]] Result<Encoding>
    process(Encoding encoding, std::optional<Encoding> pair_encoding,
            bool add_special_tokens) const;
};

using PostProcessorPtr = std::unique_ptr<PostProcessor>;

} // namespace tokenizers
