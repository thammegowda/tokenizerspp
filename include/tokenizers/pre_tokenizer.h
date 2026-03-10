#pragma once
/// @file tokenizers/pre_tokenizer.h
/// PreTokenizer trait.

#include "tokenizers/common.h"
#include "tokenizers/error.h"
#include <memory>

namespace tokenizers {

class PreTokenizedString;

/// Base class for all pre-tokenizers.
class PreTokenizer {
public:
    virtual ~PreTokenizer() = default;
    [[nodiscard]] virtual Result<void> pre_tokenize(PreTokenizedString& pretokenized) const = 0;
};

using PreTokenizerPtr = std::unique_ptr<PreTokenizer>;

} // namespace tokenizers
