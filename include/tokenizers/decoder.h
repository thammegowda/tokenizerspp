#pragma once
/// @file tokenizers/decoder.h
/// Decoder trait: converts token strings back to readable text.

#include "tokenizers/common.h"
#include "tokenizers/error.h"
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {

/// Base class for decoders.
class Decoder {
public:
    virtual ~Decoder() = default;

    /// Decode a list of token strings into a single string.
    [[nodiscard]] virtual Result<std::string>
    decode(std::vector<std::string> tokens) const {
        auto result = decode_chain(std::move(tokens));
        if (!result) return std::unexpected(result.error());
        std::string out;
        for (auto& s : *result) out += s;
        return out;
    }

    /// Decode a list of token strings, returning each segment separately.
    [[nodiscard]] virtual Result<std::vector<std::string>>
    decode_chain(std::vector<std::string> tokens) const = 0;
};

using DecoderPtr = std::unique_ptr<Decoder>;

} // namespace tokenizers
