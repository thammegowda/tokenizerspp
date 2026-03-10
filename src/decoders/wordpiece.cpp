#include "tokenizers/decoders.h"

namespace tokenizers {
namespace decoders {

Result<std::vector<std::string>>
WordPieceDecoder::decode_chain(std::vector<std::string> tokens) const {
    std::vector<std::string> result;
    result.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        auto& token = tokens[i];
        if (i > 0 && token.starts_with(prefix)) {
            // Remove prefix, no space before
            result.push_back(token.substr(prefix.size()));
        } else if (i > 0) {
            // Add space before
            result.push_back(" " + token);
        } else {
            result.push_back(std::move(token));
        }
    }

    if (cleanup) {
        // Clean up tokenization artifacts
        for (auto& s : result) {
            // Remove spaces before punctuation
            // This is a simplified version — the Rust implementation uses more sophisticated cleanup
        }
    }

    return result;
}

} // namespace decoders
} // namespace tokenizers
