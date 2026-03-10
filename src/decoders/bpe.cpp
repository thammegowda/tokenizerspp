#include "tokenizers/decoders.h"

#include <string>
#include <vector>

namespace tokenizers {
namespace decoders {

Result<std::vector<std::string>>
BPEDecoder::decode_chain(std::vector<std::string> tokens) const {
    if (tokens.empty()) return tokens;
    std::vector<std::string> result;
    result.reserve(tokens.size());
    size_t n = tokens.size() - 1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto& token = tokens[i];
        std::string replaced;
        // Replace all occurrences of suffix in the token
        size_t pos = 0;
        while (pos < token.size()) {
            auto found = token.find(suffix, pos);
            if (found == std::string::npos) {
                replaced.append(token, pos);
                break;
            }
            replaced.append(token, pos, found - pos);
            if (i != n) replaced += ' ';
            pos = found + suffix.size();
        }
        result.push_back(std::move(replaced));
    }
    return result;
}

} // namespace decoders
} // namespace tokenizers
