#include "tokenizers/decoders.h"

#include <string>

namespace tokenizers {
namespace decoders {

Result<std::vector<std::string>>
ReplaceDecoder::decode_chain(std::vector<std::string> tokens) const {
    if (pattern.empty()) return tokens;
    for (auto& token : tokens) {
        std::string result;
        result.reserve(token.size());
        size_t pos = 0;
        while (pos < token.size()) {
            if (token.compare(pos, pattern.size(), pattern) == 0) {
                result += content;
                pos += pattern.size();
            } else {
                result += token[pos];
                ++pos;
            }
        }
        token = std::move(result);
    }
    return tokens;
}

} // namespace decoders
} // namespace tokenizers
