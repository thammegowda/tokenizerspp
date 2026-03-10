#include "tokenizers/decoders.h"

#include <algorithm>
#include <string>

namespace tokenizers {
namespace decoders {

Result<std::vector<std::string>>
MetaspaceDecoder::decode_chain(std::vector<std::string> tokens) const {
    for (auto& token : tokens) {
        // Replace all occurrences of replacement char with space
        std::string result;
        result.reserve(token.size());
        size_t pos = 0;
        while (pos < token.size()) {
            if (token.compare(pos, replacement.size(), replacement) == 0) {
                result += ' ';
                pos += replacement.size();
            } else {
                result += token[pos];
                ++pos;
            }
        }
        token = std::move(result);
    }

    // Strip leading space from first token (added by prepend)
    if (!tokens.empty() && !tokens[0].empty() && tokens[0][0] == ' ') {
        tokens[0] = tokens[0].substr(1);
    }

    return tokens;
}

} // namespace decoders
} // namespace tokenizers
