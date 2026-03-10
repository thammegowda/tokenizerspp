#include "tokenizers/decoders.h"

#include <string>
#include <vector>

namespace tokenizers {
namespace decoders {

// Simplified WordPiece-style cleanup: spaces before punctuation etc.
static std::string wp_cleanup(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == ' ' && i + 1 < s.size()) {
            char next = s[i + 1];
            // Remove space before common punctuation
            if (next == '.' || next == '?' || next == '!' || next == ',' ||
                next == '\'' || next == ')' || next == ']' || next == '}' ||
                next == ':' || next == ';') {
                continue;
            }
        }
        // Handle common contractions: " 's" → "'s", " n't" → "n't" etc.
        out += c;
    }
    return out;
}

Result<std::vector<std::string>>
CTCDecoder::decode_chain(std::vector<std::string> tokens) const {
    std::vector<std::string> result;

    // 1. Deduplicate consecutive tokens
    std::vector<std::string> deduped;
    for (auto& token : tokens) {
        if (deduped.empty() || deduped.back() != token) {
            deduped.push_back(std::move(token));
        }
    }

    // 2. Process each token
    for (auto& token : deduped) {
        // Remove pad_token occurrences
        std::string replaced;
        size_t pos = 0;
        while (pos < token.size()) {
            auto found = token.find(pad_token, pos);
            if (found == std::string::npos) {
                replaced.append(token, pos);
                break;
            }
            replaced.append(token, pos, found - pos);
            pos = found + pad_token.size();
        }

        if (cleanup) {
            replaced = wp_cleanup(replaced);
            // Replace word_delimiter_token with space
            std::string final_str;
            pos = 0;
            while (pos < replaced.size()) {
                auto found = replaced.find(word_delimiter_token, pos);
                if (found == std::string::npos) {
                    final_str.append(replaced, pos);
                    break;
                }
                final_str.append(replaced, pos, found - pos);
                final_str += ' ';
                pos = found + word_delimiter_token.size();
            }
            replaced = std::move(final_str);
        }

        if (!replaced.empty()) {
            result.push_back(std::move(replaced));
        }
    }

    return result;
}

} // namespace decoders
} // namespace tokenizers
