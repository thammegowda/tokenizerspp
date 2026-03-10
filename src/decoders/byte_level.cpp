#include "tokenizers/decoders.h"
#include "tokenizers/pre_tokenizers.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tokenizers {
namespace decoders {

Result<std::vector<std::string>>
ByteLevelDecoder::decode_chain(std::vector<std::string> tokens) const {
    auto& c2b = pre_tokenizers::ByteLevel::char_bytes_array();
    auto& c2b_valid = pre_tokenizers::ByteLevel::char_bytes_valid();

    std::vector<std::string> result;
    result.reserve(tokens.size());

    for (auto& token : tokens) {
        std::vector<uint8_t> bytes;
        bytes.reserve(token.size());

        // Iterate code points in the token
        size_t i = 0;
        auto sv = std::string_view(token);
        while (i < sv.size()) {
            // Decode UTF-8 code point
            auto ch = static_cast<unsigned char>(sv[i]);
            char32_t cp;
            size_t len;
            if ((ch & 0x80) == 0) {
                cp = ch; len = 1;
            } else if ((ch & 0xE0) == 0xC0) {
                cp = ch & 0x1F; len = 2;
            } else if ((ch & 0xF0) == 0xE0) {
                cp = ch & 0x0F; len = 3;
            } else {
                cp = ch & 0x07; len = 4;
            }
            for (size_t j = 1; j < len && (i + j) < sv.size(); ++j) {
                cp = (cp << 6) | (static_cast<unsigned char>(sv[i + j]) & 0x3F);
            }

            auto byte_idx = static_cast<size_t>(cp);
            if (byte_idx < c2b_valid.size() && c2b_valid[byte_idx]) {
                bytes.push_back(c2b[byte_idx]);
            } else {
                // Not in mapping — use the original UTF-8 bytes
                for (size_t j = 0; j < len && (i + j) < sv.size(); ++j) {
                    bytes.push_back(static_cast<uint8_t>(sv[i + j]));
                }
            }
            i += len;
        }

        result.emplace_back(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

    return result;
}

} // namespace decoders
} // namespace tokenizers
