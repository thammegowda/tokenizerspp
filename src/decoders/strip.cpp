#include "tokenizers/decoders.h"

#include <string>
#include <vector>

namespace tokenizers {
namespace decoders {

// Decode one UTF-8 code point from the string view starting at pos.
// Returns the code point and advances pos past it.
static char32_t decode_utf8(std::string_view sv, size_t& pos) {
    auto ch = static_cast<unsigned char>(sv[pos]);
    char32_t cp;
    size_t len;
    if ((ch & 0x80) == 0) { cp = ch; len = 1; }
    else if ((ch & 0xE0) == 0xC0) { cp = ch & 0x1F; len = 2; }
    else if ((ch & 0xF0) == 0xE0) { cp = ch & 0x0F; len = 3; }
    else { cp = ch & 0x07; len = 4; }
    for (size_t j = 1; j < len && (pos + j) < sv.size(); ++j)
        cp = (cp << 6) | (static_cast<unsigned char>(sv[pos + j]) & 0x3F);
    pos += len;
    return cp;
}

// Encode a code point to UTF-8 and append to string.
static void encode_utf8(char32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

Result<std::vector<std::string>>
StripDecoder::decode_chain(std::vector<std::string> tokens) const {
    std::vector<std::string> result;
    result.reserve(tokens.size());

    for (auto& token : tokens) {
        // Collect code points
        std::vector<char32_t> chars;
        size_t pos = 0;
        auto sv = std::string_view(token);
        while (pos < sv.size()) {
            chars.push_back(decode_utf8(sv, pos));
        }

        // Count left strip
        size_t left = 0;
        for (size_t i = 0; i < start && i < chars.size(); ++i) {
            if (chars[i] == content) left = i + 1;
            else break;
        }

        // Count right strip
        size_t right = 0;
        for (size_t i = 0; i < stop && i < chars.size(); ++i) {
            size_t idx = chars.size() - 1 - i;
            if (chars[idx] == content) right = i + 1;
            else break;
        }

        size_t end_idx = chars.size() >= right ? chars.size() - right : 0;
        if (left > end_idx) left = end_idx;

        std::string out;
        for (size_t i = left; i < end_idx; ++i) {
            encode_utf8(chars[i], out);
        }
        result.push_back(std::move(out));
    }

    return result;
}

} // namespace decoders
} // namespace tokenizers
