#include "tokenizers/normalizers.h"

#include <uni_algo/prop.h>

namespace tokenizers {
namespace normalizers {

namespace {

bool is_whitespace(char32_t c) {
    switch (c) {
        case '\t': case '\n': case '\r': return true;
        default: return una::codepoint::prop(c).White_Space();
    }
}

bool is_control(char32_t c) {
    switch (c) {
        case '\t': case '\n': case '\r': return false;
        default: {
            auto p = una::codepoint::prop(c);
            return p.General_Category_Cc() ||
                   p.General_Category_Cf() ||
                   p.General_Category_Co();
        }
    }
}

bool is_chinese_char(char32_t c) {
    auto cp = static_cast<uint32_t>(c);
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// Decode one UTF-8 codepoint
std::pair<char32_t, size_t> decode_utf8(const char* p, size_t remaining) {
    auto b = static_cast<uint8_t>(*p);
    if (b < 0x80) return {b, 1};
    if (remaining >= 2 && (b & 0xE0) == 0xC0) {
        return {(b & 0x1F) << 6 | (static_cast<uint8_t>(p[1]) & 0x3F), 2};
    }
    if (remaining >= 3 && (b & 0xF0) == 0xE0) {
        return {(b & 0x0F) << 12 | (static_cast<uint8_t>(p[1]) & 0x3F) << 6 | (static_cast<uint8_t>(p[2]) & 0x3F), 3};
    }
    if (remaining >= 4 && (b & 0xF8) == 0xF0) {
        return {(b & 0x07) << 18 | (static_cast<uint8_t>(p[1]) & 0x3F) << 12 | (static_cast<uint8_t>(p[2]) & 0x3F) << 6 | (static_cast<uint8_t>(p[3]) & 0x3F), 4};
    }
    return {0xFFFD, 1};
}

} // namespace

Result<void> BertNormalizer::normalize(NormalizedString& normalized) const {
    if (clean_text) {
        normalized.filter([](char32_t c) {
            return !(c == 0 || c == 0xFFFD || is_control(c));
        });
        normalized.map([](char32_t c) -> char32_t {
            return is_whitespace(c) ? ' ' : c;
        });
    }
    if (handle_chinese_chars) {
        std::vector<std::pair<char32_t, int>> new_chars;
        std::string_view sv = normalized.get();
        size_t pos = 0;
        while (pos < sv.size()) {
            auto [cp, len] = decode_utf8(sv.data() + pos, sv.size() - pos);
            if (is_chinese_char(cp)) {
                new_chars.push_back({' ', 0});
                new_chars.push_back({cp, 1});
                new_chars.push_back({' ', 1});
            } else {
                new_chars.push_back({cp, 0});
            }
            pos += len;
        }
        normalized.transform(new_chars, 0);
    }
    bool do_strip = strip_accents.value_or(lowercase);
    if (do_strip) {
        normalized.nfd();
        normalized.filter([](char32_t c) {
            return !una::codepoint::prop(c).General_Category_Mn();
        });
    }
    if (lowercase) {
        normalized.lowercase();
    }
    return {};
}

} // namespace normalizers
} // namespace tokenizers
