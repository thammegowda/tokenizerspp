#include "tokenizers/decoders.h"

#include <charconv>
#include <cstdint>
#include <string>
#include <vector>

namespace tokenizers {
namespace decoders {

static std::optional<uint8_t> parse_hex_byte(const std::string& token) {
    // Match format: <0xNN> (exactly 6 chars)
    if (token.size() == 6 && token[0] == '<' && token[1] == '0' &&
        token[2] == 'x' && token[5] == '>') {
        uint8_t val = 0;
        auto [ptr, ec] = std::from_chars(token.data() + 3, token.data() + 5, val, 16);
        if (ec == std::errc{} && ptr == token.data() + 5) return val;
    }
    return std::nullopt;
}

static void flush_bytes(std::vector<uint8_t>& bytes, std::vector<std::string>& out) {
    if (bytes.empty()) return;
    // Check if valid UTF-8
    // Simple UTF-8 validation
    auto sv = std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    bool valid = true;
    size_t i = 0;
    while (i < sv.size()) {
        auto ch = static_cast<uint8_t>(sv[i]);
        size_t len;
        if ((ch & 0x80) == 0) len = 1;
        else if ((ch & 0xE0) == 0xC0) len = 2;
        else if ((ch & 0xF0) == 0xE0) len = 3;
        else if ((ch & 0xF8) == 0xF0) len = 4;
        else { valid = false; break; }
        if (i + len > sv.size()) { valid = false; break; }
        for (size_t j = 1; j < len; ++j) {
            if ((static_cast<uint8_t>(sv[i + j]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
        i += len;
    }
    if (valid) {
        out.emplace_back(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    } else {
        // Each byte becomes U+FFFD
        for (size_t j = 0; j < bytes.size(); ++j) {
            out.push_back("\xEF\xBF\xBD"); // U+FFFD in UTF-8
        }
    }
    bytes.clear();
}

Result<std::vector<std::string>>
ByteFallbackDecoder::decode_chain(std::vector<std::string> tokens) const {
    std::vector<std::string> result;
    std::vector<uint8_t> pending_bytes;

    for (auto& token : tokens) {
        auto byte_val = parse_hex_byte(token);
        if (byte_val) {
            pending_bytes.push_back(*byte_val);
        } else {
            flush_bytes(pending_bytes, result);
            result.push_back(std::move(token));
        }
    }
    flush_bytes(pending_bytes, result);
    return result;
}

} // namespace decoders
} // namespace tokenizers
