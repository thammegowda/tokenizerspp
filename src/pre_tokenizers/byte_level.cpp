#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/pattern.h"
#include "tokenizers/normalized_string.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizers {
namespace pre_tokenizers {

// ── static bytes_char / char_bytes tables (flat arrays) ──────────────────────

struct ByteMaps {
    std::array<char32_t, 256> b2c;  // byte → char32_t
    std::array<uint8_t, 324> c2b;   // char32_t → byte (max codepoint is 256+67=323)
    std::array<bool, 324> c2b_valid; // whether c2b entry is a valid mapping
    size_t c2b_size;

    ByteMaps() {
        // Collect the "good" bytes that map to themselves
        std::vector<uint8_t> bs;
        bs.reserve(188);
        for (unsigned b = 0x21; b <= 0x7E; ++b) bs.push_back(static_cast<uint8_t>(b));
        for (unsigned b = 0xA1; b <= 0xAC; ++b) bs.push_back(static_cast<uint8_t>(b));
        for (unsigned b = 0xAE; b <= 0xFF; ++b) bs.push_back(static_cast<uint8_t>(b));

        std::vector<char32_t> cs;
        cs.reserve(256);
        for (auto b : bs) cs.push_back(static_cast<char32_t>(b));

        // Remaining bytes get mapped to 256+n
        uint32_t n = 0;
        for (uint32_t b = 0; b <= 255; ++b) {
            bool found = false;
            for (auto g : bs) {
                if (g == static_cast<uint8_t>(b)) { found = true; break; }
            }
            if (!found) {
                bs.push_back(static_cast<uint8_t>(b));
                cs.push_back(static_cast<char32_t>(256 + n));
                ++n;
            }
        }

        // Fill flat arrays
        b2c.fill(0);
        c2b_size = 256 + n;
        c2b.fill(0);
        c2b_valid.fill(false);
        for (size_t i = 0; i < bs.size(); ++i) {
            b2c[bs[i]] = cs[i];
            c2b[cs[i]] = bs[i];
            c2b_valid[cs[i]] = true;
        }
    }
};

static const ByteMaps& get_byte_maps() {
    static const ByteMaps maps;
    return maps;
}

const std::array<char32_t, 256>& ByteLevel::bytes_char_array() {
    return get_byte_maps().b2c;
}

const std::array<uint8_t, 324>& ByteLevel::char_bytes_array() {
    return get_byte_maps().c2b;
}

const std::array<bool, 324>& ByteLevel::char_bytes_valid() {
    return get_byte_maps().c2b_valid;
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Encode a char32_t code point to UTF-8 bytes appended to out.
static void encode_utf8(char32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Apply bytes_char mapping: for each byte of the input, replace with the
// corresponding unicode character from the GPT-2 mapping.
static std::string byte_level_encode(std::string_view input) {
    auto& b2c = ByteLevel::bytes_char_array();
    std::string result;
    result.reserve(input.size() * 2);
    for (unsigned char byte : input) {
        encode_utf8(b2c[byte], result);
    }
    return result;
}

// GPT-2 regex pattern
static const char* GPT2_REGEX =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

// ── ByteLevel implementation ────────────────────────────────────────────────

ByteLevel::ByteLevel(bool add_prefix_space, bool trim_offsets, bool use_regex)
    : add_prefix_space(add_prefix_space), trim_offsets(trim_offsets), use_regex(use_regex) {}

Result<void> ByteLevel::pre_tokenize(PreTokenizedString& pretokenized) const {
    // Step 1: optionally add prefix space
    if (add_prefix_space) {
        auto r = pretokenized.split(
            [](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
                auto sv = ns.get();
                if (sv.empty() || sv[0] != ' ') {
                    ns.prepend(" ");
                }
                return std::vector<NormalizedString>{std::move(ns)};
            });
        if (!r) return r;
    }

    // Step 2: optionally split using GPT-2 regex
    if (use_regex) {
        // Static compiled regex — avoids recompilation on every call
        static const RegexPattern gpt2_regex(GPT2_REGEX);
        static const InvertPattern gpt2_invert(
            std::make_unique<RegexPattern>(GPT2_REGEX));

        auto r = pretokenized.split(
            [](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
                static const InvertPattern& inv = *[]() {
                    static auto p = std::make_unique<InvertPattern>(
                        std::make_unique<RegexPattern>(GPT2_REGEX));
                    return p.get();
                }();
                return ns.split(inv, SplitDelimiterBehavior::Removed);
            });
        if (!r) return r;
    }

    // Step 3: apply byte-level encoding to each split
    return pretokenized.normalize(
        [](NormalizedString& ns) -> Result<void> {
            ns.byte_level_encode(ByteLevel::bytes_char_array());
            return {};
        });
}

} // namespace pre_tokenizers
} // namespace tokenizers
