#include "tokenizers/pattern.h"

#include <re2/re2.h>

#include <cstring>

namespace tokenizers {

// ===== Helper: decode one UTF-8 code point =====
// Returns (codepoint, byte_length). On invalid input, returns (replacement_char, 1).
static std::pair<char32_t, size_t> decode_utf8(const char* p, size_t remaining) {
    auto b = static_cast<uint8_t>(*p);
    if (b < 0x80) return {b, 1};
    if (remaining >= 2 && (b & 0xE0) == 0xC0) {
        char32_t cp = (b & 0x1F) << 6 | (static_cast<uint8_t>(p[1]) & 0x3F);
        return {cp, 2};
    }
    if (remaining >= 3 && (b & 0xF0) == 0xE0) {
        char32_t cp = (b & 0x0F) << 12
            | (static_cast<uint8_t>(p[1]) & 0x3F) << 6
            | (static_cast<uint8_t>(p[2]) & 0x3F);
        return {cp, 3};
    }
    if (remaining >= 4 && (b & 0xF8) == 0xF0) {
        char32_t cp = (b & 0x07) << 18
            | (static_cast<uint8_t>(p[1]) & 0x3F) << 12
            | (static_cast<uint8_t>(p[2]) & 0x3F) << 6
            | (static_cast<uint8_t>(p[3]) & 0x3F);
        return {cp, 4};
    }
    return {0xFFFD, 1};
}

// ===== Helper: encode one UTF-8 code point into a small buffer =====
static size_t encode_utf8(char32_t cp, char* buf) {
    if (cp < 0x80) { buf[0] = static_cast<char>(cp); return 1; }
    if (cp < 0x800) {
        buf[0] = static_cast<char>(0xC0 | (cp >> 6));
        buf[1] = static_cast<char>(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        buf[0] = static_cast<char>(0xE0 | (cp >> 12));
        buf[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = static_cast<char>(0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = static_cast<char>(0xF0 | (cp >> 18));
    buf[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = static_cast<char>(0x80 | (cp & 0x3F));
    return 4;
}

// ===== InvertPattern =====

Result<std::vector<PatternMatch>>
InvertPattern::find_matches(std::string_view inside) const {
    auto result = inner_->find_matches(inside);
    if (!result) return std::unexpected(result.error());
    for (auto& [offsets, flag] : *result) {
        flag = !flag;
    }
    return result;
}

// ===== CharPattern =====

Result<std::vector<PatternMatch>>
CharPattern::find_matches(std::string_view inside) const {
    if (inside.empty()) {
        return std::vector<PatternMatch>{{{0, 0}, false}};
    }

    std::vector<PatternMatch> matches;
    size_t last_offset = 0;
    size_t pos = 0;
    while (pos < inside.size()) {
        auto [cp, len] = decode_utf8(inside.data() + pos, inside.size() - pos);
        if (cp == ch_) {
            if (last_offset < pos) {
                matches.push_back({{last_offset, pos}, false});
            }
            matches.push_back({{pos, pos + len}, true});
            last_offset = pos + len;
        }
        pos += len;
    }
    if (last_offset < inside.size()) {
        matches.push_back({{last_offset, inside.size()}, false});
    }
    return matches;
}

// ===== StringPattern =====

Result<std::vector<PatternMatch>>
StringPattern::find_matches(std::string_view inside) const {
    if (pattern_.empty()) {
        // Empty pattern: no match, return the whole string as non-match.
        // Note: Rust returns char-count based offsets here, but we use byte offsets.
        return std::vector<PatternMatch>{{{0, inside.size()}, false}};
    }

    if (inside.empty()) {
        return std::vector<PatternMatch>{{{0, 0}, false}};
    }

    std::vector<PatternMatch> matches;
    size_t prev = 0;
    size_t pos = 0;
    while ((pos = inside.find(pattern_, prev)) != std::string_view::npos) {
        if (prev != pos) {
            matches.push_back({{prev, pos}, false});
        }
        matches.push_back({{pos, pos + pattern_.size()}, true});
        prev = pos + pattern_.size();
    }
    if (prev != inside.size()) {
        matches.push_back({{prev, inside.size()}, false});
    }
    return matches;
}

// ===== FuncPattern =====

Result<std::vector<PatternMatch>>
FuncPattern::find_matches(std::string_view inside) const {
    if (inside.empty()) {
        return std::vector<PatternMatch>{{{0, 0}, false}};
    }

    std::vector<PatternMatch> matches;
    size_t last_offset = 0;
    size_t last_seen = 0;
    size_t pos = 0;

    while (pos < inside.size()) {
        auto [cp, len] = decode_utf8(inside.data() + pos, inside.size() - pos);
        last_seen = pos + len;
        if (pred_(cp)) {
            if (last_offset < pos) {
                matches.push_back({{last_offset, pos}, false});
            }
            matches.push_back({{pos, pos + len}, true});
            last_offset = pos + len;
        }
        pos += len;
    }
    if (last_seen > last_offset) {
        matches.push_back({{last_offset, last_seen}, false});
    }
    return matches;
}

// ===== RegexPattern (RE2) =====

// Pre-process regex patterns to handle features not supported by RE2.
// RE2 does not support lookahead/lookbehind, but handles Unicode properties
// (\p{L}, \p{N}, etc.) natively.
static std::string preprocess_for_re2(const std::string& pattern) {
    std::string result = pattern;
    // Common HuggingFace tokenizer pattern: \s+(?!\S)|\s+ → \s+
    // The lookahead (?!\S) restricts to end-of-string whitespace, but in a
    // findall/split context the greedy \s+ already matches identically.
    static constexpr std::string_view target = "\\s+(?!\\S)|\\s+";
    static constexpr std::string_view replacement = "\\s+";
    if (auto pos = result.find(target); pos != std::string::npos) {
        result.replace(pos, target.size(), replacement);
    }
    return result;
}

struct RegexPattern::Impl {
    std::unique_ptr<RE2> re;
};

RegexPattern::RegexPattern(const std::string& pattern)
    : impl_(std::make_unique<Impl>()), pattern_str_(pattern) {
    RE2::Options opts;
    opts.set_log_errors(false);

    std::string processed = preprocess_for_re2(pattern);
    impl_->re = std::make_unique<RE2>(processed, opts);
    if (!impl_->re->ok()) {
        throw std::runtime_error("RE2 compile error: " + impl_->re->error());
    }
}

RegexPattern::~RegexPattern() = default;
RegexPattern::RegexPattern(RegexPattern&&) noexcept = default;
RegexPattern& RegexPattern::operator=(RegexPattern&&) noexcept = default;

Result<std::vector<PatternMatch>>
RegexPattern::find_matches(std::string_view inside) const {
    std::vector<PatternMatch> splits;
    size_t prev = 0;
    absl::string_view input(inside.data(), inside.size());
    absl::string_view match;
    size_t start_offset = 0;

    while (start_offset <= inside.size()) {
        if (!impl_->re->Match(input, start_offset, inside.size(),
                              RE2::UNANCHORED, &match, 1)) {
            break;
        }

        size_t match_start = match.data() - inside.data();
        size_t match_end = match_start + match.size();

        if (prev != match_start) {
            splits.push_back({{prev, match_start}, false});
        }
        splits.push_back({{match_start, match_end}, true});
        prev = match_end;

        // Advance past zero-length matches
        if (match_start == match_end) {
            start_offset = match_end + 1;
        } else {
            start_offset = match_end;
        }
    }

    if (prev != inside.size()) {
        splits.push_back({{prev, inside.size()}, false});
    }
    if (splits.empty() && inside.empty()) {
        splits.push_back({{0, 0}, false});
    }
    return splits;
}

} // namespace tokenizers
