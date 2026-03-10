#include "tokenizers/normalized_string.h"

#include <uni_algo/case.h>
#include <uni_algo/norm.h>
#include <uni_algo/prop.h>

#include <algorithm>
#include <cassert>

namespace tokenizers {

// ===== UTF-8 helpers =====

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

static size_t len_utf8(char32_t cp) {
    if (cp < 0x80) return 1;
    if (cp < 0x800) return 2;
    if (cp < 0x10000) return 3;
    return 4;
}

struct CharInfo { char32_t cp; size_t offset; size_t len; };

static std::vector<CharInfo> parse_utf8(std::string_view s) {
    std::vector<CharInfo> result;
    size_t pos = 0;
    while (pos < s.size()) {
        auto [cp, n] = decode_utf8(s.data() + pos, s.size() - pos);
        result.push_back({cp, pos, n});
        pos += n;
    }
    return result;
}

static std::string char_to_string(char32_t cp) {
    char buf[4];
    size_t n = encode_utf8(cp, buf);
    return {buf, n};
}

static bool is_unicode_whitespace(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

// ===== Constructor =====

NormalizedString::NormalizedString(std::string input)
    : original_(input), normalized_(std::move(input)), identity_alignments_(true) {
}

// ===== materialize_alignments =====

void NormalizedString::materialize_alignments() const {
    if (!identity_alignments_) return;
    identity_alignments_ = false;
    size_t pos = 0;
    alignments_.reserve(normalized_.size());
    while (pos < normalized_.size()) {
        auto [cp, n] = decode_utf8(normalized_.data() + pos, normalized_.size() - pos);
        for (size_t i = 0; i < n; i++)
            alignments_.push_back({pos, pos + n});
        pos += n;
    }
}

// ===== validate_range =====

std::optional<std::pair<size_t, size_t>>
NormalizedString::validate_range(Range range) const {
    auto is_char_boundary = [](std::string_view s, size_t pos) {
        return pos == 0 || pos >= s.size() ||
               (static_cast<uint8_t>(s[pos]) & 0xC0) != 0x80;
    };
    if (range.referential == OffsetReferential::Original) {
        size_t s = range.start, e = range.end;
        if (e > original_.size()) return std::nullopt;
        if (!is_char_boundary(original_, s) || !is_char_boundary(original_, e))
            return std::nullopt;
        return std::pair{s, e};
    } else {
        size_t s = range.start, e = range.end;
        if (e > normalized_.size()) return std::nullopt;
        if (!is_char_boundary(normalized_, s) || !is_char_boundary(normalized_, e))
            return std::nullopt;
        return std::pair{s, e};
    }
}

// ===== convert_offsets =====

std::optional<std::pair<size_t, size_t>>
NormalizedString::convert_offsets(Range range) const {
    bool original = (range.referential == OffsetReferential::Original);
    size_t target_start = range.start;
    size_t target_end = range.end;

    if (target_start == target_end) return std::pair{target_start, target_end};
    if (target_start > target_end) return std::nullopt;

    if (original && original_.empty() && target_start == 0 && target_end == 0)
        return std::pair{size_t(0), normalized_.size()};
    if (!original && normalized_.empty() && target_start == 0 && target_end == 0)
        return std::pair{size_t(0), original_.size()};

    // Fast path for identity alignments: offsets map 1:1
    if (identity_alignments_) {
        if (original) {
            if (target_end > original_.size()) return std::nullopt;
            return std::pair{target_start, target_end};
        } else {
            if (target_end > normalized_.size()) return std::nullopt;
            return std::pair{target_start, target_end};
        }
    }

    if (original) {
        std::optional<size_t> start, end;
        for (size_t i = 0; i < alignments_.size(); i++) {
            auto [a_start, a_end] = alignments_[i];
            if (target_end < a_end) break;
            if (!start && target_start <= a_start && a_start != a_end)
                start = i;
            if (target_end >= a_end)
                end = i + 1;
        }
        if (start && !end) return std::pair{*start, *start};
        if (!start && end) return std::pair{*end, *end};
        if (start && end) return std::pair{*start, *end};
        return std::nullopt;
    } else {
        if (target_start >= alignments_.size() || target_end > alignments_.size())
            return std::nullopt;
        size_t s = alignments_[target_start].first;
        size_t e = alignments_[target_end - 1].second;
        return std::pair{s, e};
    }
}

// ===== get_range / get_range_original =====

std::optional<std::string_view> NormalizedString::get_range(Range range) const {
    if (range.referential == OffsetReferential::Original) {
        auto conv = convert_offsets(range);
        if (!conv) return std::nullopt;
        auto [s, e] = *conv;
        if (e > normalized_.size()) return std::nullopt;
        return std::string_view(normalized_).substr(s, e - s);
    }
    size_t s = range.start, e = range.end;
    if (e > normalized_.size()) return std::nullopt;
    return std::string_view(normalized_).substr(s, e - s);
}

std::optional<std::string_view> NormalizedString::get_range_original(Range range) const {
    if (range.referential == OffsetReferential::Original) {
        size_t s = range.start, e = range.end;
        if (e > original_.size()) return std::nullopt;
        return std::string_view(original_).substr(s, e - s);
    }
    auto conv = convert_offsets(range);
    if (!conv) return std::nullopt;
    auto [s, e] = *conv;
    if (e > original_.size()) return std::nullopt;
    return std::string_view(original_).substr(s, e - s);
}

// ===== slice =====

std::optional<NormalizedString> NormalizedString::slice(Range range) const {
    auto valid = validate_range(range);
    if (!valid) return std::nullopt;

    // Fast path for identity alignments
    if (identity_alignments_) {
        auto [s, e] = *valid;
        NormalizedString result;
        result.original_ = normalized_.substr(s, e - s);
        result.normalized_ = result.original_;
        result.identity_alignments_ = true;
        result.original_shift_ = original_shift_ + s;
        return result;
    }

    std::pair<size_t, size_t> norm_range, orig_range;
    if (range.referential == OffsetReferential::Original) {
        orig_range = *valid;
        auto conv = convert_offsets(range);
        if (!conv) return std::nullopt;
        norm_range = *conv;
    } else {
        norm_range = *valid;
        auto conv = convert_offsets(range);
        if (!conv) return std::nullopt;
        orig_range = *conv;
    }

    size_t n_shift = orig_range.first;
    NormalizedString result;
    result.original_ = std::string(
        get_range_original(range).value_or(""));
    result.normalized_ = std::string(
        get_range(range).value_or(""));
    if (norm_range.first < norm_range.second &&
        norm_range.second <= alignments_.size()) {
        result.alignments_.reserve(norm_range.second - norm_range.first);
        for (size_t i = norm_range.first; i < norm_range.second; i++)
            result.alignments_.push_back(
                {alignments_[i].first - n_shift,
                 alignments_[i].second - n_shift});
    }
    result.original_shift_ = original_shift_ + orig_range.first;
    return result;
}

// ===== transform_range =====

void NormalizedString::transform_range(
    Range range,
    const std::vector<std::pair<char32_t, int>>& dest,
    size_t initial_offset) {

    materialize_alignments();
    identity_alignments_ = false;

    std::pair<size_t, size_t> n_range;
    if (range.referential == OffsetReferential::Normalized) {
        n_range = {range.start, std::min(range.end, normalized_.size())};
    } else {
        auto conv = convert_offsets(range);
        if (!conv) return;
        n_range = *conv;
    }

    std::string_view old_text(
        normalized_.data() + n_range.first,
        n_range.second - n_range.first);
    auto replaced_chars = parse_utf8(old_text);
    size_t ri = 0; // index into replaced_chars

    size_t initial_removed = 0;
    for (size_t i = 0; i < initial_offset && ri < replaced_chars.size(); i++, ri++)
        initial_removed += replaced_chars[ri].len;

    auto offset = static_cast<ptrdiff_t>(initial_removed + n_range.first);

    std::vector<std::pair<size_t, size_t>> new_alignments;
    new_alignments.reserve(dest.size());
    std::string new_normalized;

    for (auto [c, changes] : dest) {
        auto idx = static_cast<size_t>(offset);
        std::pair<size_t, size_t> align;
        if (changes > 0) {
            align = (idx < 1) ? std::pair<size_t,size_t>{0, 0}
                              : alignments_[idx - 1];
        } else {
            align = alignments_[idx];
        }

        size_t replaced_sz = 0;
        if (changes <= 0 && ri < replaced_chars.size()) {
            replaced_sz = replaced_chars[ri].len;
            ri++;
        }

        size_t removed_bytes = 0;
        if (changes < 0) {
            auto to_remove = static_cast<size_t>(-changes);
            for (size_t j = 0; j < to_remove && ri < replaced_chars.size(); j++, ri++)
                removed_bytes += replaced_chars[ri].len;
        }

        offset += static_cast<ptrdiff_t>(replaced_sz + removed_bytes);

        size_t clen = len_utf8(c);
        for (size_t j = 0; j < clen; j++)
            new_alignments.push_back(align);

        char buf[4];
        new_normalized.append(buf, encode_utf8(c, buf));
    }

    // Splice alignments
    auto ab = alignments_.begin();
    alignments_.erase(
        ab + static_cast<ptrdiff_t>(n_range.first),
        ab + static_cast<ptrdiff_t>(n_range.second));
    alignments_.insert(
        alignments_.begin() + static_cast<ptrdiff_t>(n_range.first),
        new_alignments.begin(), new_alignments.end());

    // Splice normalized string
    normalized_.replace(
        n_range.first, n_range.second - n_range.first, new_normalized);
}

void NormalizedString::transform(
    const std::vector<std::pair<char32_t, int>>& dest,
    size_t initial_offset) {
    transform_range(
        Range::original(0, original_.size()), dest, initial_offset);
}

// ===== byte_level_encode =====

void NormalizedString::byte_level_encode(const std::array<char32_t, 256>& b2c) {
    // Build new normalized string and alignments in one pass.
    // Each original byte maps through b2c to a char that's 1-2 bytes in UTF-8
    // (all GPT-2 byte-level chars are <= U+01FF, so max 2 UTF-8 bytes).
    auto sv = normalized_;
    std::string new_norm;
    new_norm.reserve(sv.size() * 2);
    std::vector<std::pair<size_t, size_t>> new_align;
    new_align.reserve(sv.size() * 2);

    // We need to know the original alignment for each byte.
    // If identity, each byte at position i aligns to the char boundary containing i.
    // If not identity, use existing alignments.

    if (identity_alignments_) {
        // Each normalized byte at position i maps to original range [char_start, char_end]
        size_t pos = 0;
        while (pos < sv.size()) {
            // Find UTF-8 char boundary
            auto b = static_cast<uint8_t>(sv[pos]);
            size_t char_len = 1;
            if ((b & 0x80) == 0) char_len = 1;
            else if ((b & 0xE0) == 0xC0) char_len = 2;
            else if ((b & 0xF0) == 0xE0) char_len = 3;
            else if ((b & 0xF8) == 0xF0) char_len = 4;
            if (pos + char_len > sv.size()) char_len = sv.size() - pos;

            std::pair<size_t, size_t> orig_align = {pos, pos + char_len};

            for (size_t j = 0; j < char_len; ++j) {
                char32_t mapped = b2c[static_cast<uint8_t>(sv[pos + j])];
                char buf[4];
                size_t n = encode_utf8(mapped, buf);
                new_norm.append(buf, n);
                for (size_t k = 0; k < n; ++k)
                    new_align.push_back(orig_align);
            }
            pos += char_len;
        }
    } else {
        // Use existing alignments
        materialize_alignments();
        size_t pos = 0;
        while (pos < sv.size()) {
            auto b = static_cast<uint8_t>(sv[pos]);
            size_t char_len = 1;
            if ((b & 0x80) == 0) char_len = 1;
            else if ((b & 0xE0) == 0xC0) char_len = 2;
            else if ((b & 0xF0) == 0xE0) char_len = 3;
            else if ((b & 0xF8) == 0xF0) char_len = 4;
            if (pos + char_len > sv.size()) char_len = sv.size() - pos;

            for (size_t j = 0; j < char_len; ++j) {
                // Use alignment of first byte of original char for all bytes
                auto align = alignments_[pos];
                char32_t mapped = b2c[static_cast<uint8_t>(sv[pos + j])];
                char buf[4];
                size_t n = encode_utf8(mapped, buf);
                new_norm.append(buf, n);
                for (size_t k = 0; k < n; ++k)
                    new_align.push_back(align);
            }
            pos += char_len;
        }
    }

    normalized_ = std::move(new_norm);
    alignments_ = std::move(new_align);
    identity_alignments_ = false;
}

// ===== Unicode normalization helpers =====

static bool is_non_starter(char32_t cp) {
    auto gc = una::codepoint::get_general_category(cp);
    return gc == una::codepoint::general_category::Mn ||
           gc == una::codepoint::general_category::Mc ||
           gc == una::codepoint::general_category::Me;
}

// Per-character normalization (for NFD/NFKD where decomposition is per-char)
static std::vector<std::pair<char32_t, int>> per_char_norm_changes(
    std::string_view old_str,
    std::string (*norm_fn)(std::string_view)) {
    std::vector<std::pair<char32_t, int>> result;
    auto old_chars = parse_utf8(old_str);
    for (auto& ci : old_chars) {
        auto normed = norm_fn(char_to_string(ci.cp));
        auto nc = parse_utf8(normed);
        if (nc.empty()) continue;
        result.push_back({nc[0].cp, 0});
        for (size_t i = 1; i < nc.size(); i++)
            result.push_back({nc[i].cp, 1});
    }
    return result;
}

// Segment-based normalization (for NFC/NFKC where composition can merge chars)
static std::vector<std::pair<char32_t, int>> segment_norm_changes(
    std::string_view old_str,
    std::string (*norm_fn)(std::string_view)) {
    std::vector<std::pair<char32_t, int>> result;
    auto old_chars = parse_utf8(old_str);
    if (old_chars.empty()) return result;

    size_t i = 0;
    while (i < old_chars.size()) {
        size_t seg_start = i++;
        while (i < old_chars.size() && is_non_starter(old_chars[i].cp))
            i++;
        size_t old_count = i - seg_start;

        std::string seg;
        for (size_t j = seg_start; j < i; j++)
            seg += char_to_string(old_chars[j].cp);

        auto normed = norm_fn(seg);
        auto nc = parse_utf8(normed);
        size_t new_count = nc.size();
        if (new_count == 0) continue;

        if (new_count <= old_count) {
            result.push_back({nc[0].cp, -static_cast<int>(old_count - new_count)});
            for (size_t j = 1; j < new_count; j++)
                result.push_back({nc[j].cp, 0});
        } else {
            for (size_t j = 0; j < old_count; j++)
                result.push_back({nc[j].cp, 0});
            for (size_t j = old_count; j < new_count; j++)
                result.push_back({nc[j].cp, 1});
        }
    }
    return result;
}

static std::string wrap_nfc(std::string_view s) { return una::norm::to_nfc_utf8(s); }
static std::string wrap_nfkc(std::string_view s) { return una::norm::to_nfkc_utf8(s); }
static std::string wrap_nfd(std::string_view s) { return una::norm::to_nfd_utf8(s); }
static std::string wrap_nfkd(std::string_view s) { return una::norm::to_nfkd_utf8(s); }

NormalizedString& NormalizedString::nfc() {
    transform(segment_norm_changes(normalized_, wrap_nfc), 0);
    return *this;
}
NormalizedString& NormalizedString::nfkc() {
    transform(segment_norm_changes(normalized_, wrap_nfkc), 0);
    return *this;
}
NormalizedString& NormalizedString::nfd() {
    transform(per_char_norm_changes(normalized_, wrap_nfd), 0);
    return *this;
}
NormalizedString& NormalizedString::nfkd() {
    transform(per_char_norm_changes(normalized_, wrap_nfkd), 0);
    return *this;
}

// ===== lowercase / uppercase =====

NormalizedString& NormalizedString::lowercase() {
    std::vector<std::pair<char32_t, int>> pairs;
    for (auto& ci : parse_utf8(normalized_)) {
        auto lower = una::cases::to_lowercase_utf8(char_to_string(ci.cp));
        auto lc = parse_utf8(lower);
        for (size_t i = 0; i < lc.size(); i++)
            pairs.push_back({lc[i].cp, i > 0 ? 1 : 0});
    }
    transform(pairs, 0);
    return *this;
}

NormalizedString& NormalizedString::uppercase() {
    std::vector<std::pair<char32_t, int>> pairs;
    for (auto& ci : parse_utf8(normalized_)) {
        auto upper = una::cases::to_uppercase_utf8(char_to_string(ci.cp));
        auto uc = parse_utf8(upper);
        for (size_t i = 0; i < uc.size(); i++)
            pairs.push_back({uc[i].cp, i > 0 ? 1 : 0});
    }
    transform(pairs, 0);
    return *this;
}

// ===== filter =====

NormalizedString& NormalizedString::filter(std::function<bool(char32_t)> keep) {
    int removed = 0;
    size_t removed_start = 0;
    std::vector<std::pair<char32_t, int>> transforms;
    std::optional<char32_t> last_c;

    for (auto& ci : parse_utf8(normalized_)) {
        if (keep(ci.cp)) {
            if (last_c) {
                transforms.push_back({*last_c, -removed});
            } else {
                removed_start = static_cast<size_t>(removed);
            }
            last_c = ci.cp;
            removed = 0;
        } else {
            removed++;
        }
    }
    if (last_c)
        transforms.push_back({*last_c, -removed});

    transform(transforms, removed_start);
    return *this;
}

// ===== prepend / append =====

NormalizedString& NormalizedString::prepend(std::string_view s) {
    auto chars = parse_utf8(normalized_);
    if (!chars.empty()) {
        char32_t next = chars[0].cp;
        auto sc = parse_utf8(s);
        std::vector<std::pair<char32_t, int>> t;
        for (size_t i = 0; i < sc.size(); i++)
            t.push_back({sc[i].cp, i != 0 ? 1 : 0});
        t.push_back({next, 1});
        transform_range(Range::normalized(0, len_utf8(next)), t, 0);
    }
    return *this;
}

NormalizedString& NormalizedString::append(std::string_view s) {
    auto chars = parse_utf8(normalized_);
    auto sc = parse_utf8(s);
    if (!chars.empty()) {
        auto& last = chars.back();
        std::vector<std::pair<char32_t, int>> t;
        t.push_back({last.cp, 0});
        for (auto& c : sc) t.push_back({c.cp, 1});
        transform_range(
            Range::normalized(last.offset, normalized_.size()), t, 0);
    } else {
        std::vector<std::pair<char32_t, int>> t;
        for (auto& c : sc) t.push_back({c.cp, 1});
        transform_range(Range::normalized(0, normalized_.size()), t, 0);
    }
    return *this;
}

// ===== map =====

NormalizedString& NormalizedString::map(std::function<char32_t(char32_t)> func) {
    std::vector<std::pair<char32_t, int>> t;
    for (auto& ci : parse_utf8(normalized_))
        t.push_back({func(ci.cp), 0});
    transform(t, 0);
    return *this;
}

// ===== replace =====

Result<void> NormalizedString::replace(
    const Pattern& pattern, std::string_view content) {
    materialize_alignments();
    identity_alignments_ = false;
    auto matches = pattern.find_matches(normalized_);
    if (!matches) return std::unexpected(matches.error());

    std::string new_norm;
    new_norm.reserve(normalized_.size());
    std::vector<std::pair<size_t, size_t>> new_align;
    new_align.reserve(alignments_.size());
    size_t last_end = 0;

    for (auto& [offsets, is_match] : *matches) {
        if (!is_match) continue;
        auto [start, end] = offsets;

        // Copy text before this match
        new_norm.append(normalized_, last_end, start - last_end);
        new_align.insert(new_align.end(),
            alignments_.begin() + static_cast<ptrdiff_t>(last_end),
            alignments_.begin() + static_cast<ptrdiff_t>(start));

        // Compute initial_removed: byte size of all chars in the matched range
        auto match_chars = parse_utf8(
            std::string_view(normalized_.data() + start, end - start));
        size_t initial_removed = 0;
        for (auto& mc : match_chars)
            initial_removed += mc.len;

        auto offset = static_cast<ptrdiff_t>(initial_removed + start);

        auto content_chars = parse_utf8(content);
        for (auto& cc : content_chars) {
            auto idx = static_cast<size_t>(offset);
            // All content chars are insertions (changes > 0)
            auto align = (idx < 1) ? std::pair<size_t,size_t>{0, 0}
                                   : alignments_[idx - 1];
            size_t clen = len_utf8(cc.cp);
            for (size_t j = 0; j < clen; j++)
                new_align.push_back(align);
            char buf[4];
            new_norm.append(buf, encode_utf8(cc.cp, buf));
        }
        last_end = end;
    }

    // Copy remaining
    new_norm.append(normalized_, last_end);
    new_align.insert(new_align.end(),
        alignments_.begin() + static_cast<ptrdiff_t>(last_end),
        alignments_.end());

    normalized_ = std::move(new_norm);
    alignments_ = std::move(new_align);
    return {};
}

// ===== clear =====

size_t NormalizedString::clear() {
    size_t l = len();
    transform({}, l);
    return l;
}

// ===== split =====

Result<std::vector<NormalizedString>>
NormalizedString::split(
    const Pattern& pattern, SplitDelimiterBehavior behavior) const {
    auto matches = pattern.find_matches(normalized_);
    if (!matches) return std::unexpected(matches.error());

    using SDB = SplitDelimiterBehavior;
    std::vector<std::pair<Offsets, bool>> splits;
    splits.reserve(matches->size());

    switch (behavior) {
    case SDB::Isolated:
        for (auto& [o, _] : *matches) splits.push_back({o, false});
        break;
    case SDB::Removed:
        splits = std::move(*matches);
        break;
    case SDB::Contiguous: {
        bool prev = false;
        for (auto& [o, m] : *matches) {
            if (m == prev) {
                if (!splits.empty()) splits.back().first.second = o.second;
                else splits.push_back({o, false});
            } else {
                splits.push_back({o, false});
            }
            prev = m;
        }
        break;
    }
    case SDB::MergedWithPrevious: {
        bool prev = false;
        for (auto& [o, m] : *matches) {
            if (m && !prev) {
                if (!splits.empty()) splits.back().first.second = o.second;
                else splits.push_back({o, false});
            } else {
                splits.push_back({o, false});
            }
            prev = m;
        }
        break;
    }
    case SDB::MergedWithNext: {
        bool prev = false;
        std::vector<std::pair<Offsets, bool>> rev;
        for (auto it = matches->rbegin(); it != matches->rend(); ++it) {
            auto& [o, m] = *it;
            if (m && !prev) {
                if (!rev.empty()) rev.back().first.first = o.first;
                else rev.push_back({o, false});
            } else {
                rev.push_back({o, false});
            }
            prev = m;
        }
        std::reverse(rev.begin(), rev.end());
        splits = std::move(rev);
        break;
    }
    }

    std::vector<NormalizedString> result;
    result.reserve(splits.size());
    for (auto& [o, remove] : splits) {
        if (!remove) {
            auto s = slice(Range::normalized(o.first, o.second));
            if (s) result.push_back(std::move(*s));
        }
    }
    return result;
}

// ===== lstrip / rstrip / strip =====

NormalizedString& NormalizedString::lstrip() { lrstrip(true, false); return *this; }
NormalizedString& NormalizedString::rstrip() { lrstrip(false, true); return *this; }
NormalizedString& NormalizedString::strip() { lrstrip(true, true); return *this; }

void NormalizedString::lrstrip(bool left, bool right) {
    auto chars = parse_utf8(normalized_);
    size_t count = chars.size();

    size_t leading = 0;
    if (left)
        for (auto& ci : chars) {
            if (!is_unicode_whitespace(ci.cp)) break;
            leading++;
        }

    size_t trailing = 0;
    if (right)
        for (auto it = chars.rbegin(); it != chars.rend(); ++it) {
            if (!is_unicode_whitespace(it->cp)) break;
            trailing++;
        }

    if (leading == 0 && trailing == 0) return;

    std::vector<std::pair<char32_t, int>> t;
    for (size_t i = 0; i < count; i++) {
        if (i < leading || i >= count - trailing)
            continue;
        // Match Rust: uses self.len() (bytes) in the comparison
        if (i == len() - trailing - 1)
            t.push_back({chars[i].cp, -static_cast<int>(trailing)});
        else
            t.push_back({chars[i].cp, 0});
    }
    transform(t, leading);
}

} // namespace tokenizers
