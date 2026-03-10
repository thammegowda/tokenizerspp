#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/normalized_string.h"

namespace tokenizers {
namespace pre_tokenizers {

// Encode a char32_t to UTF-8
static std::string char32_to_utf8(char32_t ch) {
    std::string result;
    if (ch <= 0x7F) {
        result += static_cast<char>(ch);
    } else if (ch <= 0x7FF) {
        result += static_cast<char>(0xC0 | (ch >> 6));
        result += static_cast<char>(0x80 | (ch & 0x3F));
    } else if (ch <= 0xFFFF) {
        result += static_cast<char>(0xE0 | (ch >> 12));
        result += static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (ch & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (ch >> 18));
        result += static_cast<char>(0x80 | ((ch >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (ch & 0x3F));
    }
    return result;
}

Metaspace::Metaspace(char32_t repl, PrependScheme scheme, bool do_split)
    : replacement_char(repl), prepend_scheme(scheme), split(do_split) {
    replacement = char32_to_utf8(repl);
}

Result<void> Metaspace::pre_tokenize(PreTokenizedString& pretokenized) const {
    StringPattern space_pattern(" ");

    return pretokenized.split([&](size_t /*idx*/, NormalizedString normalized) -> Result<std::vector<NormalizedString>> {
        // Replace spaces with replacement char
        auto r = normalized.replace(space_pattern, replacement);
        if (!r) return std::unexpected(r.error());

        // Prepend replacement if needed
        if (prepend_scheme == PrependScheme::Always) {
            auto sv = normalized.get();
            if (sv.empty() || sv.substr(0, replacement.size()) != replacement) {
                normalized.prepend(replacement);
            }
        } else if (prepend_scheme == PrependScheme::First) {
            auto sv = normalized.get();
            auto orig_offsets = normalized.offsets_original();
            if (orig_offsets.first == 0 &&
                (sv.empty() || sv.substr(0, replacement.size()) != replacement)) {
                normalized.prepend(replacement);
            }
        }

        // Split on replacement char if requested
        if (split) {
            CharPattern char_pat(replacement_char);
            return normalized.split(char_pat, SplitDelimiterBehavior::MergedWithNext);
        } else {
            std::vector<NormalizedString> result;
            result.push_back(std::move(normalized));
            return result;
        }
    });
}

} // namespace pre_tokenizers
} // namespace tokenizers
