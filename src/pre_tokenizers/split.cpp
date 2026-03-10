#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/normalized_string.h"

namespace tokenizers {
namespace pre_tokenizers {

SplitPreTokenizer::SplitPreTokenizer(std::unique_ptr<Pattern> pat,
                                     SplitDelimiterBehavior behav, bool inv)
    : pattern(std::move(pat)), behavior(behav), invert(inv) {}

Result<void> SplitPreTokenizer::pre_tokenize(PreTokenizedString& pretokenized) const {
    if (invert) {
        auto inverted = std::make_unique<InvertPattern>(
            // We need to make a copy-like wrapper. Since Pattern is abstract,
            // we store a raw pointer reference and use it directly.
            // But InvertPattern takes ownership... We need a different approach.
            // Actually, the pattern is const and shared. Let's use a non-owning wrapper.
            std::unique_ptr<Pattern>(nullptr));
        // Use the pattern directly with inversion logic
        return pretokenized.split([&](size_t /*idx*/, NormalizedString normalized)
                                      -> Result<std::vector<NormalizedString>> {
            InvertPattern inv_pat(std::unique_ptr<Pattern>(nullptr));
            // We need to manually invert: find matches, flip is_match flags
            auto matches_result = pattern->find_matches(normalized.get());
            if (!matches_result) return std::unexpected(matches_result.error());

            // Invert the matches
            std::vector<PatternMatch> inverted_matches;
            for (auto& [offsets, is_match] : *matches_result) {
                inverted_matches.emplace_back(offsets, !is_match);
            }

            // Now apply split behavior manually using NormalizedString::split
            // We can create a temporary FuncPattern or use the inverted pattern.
            // Simplest: construct an InvertPattern wrapping our pattern... but we don't own it.
            // Let's just use a trick: call normalized.split with our pattern directly
            // but we need the invert. Let's use the NormalizedString split by leveraging
            // a custom pattern that inverts results.

            // Actually, we need a cleaner approach. Let's create a wrapper pattern
            // that references the original pattern.

            // Create a thin non-owning InvertPattern-like class
            struct RefInvertPattern : public Pattern {
                const Pattern* inner;
                explicit RefInvertPattern(const Pattern* p) : inner(p) {}
                Result<std::vector<PatternMatch>> find_matches(std::string_view inside) const override {
                    auto r = inner->find_matches(inside);
                    if (!r) return r;
                    for (auto& [o, m] : *r) m = !m;
                    return r;
                }
            };

            RefInvertPattern rip(pattern.get());
            return normalized.split(rip, behavior);
        });
    }

    return pretokenized.split([&](size_t /*idx*/, NormalizedString normalized)
                                  -> Result<std::vector<NormalizedString>> {
        return normalized.split(*pattern, behavior);
    });
}

} // namespace pre_tokenizers
} // namespace tokenizers
