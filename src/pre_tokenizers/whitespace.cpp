#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/pattern.h"
#include "tokenizers/normalized_string.h"

namespace tokenizers {
namespace pre_tokenizers {

Result<void> WhitespaceSplit::pre_tokenize(PreTokenizedString& pretokenized) const {
    return pretokenized.split([](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
        FuncPattern ws_pat([](char32_t c) -> bool {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r';
        });
        return ns.split(ws_pat, SplitDelimiterBehavior::Removed);
    });
}

Result<void> Whitespace::pre_tokenize(PreTokenizedString& pretokenized) const {
    return pretokenized.split([](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
        static const InvertPattern& inv = *[]() {
            static auto p = std::make_unique<InvertPattern>(
                std::make_unique<RegexPattern>(R"(\w+|[^\w\s]+)"));
            return p.get();
        }();
        return ns.split(inv, SplitDelimiterBehavior::Removed);
    });
}

} // namespace pre_tokenizers
} // namespace tokenizers
