#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/pattern.h"
#include "tokenizers/normalized_string.h"

namespace tokenizers {
namespace pre_tokenizers {

Result<void> CharDelimiterSplit::pre_tokenize(PreTokenizedString& pretokenized) const {
    char32_t d = delimiter;
    return pretokenized.split([d](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
        CharPattern pat(d);
        return ns.split(pat, SplitDelimiterBehavior::Removed);
    });
}

} // namespace pre_tokenizers
} // namespace tokenizers
