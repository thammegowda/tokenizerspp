#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/pattern.h"
#include "tokenizers/normalized_string.h"

namespace tokenizers {
namespace pre_tokenizers {

Result<void> Digits::pre_tokenize(PreTokenizedString& pretokenized) const {
    bool individual = individual_digits;
    return pretokenized.split([individual](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
        FuncPattern digit_pat([](char32_t c) -> bool {
            return c >= '0' && c <= '9';
        });
        if (individual) {
            return ns.split(digit_pat, SplitDelimiterBehavior::Isolated);
        } else {
            return ns.split(digit_pat, SplitDelimiterBehavior::Contiguous);
        }
    });
}

} // namespace pre_tokenizers
} // namespace tokenizers
