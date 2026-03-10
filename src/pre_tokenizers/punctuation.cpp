#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/pattern.h"
#include "tokenizers/normalized_string.h"

#include <uni_algo/prop.h>

namespace tokenizers {
namespace pre_tokenizers {

Result<void> Punctuation::pre_tokenize(PreTokenizedString& pretokenized) const {
    return pretokenized.split([](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
        FuncPattern punct_pat([](char32_t c) -> bool {
            return una::codepoint::prop(c).General_Category_P();
        });
        return ns.split(punct_pat, SplitDelimiterBehavior::Isolated);
    });
}

} // namespace pre_tokenizers
} // namespace tokenizers
