#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/pattern.h"
#include "tokenizers/normalized_string.h"

#include <uni_algo/prop.h>

namespace tokenizers {
namespace pre_tokenizers {

namespace {

bool is_bert_punctuation(char32_t c) {
    // ASCII punctuation
    if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
        (c >= 91 && c <= 96) || (c >= 123 && c <= 126))
        return true;
    // Unicode punctuation
    auto p = una::codepoint::prop(c);
    return p.General_Category_P();
}

} // namespace

Result<void> BertPreTokenizer::pre_tokenize(PreTokenizedString& pretokenized) const {
    // First split on whitespace
    auto r1 = pretokenized.split([](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
        FuncPattern ws_pat([](char32_t c) -> bool {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                   una::codepoint::prop(c).White_Space();
        });
        return ns.split(ws_pat, SplitDelimiterBehavior::Removed);
    });
    if (!r1) return r1;

    // Then isolate punctuation
    return pretokenized.split([](size_t, NormalizedString ns) -> Result<std::vector<NormalizedString>> {
        FuncPattern punct_pat(is_bert_punctuation);
        return ns.split(punct_pat, SplitDelimiterBehavior::Isolated);
    });
}

} // namespace pre_tokenizers
} // namespace tokenizers
