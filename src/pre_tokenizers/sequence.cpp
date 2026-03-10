#include "tokenizers/pre_tokenizers.h"

namespace tokenizers {
namespace pre_tokenizers {

Result<void> SequencePreTokenizer::pre_tokenize(PreTokenizedString& pretokenized) const {
    for (const auto& pt : pre_tokenizers) {
        auto result = pt->pre_tokenize(pretokenized);
        if (!result) return result;
    }
    return {};
}

} // namespace pre_tokenizers
} // namespace tokenizers
