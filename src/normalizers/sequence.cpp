#include "tokenizers/normalizers.h"

namespace tokenizers {
namespace normalizers {

Result<void> SequenceNormalizer::normalize(NormalizedString& normalized) const {
    for (const auto& n : normalizers) {
        auto result = n->normalize(normalized);
        if (!result) return result;
    }
    return {};
}

} // namespace normalizers
} // namespace tokenizers
