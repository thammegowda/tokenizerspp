#include "tokenizers/normalizers.h"

namespace tokenizers {
namespace normalizers {

Result<void> ReplaceNormalizer::normalize(NormalizedString& normalized) const {
    return normalized.replace(*pattern, content);
}

} // namespace normalizers
} // namespace tokenizers
