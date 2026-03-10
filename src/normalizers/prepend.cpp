#include "tokenizers/normalizers.h"

namespace tokenizers {
namespace normalizers {

Result<void> PrependNormalizer::normalize(NormalizedString& normalized) const {
    normalized.prepend(prepend_str);
    return {};
}

} // namespace normalizers
} // namespace tokenizers
