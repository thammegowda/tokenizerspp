#include "tokenizers/normalizers.h"
#include "tokenizers/normalized_string.h"

namespace tokenizers {
namespace normalizers {

Result<void> PrecompiledNormalizer::normalize(NormalizedString& /*normalized*/) const {
    // Stub: precompiled charsmap normalization is complex. Pass through for now.
    return {};
}

} // namespace normalizers
} // namespace tokenizers
