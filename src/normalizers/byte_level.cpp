#include "tokenizers/normalizers.h"
#include "tokenizers/normalized_string.h"
#include "tokenizers/pre_tokenizers.h"

namespace tokenizers {
namespace normalizers {

Result<void> ByteLevelNormalizer::normalize(NormalizedString& normalized) const {
    if (normalized.is_empty()) return {};
    normalized.byte_level_encode(pre_tokenizers::ByteLevel::bytes_char_array());
    return {};
}

} // namespace normalizers
} // namespace tokenizers
