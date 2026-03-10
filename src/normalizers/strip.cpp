#include "tokenizers/normalizers.h"

#include <uni_algo/prop.h>

namespace tokenizers {
namespace normalizers {

Result<void> StripNormalizer::normalize(NormalizedString& normalized) const {
    if (strip_left && strip_right) {
        normalized.strip();
    } else {
        if (strip_left) normalized.lstrip();
        if (strip_right) normalized.rstrip();
    }
    return {};
}

Result<void> StripAccentsNormalizer::normalize(NormalizedString& normalized) const {
    normalized.filter([](char32_t c) {
        return !una::codepoint::prop(c).General_Category_Mn();
    });
    return {};
}

} // namespace normalizers
} // namespace tokenizers
