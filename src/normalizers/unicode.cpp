#include "tokenizers/normalizers.h"

namespace tokenizers {
namespace normalizers {

Result<void> NFDNormalizer::normalize(NormalizedString& normalized) const {
    normalized.nfd();
    return {};
}

Result<void> NFKDNormalizer::normalize(NormalizedString& normalized) const {
    normalized.nfkd();
    return {};
}

Result<void> NFCNormalizer::normalize(NormalizedString& normalized) const {
    normalized.nfc();
    return {};
}

Result<void> NFKCNormalizer::normalize(NormalizedString& normalized) const {
    normalized.nfkc();
    return {};
}

Result<void> NmtNormalizer::normalize(NormalizedString& normalized) const {
    normalized.filter([](char32_t c) {
        auto cp = static_cast<uint32_t>(c);
        return !((cp >= 0x0001 && cp <= 0x0008) ||
                 cp == 0x000B ||
                 (cp >= 0x000E && cp <= 0x001F) ||
                 cp == 0x007F ||
                 cp == 0x008F ||
                 cp == 0x009F);
    });
    normalized.map([](char32_t c) -> char32_t {
        auto cp = static_cast<uint32_t>(c);
        switch (cp) {
            case 0x0009: case 0x000A: case 0x000C: case 0x000D:
            case 0x1680: case 0x2028: case 0x2029: case 0x2581:
            case 0xFEFF: case 0xFFFD:
                return ' ';
            default:
                if (cp >= 0x200B && cp <= 0x200F) return ' ';
                return c;
        }
    });
    return {};
}

Result<void> LowercaseNormalizer::normalize(NormalizedString& normalized) const {
    normalized.lowercase();
    return {};
}

} // namespace normalizers
} // namespace tokenizers
