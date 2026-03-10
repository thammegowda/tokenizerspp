#pragma once
/// @file tokenizers/normalizers.h
/// All concrete normalizer implementations.

#include "tokenizers/normalizer.h"
#include "tokenizers/normalized_string.h"
#include "tokenizers/pattern.h"

#include <optional>
#include <string>
#include <vector>

namespace tokenizers {
namespace normalizers {

class BertNormalizer : public Normalizer {
public:
    bool clean_text = true;
    bool handle_chinese_chars = true;
    std::optional<bool> strip_accents;
    bool lowercase = true;

    BertNormalizer() = default;
    BertNormalizer(bool clean, bool chinese, std::optional<bool> accents, bool lower)
        : clean_text(clean), handle_chinese_chars(chinese),
          strip_accents(accents), lowercase(lower) {}

    Result<void> normalize(NormalizedString& normalized) const override;
};

class NFDNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

class NFKDNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

class NFCNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

class NFKCNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

class NmtNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

class LowercaseNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

class StripNormalizer : public Normalizer {
public:
    bool strip_left = true;
    bool strip_right = true;

    StripNormalizer() = default;
    StripNormalizer(bool left, bool right) : strip_left(left), strip_right(right) {}

    Result<void> normalize(NormalizedString& normalized) const override;
};

class StripAccentsNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

class ReplaceNormalizer : public Normalizer {
public:
    std::unique_ptr<Pattern> pattern;
    std::string content;

    ReplaceNormalizer(std::unique_ptr<Pattern> pattern, std::string content)
        : pattern(std::move(pattern)), content(std::move(content)) {}

    Result<void> normalize(NormalizedString& normalized) const override;
};

class PrependNormalizer : public Normalizer {
public:
    std::string prepend_str;

    explicit PrependNormalizer(std::string s) : prepend_str(std::move(s)) {}

    Result<void> normalize(NormalizedString& normalized) const override;
};

class SequenceNormalizer : public Normalizer {
public:
    std::vector<NormalizerPtr> normalizers;

    explicit SequenceNormalizer(std::vector<NormalizerPtr> normalizers)
        : normalizers(std::move(normalizers)) {}

    Result<void> normalize(NormalizedString& normalized) const override;
};

/// ByteLevel normalizer: converts each byte to GPT-2 byte-level character.
class ByteLevelNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

/// Precompiled normalizer: stub (passes through).
class PrecompiledNormalizer : public Normalizer {
public:
    Result<void> normalize(NormalizedString& normalized) const override;
};

} // namespace normalizers
} // namespace tokenizers
