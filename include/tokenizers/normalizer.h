#pragma once
/// @file tokenizers/normalizer.h
/// Normalizer trait and SplitDelimiterBehavior enum.

#include "tokenizers/common.h"
#include "tokenizers/error.h"
#include <memory>

namespace tokenizers {

class NormalizedString;

/// Defines the expected behavior for the delimiter of a split pattern.
enum class SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
};

/// Base class for all normalizers.
class Normalizer {
public:
    virtual ~Normalizer() = default;
    [[nodiscard]] virtual Result<void> normalize(NormalizedString& normalized) const = 0;
};

using NormalizerPtr = std::unique_ptr<Normalizer>;

} // namespace tokenizers
