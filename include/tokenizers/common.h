#pragma once
/// @file tokenizers/common.h
/// Core type aliases and forward declarations for the tokenizerpp library.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tokenizers {

/// Token ID type.  Signed so that -1 can serve as a "not found" sentinel,
/// matching the convention used by most C++ tokenizer consumers.
using TokenId = int32_t;

/// Byte offset pair (start, end) in original or normalized string.
using Offsets = std::pair<size_t, size_t>;

// Forward declarations
class Error;

/// Result type: either a value T or an Error.
template <typename T>
using Result = std::expected<T, Error>;

} // namespace tokenizers
