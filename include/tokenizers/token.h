#pragma once
/// @file tokenizers/token.h
/// Token struct produced by Model::tokenize.

#include "tokenizers/common.h"
#include <string>

namespace tokenizers {

/// A single token produced by a Model.
struct Token {
    uint32_t id = 0;
    std::string value;
    Offsets offsets{0, 0};

    Token() = default;
    Token(uint32_t id, std::string value, Offsets offsets)
        : id(id), value(std::move(value)), offsets(std::move(offsets)) {}

    bool operator==(const Token&) const = default;
};

} // namespace tokenizers
