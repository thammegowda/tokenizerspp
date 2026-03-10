#pragma once
/// @file tokenizers/added_token.h
/// AddedToken struct and AddedVocabulary.

#include "tokenizers/common.h"
#include <cstdint>
#include <string>
#include <vector>

namespace tokenizers {

/// Represents a token added to the vocabulary (special or not).
struct AddedToken {
    std::string content;
    bool single_word = false;
    bool lstrip = false;
    bool rstrip = false;
    bool normalized = true;
    bool special = false;

    AddedToken() = default;
    explicit AddedToken(std::string content, bool special = false)
        : content(std::move(content)), special(special), normalized(!special) {}

    bool operator==(const AddedToken&) const = default;
};

} // namespace tokenizers
