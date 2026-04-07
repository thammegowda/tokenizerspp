#include "tokenizers/processors.h"

namespace tokenizers {
namespace processors {

namespace {

Encoding make_special_token_encoding(const std::string& token, TokenId id, TokenId type_id) {
    return Encoding(
        {id},
        {type_id},
        {token},
        {std::nullopt},
        {{0, 0}},
        {1},  // special_tokens_mask
        {1},  // attention_mask
        {},
        {}
    );
}

} // namespace

size_t BertProcessing::added_tokens(bool is_pair) const {
    return is_pair ? 3 : 2;  // CLS + SEP (+ SEP for pair)
}

Result<std::vector<Encoding>> BertProcessing::process_encodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {

    if (!add_special_tokens) {
        return encodings;
    }

    std::vector<Encoding> result;

    // CLS token (type_id 0)
    result.push_back(make_special_token_encoding(cls.first, cls.second, 0));

    // First encoding
    if (!encodings.empty()) {
        result.push_back(std::move(encodings[0]));
    }

    // SEP token after first encoding (type_id 0)
    result.push_back(make_special_token_encoding(sep.first, sep.second, 0));

    // If pair exists
    if (encodings.size() > 1) {
        result.push_back(std::move(encodings[1]));
        // SEP token after pair (type_id 1)
        result.push_back(make_special_token_encoding(sep.first, sep.second, 1));
    }

    return result;
}

} // namespace processors
} // namespace tokenizers
