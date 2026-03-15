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
        {1},
        {1},
        {},
        {}
    );
}

} // namespace

size_t RobertaProcessing::added_tokens(bool is_pair) const {
    return is_pair ? 4 : 2;  // <s> + </s> (+ </s> + </s> for pair)
}

Result<std::vector<Encoding>> RobertaProcessing::process_encodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {

    if (!add_special_tokens) {
        return encodings;
    }

    std::vector<Encoding> result;

    // CLS token
    result.push_back(make_special_token_encoding(cls.first, cls.second, 0));

    if (!encodings.empty()) {
        result.push_back(std::move(encodings[0]));
    }

    // SEP token after first
    result.push_back(make_special_token_encoding(sep.first, sep.second, 0));

    if (encodings.size() > 1) {
        // Extra SEP before pair
        result.push_back(make_special_token_encoding(sep.first, sep.second, 0));
        result.push_back(std::move(encodings[1]));
        // SEP after pair
        result.push_back(make_special_token_encoding(sep.first, sep.second, 0));
    }

    return result;
}

} // namespace processors
} // namespace tokenizers
