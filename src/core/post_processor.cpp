#include "tokenizers/post_processor.h"

namespace tokenizers {

Result<Encoding> PostProcessor::process(
    Encoding encoding,
    std::optional<Encoding> pair_encoding,
    bool add_special_tokens) const {

    // Set sequence IDs and type IDs
    encoding.set_sequence_id(0);

    std::vector<Encoding> encodings;
    encodings.push_back(std::move(encoding));

    if (pair_encoding) {
        pair_encoding->set_sequence_id(1);
        // Set type_id = 1 for the pair encoding
        std::vector<TokenId> type_ids(pair_encoding->len(), 1);
        pair_encoding->set_type_ids(std::move(type_ids));
        encodings.push_back(std::move(*pair_encoding));
    }

    auto result = process_encodings(std::move(encodings), add_special_tokens);
    if (!result) return std::unexpected(result.error());

    // Merge all returned encodings into one
    return Encoding::merge(std::move(*result), false);
}

} // namespace tokenizers
