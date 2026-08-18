#include "tokenizers/processors.h"

#include <stdexcept>

namespace tokenizers {
namespace processors {

SequenceProcessing::SequenceProcessing(std::vector<PostProcessorPtr> processors)
    : processors(std::move(processors)) {
    for (const auto& processor : this->processors) {
        if (!processor) {
            throw std::invalid_argument(
                "SequenceProcessing does not accept null processors");
        }
    }
}

size_t SequenceProcessing::added_tokens(bool is_pair) const {
    size_t total = 0;
    for (const auto& p : processors) {
        total += p->added_tokens(is_pair);
    }
    return total;
}

Result<std::vector<Encoding>> SequenceProcessing::process_encodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {

    for (const auto& processor : processors) {
        auto result = processor->process_encodings(std::move(encodings), add_special_tokens);
        if (!result) return std::unexpected(result.error());
        encodings = std::move(*result);
    }
    return encodings;
}

} // namespace processors
} // namespace tokenizers
