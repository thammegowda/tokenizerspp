#include "tokenizers/decoders.h"

namespace tokenizers {
namespace decoders {

Result<std::vector<std::string>>
SequenceDecoder::decode_chain(std::vector<std::string> tokens) const {
    for (const auto& d : decoders) {
        auto result = d->decode_chain(std::move(tokens));
        if (!result) return result;
        tokens = std::move(*result);
    }
    return tokens;
}

} // namespace decoders
} // namespace tokenizers
