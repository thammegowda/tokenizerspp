#include "tokenizers/processors.h"

namespace tokenizers::processors {

size_t ByteLevelProcessing::added_tokens(bool) const {
    return 0;
}

Result<std::vector<Encoding>> ByteLevelProcessing::process_encodings(
    std::vector<Encoding> encodings, bool) const {
    return encodings;
}

} // namespace tokenizers::processors