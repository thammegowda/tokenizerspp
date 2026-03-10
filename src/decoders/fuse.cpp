#include "tokenizers/decoders.h"

#include <string>
#include <vector>

namespace tokenizers {
namespace decoders {

Result<std::vector<std::string>>
FuseDecoder::decode_chain(std::vector<std::string> tokens) const {
    std::string joined;
    for (auto& t : tokens) joined += t;
    return std::vector<std::string>{std::move(joined)};
}

} // namespace decoders
} // namespace tokenizers
