#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/normalized_string.h"

namespace tokenizers {
namespace pre_tokenizers {

Result<void> FixedLength::pre_tokenize(PreTokenizedString& pretokenized) const {
    if (length == 0) {
        return make_error("FixedLength: length must be > 0");
    }

    return pretokenized.split(
        [&](size_t /*idx*/, NormalizedString normalized) -> Result<std::vector<NormalizedString>> {
            auto sv = normalized.get();
            if (sv.empty()) {
                return std::vector<NormalizedString>{std::move(normalized)};
            }

            std::vector<NormalizedString> chunks;
            size_t pos = 0;
            while (pos < sv.size()) {
                size_t end = std::min(pos + length, sv.size());
                auto slice = normalized.slice(
                    Range::normalized(pos, end));
                if (slice) {
                    chunks.push_back(std::move(*slice));
                }
                pos = end;
            }
            return chunks;
        });
}

} // namespace pre_tokenizers
} // namespace tokenizers
