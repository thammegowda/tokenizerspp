#include "tokenizers/encoding.h"

#include <algorithm>
#include <numeric>

namespace tokenizers {

Encoding::Encoding(std::vector<TokenId> ids,
                   std::vector<TokenId> type_ids,
                   std::vector<std::string> tokens,
                   std::vector<std::optional<TokenId>> words,
                   std::vector<Offsets> offsets,
                   std::vector<TokenId> special_tokens_mask,
                   std::vector<TokenId> attention_mask,
                   std::vector<Encoding> overflowing,
                   std::unordered_map<size_t, std::pair<size_t, size_t>> sequence_ranges)
    : ids_(std::move(ids)),
      type_ids_(std::move(type_ids)),
      tokens_(std::move(tokens)),
      words_(std::move(words)),
      offsets_(std::move(offsets)),
      special_tokens_mask_(std::move(special_tokens_mask)),
      attention_mask_(std::move(attention_mask)),
      overflowing_(std::move(overflowing)),
      sequence_ranges_(std::move(sequence_ranges)) {}

Encoding Encoding::with_capacity(size_t len) {
    Encoding e;
    e.ids_.reserve(len);
    e.type_ids_.reserve(len);
    e.tokens_.reserve(len);
    e.words_.reserve(len);
    e.offsets_.reserve(len);
    e.special_tokens_mask_.reserve(len);
    e.attention_mask_.reserve(len);
    return e;
}

Encoding Encoding::from_tokens(const std::vector<Token>& tokens, TokenId type_id) {
    size_t n = tokens.size();
    Encoding e;
    e.ids_.reserve(n);
    e.tokens_.reserve(n);
    e.offsets_.reserve(n);
    for (const auto& t : tokens) {
        e.ids_.push_back(t.id);
        e.tokens_.push_back(t.value);
        e.offsets_.push_back(t.offsets);
    }
    e.words_.assign(n, std::nullopt);
    e.type_ids_.assign(n, type_id);
    e.attention_mask_.assign(n, 1);
    e.special_tokens_mask_.assign(n, 0);
    return e;
}

size_t Encoding::n_sequences() const {
    if (sequence_ranges_.empty()) return 1;
    return sequence_ranges_.size();
}

void Encoding::set_sequence_id(size_t id) {
    if (!ids_.empty()) {
        sequence_ranges_[id] = {0, ids_.size()};
    }
}

void Encoding::merge_with(Encoding other, bool growing_offsets) {
    size_t starting_offset = 0;
    if (growing_offsets && !offsets_.empty()) {
        starting_offset = offsets_.back().second;
    }

    // Update sequence ranges for the incoming encoding
    size_t offset = ids_.size();
    for (auto& [seq_id, range] : other.sequence_ranges_) {
        sequence_ranges_[seq_id] = {range.first + offset, range.second + offset};
    }

    // Merge all vectors
    ids_.insert(ids_.end(), other.ids_.begin(), other.ids_.end());
    type_ids_.insert(type_ids_.end(), other.type_ids_.begin(), other.type_ids_.end());
    tokens_.insert(tokens_.end(),
                   std::make_move_iterator(other.tokens_.begin()),
                   std::make_move_iterator(other.tokens_.end()));
    words_.insert(words_.end(), other.words_.begin(), other.words_.end());

    if (growing_offsets) {
        for (auto& o : other.offsets_) {
            offsets_.push_back({o.first + starting_offset, o.second + starting_offset});
        }
    } else {
        offsets_.insert(offsets_.end(), other.offsets_.begin(), other.offsets_.end());
    }

    special_tokens_mask_.insert(special_tokens_mask_.end(),
                                other.special_tokens_mask_.begin(),
                                other.special_tokens_mask_.end());
    attention_mask_.insert(attention_mask_.end(),
                           other.attention_mask_.begin(),
                           other.attention_mask_.end());

    // Merge overflowing
    overflowing_.insert(overflowing_.end(),
                        std::make_move_iterator(other.overflowing_.begin()),
                        std::make_move_iterator(other.overflowing_.end()));
}

Encoding Encoding::merge(std::vector<Encoding> encodings, bool growing_offsets) {
    if (encodings.empty()) return {};
    Encoding result = std::move(encodings[0]);
    for (size_t i = 1; i < encodings.size(); ++i) {
        result.merge_with(std::move(encodings[i]), growing_offsets);
    }
    return result;
}

void Encoding::truncate(size_t max_length, size_t stride, bool from_right) {
    if (ids_.size() <= max_length) return;

    // Generate overflow encoding if stride > 0
    if (stride > 0) {
        size_t keep = max_length;
        size_t start = from_right ? keep - stride : ids_.size() - keep + stride;
        // Simplified: just create the overflow from truncated portion
        // Full implementation would handle stride properly
    }

    if (from_right) {
        ids_.resize(max_length);
        type_ids_.resize(max_length);
        tokens_.resize(max_length);
        words_.resize(max_length);
        offsets_.resize(max_length);
        special_tokens_mask_.resize(max_length);
        attention_mask_.resize(max_length);
    } else {
        size_t remove = ids_.size() - max_length;
        ids_.erase(ids_.begin(), ids_.begin() + remove);
        type_ids_.erase(type_ids_.begin(), type_ids_.begin() + remove);
        tokens_.erase(tokens_.begin(), tokens_.begin() + remove);
        words_.erase(words_.begin(), words_.begin() + remove);
        offsets_.erase(offsets_.begin(), offsets_.begin() + remove);
        special_tokens_mask_.erase(special_tokens_mask_.begin(),
                                    special_tokens_mask_.begin() + remove);
        attention_mask_.erase(attention_mask_.begin(), attention_mask_.begin() + remove);
    }
}

void Encoding::pad(size_t target_length, TokenId pad_id, TokenId pad_type_id,
                   const std::string& pad_token, bool pad_right) {
    if (ids_.size() >= target_length) return;
    size_t pad_count = target_length - ids_.size();

    auto pad_ids = std::vector<TokenId>(pad_count, pad_id);
    auto pad_type_ids = std::vector<TokenId>(pad_count, pad_type_id);
    auto pad_tokens = std::vector<std::string>(pad_count, pad_token);
    auto pad_words = std::vector<std::optional<TokenId>>(pad_count, std::nullopt);
    auto pad_offsets = std::vector<Offsets>(pad_count, {0, 0});
    auto pad_special = std::vector<TokenId>(pad_count, 1);
    auto pad_attention = std::vector<TokenId>(pad_count, 0);

    if (pad_right) {
        ids_.insert(ids_.end(), pad_ids.begin(), pad_ids.end());
        type_ids_.insert(type_ids_.end(), pad_type_ids.begin(), pad_type_ids.end());
        tokens_.insert(tokens_.end(), pad_tokens.begin(), pad_tokens.end());
        words_.insert(words_.end(), pad_words.begin(), pad_words.end());
        offsets_.insert(offsets_.end(), pad_offsets.begin(), pad_offsets.end());
        special_tokens_mask_.insert(special_tokens_mask_.end(), pad_special.begin(), pad_special.end());
        attention_mask_.insert(attention_mask_.end(), pad_attention.begin(), pad_attention.end());
    } else {
        ids_.insert(ids_.begin(), pad_ids.begin(), pad_ids.end());
        type_ids_.insert(type_ids_.begin(), pad_type_ids.begin(), pad_type_ids.end());
        tokens_.insert(tokens_.begin(), pad_tokens.begin(), pad_tokens.end());
        words_.insert(words_.begin(), pad_words.begin(), pad_words.end());
        offsets_.insert(offsets_.begin(), pad_offsets.begin(), pad_offsets.end());
        special_tokens_mask_.insert(special_tokens_mask_.begin(), pad_special.begin(), pad_special.end());
        attention_mask_.insert(attention_mask_.begin(), pad_attention.begin(), pad_attention.end());
    }
}

} // namespace tokenizers
