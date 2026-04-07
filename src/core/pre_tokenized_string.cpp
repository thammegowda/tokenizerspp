#include "tokenizers/pre_tokenized_string.h"

#include <unordered_map>

namespace tokenizers {

PreTokenizedString::PreTokenizedString(const std::string& input)
    : original_(input) {
    NormalizedString ns(input);
    splits_.emplace_back(std::move(ns));
}

PreTokenizedString::PreTokenizedString(NormalizedString input)
    : original_(std::string(input.get_original())) {
    splits_.emplace_back(std::move(input));
}

Result<void> PreTokenizedString::split(SplitFn split_fn) {
    std::vector<Split> new_splits;
    new_splits.reserve(splits_.size());

    for (size_t i = 0; i < splits_.size(); ++i) {
        auto& original_split = splits_[i];
        // If already tokenized, keep as-is
        if (original_split.tokens.has_value()) {
            new_splits.push_back(std::move(original_split));
            continue;
        }

        auto result = split_fn(i, std::move(original_split.normalized));
        if (!result) return std::unexpected(result.error());

        for (auto& ns : *result) {
            if (!ns.is_empty()) {
                new_splits.emplace_back(std::move(ns));
            }
        }
    }
    splits_ = std::move(new_splits);
    return {};
}

Result<void> PreTokenizedString::normalize(NormalizeFn normalize_fn) {
    for (auto& split : splits_) {
        if (split.tokens.has_value()) continue;
        auto result = normalize_fn(split.normalized);
        if (!result) return std::unexpected(result.error());
    }
    return {};
}

Result<void> PreTokenizedString::tokenize(TokenizeFn tokenize_fn) {
    for (auto& split : splits_) {
        if (split.tokens.has_value()) continue;
        auto result = tokenize_fn(split.normalized);
        if (!result) return std::unexpected(result.error());
        split.tokens = std::move(*result);
    }
    return {};
}

// Helper: build byte→char offset map for converting offsets
struct BytesToCharConverter {
    std::unordered_map<size_t, size_t> map;

    explicit BytesToCharConverter(std::string_view seq) {
        size_t byte_pos = 0;
        size_t char_idx = 0;
        while (byte_pos < seq.size()) {
            auto b = static_cast<uint8_t>(seq[byte_pos]);
            size_t char_len = 1;
            if (b >= 0xF0) char_len = 4;
            else if (b >= 0xE0) char_len = 3;
            else if (b >= 0xC0) char_len = 2;

            for (size_t i = 0; i < char_len && byte_pos + i < seq.size(); ++i) {
                map[byte_pos + i] = char_idx;
            }
            byte_pos += char_len;
            ++char_idx;
        }
    }

    std::optional<Offsets> convert(Offsets offsets) const {
        auto start_it = map.find(offsets.first);
        auto end_it = map.find(offsets.second);
        if (start_it != map.end() && end_it != map.end()) {
            return Offsets{start_it->second, end_it->second};
        }
        if (start_it != map.end()) {
            // End is past the last byte — find the last char + 1
            auto last = map.find(offsets.second - 1);
            size_t end_char = last != map.end() ? last->second + 1 : start_it->second + 1;
            return Offsets{start_it->second, end_char};
        }
        return std::nullopt;
    }
};

Result<Encoding> PreTokenizedString::into_encoding(
    std::optional<TokenId> word_idx,
    TokenId type_id,
    OffsetType offset_type) const {

    if (splits_.empty()) return Encoding{};

    // Check all splits have tokens
    for (const auto& split : splits_) {
        if (!split.tokens.has_value()) {
            return make_error("Split has not been tokenized, call tokenize() first");
        }
    }

    std::optional<BytesToCharConverter> converter;
    if (offset_type == OffsetType::Char) {
        converter.emplace(original_);
    }

    std::vector<TokenId> ids;
    std::vector<TokenId> tids;
    std::vector<std::string> tokens;
    std::vector<std::optional<TokenId>> words;
    std::vector<Offsets> offsets;

    // Estimate total tokens for reservation
    size_t total_tokens = 0;
    for (const auto& split : splits_)
        total_tokens += split.tokens->size();
    ids.reserve(total_tokens);
    tids.reserve(total_tokens);
    tokens.reserve(total_tokens);
    words.reserve(total_tokens);
    offsets.reserve(total_tokens);

    for (size_t split_idx = 0; split_idx < splits_.size(); ++split_idx) {
        const auto& split = splits_[split_idx];
        const auto& ns = split.normalized;
        auto orig_offsets = ns.offsets_original();

        for (const auto& token : *split.tokens) {
            ids.push_back(token.id);
            tokens.push_back(token.value);

            auto token_offsets = ns.convert_offsets(
                Range::normalized(token.offsets.first, token.offsets.second));
            Offsets final_offsets;
            if (token_offsets) {
                final_offsets = {orig_offsets.first + token_offsets->first,
                                 orig_offsets.first + token_offsets->second};
            } else {
                final_offsets = token.offsets;
            }

            if (offset_type == OffsetType::None) {
                final_offsets = {0, 0};
            } else if (converter) {
                auto converted = converter->convert(final_offsets);
                if (converted) final_offsets = *converted;
            }

            offsets.push_back(final_offsets);
            words.push_back(word_idx.has_value() ? word_idx
                                                 : std::optional<TokenId>(static_cast<TokenId>(split_idx)));
            tids.push_back(type_id);
        }
    }

    size_t n = ids.size();
    return Encoding(
        std::move(ids),
        std::move(tids),
        std::move(tokens),
        std::move(words),
        std::move(offsets),
        std::vector<TokenId>(n, 0),  // special_tokens_mask
        std::vector<TokenId>(n, 1),  // attention_mask
        {},                            // overflowing
        {}                             // sequence_ranges
    );
}

std::vector<PreTokenizedString::SplitRef> PreTokenizedString::get_splits(
    OffsetReferential offset_ref, OffsetType offset_type) const {

    std::optional<BytesToCharConverter> converter;
    if (offset_type == OffsetType::Char) {
        converter.emplace(original_);
    }

    std::vector<SplitRef> result;
    size_t norm_offset = 0;

    for (const auto& split : splits_) {
        Offsets off;
        if (offset_ref == OffsetReferential::Original) {
            off = split.normalized.offsets_original();
        } else {
            size_t len = split.normalized.len();
            off = {norm_offset, norm_offset + len};
            norm_offset += len;
        }

        if (converter) {
            auto converted = converter->convert(off);
            if (converted) off = *converted;
        }

        result.push_back({split.normalized.get(), off, &split.tokens});
    }
    return result;
}

} // namespace tokenizers
