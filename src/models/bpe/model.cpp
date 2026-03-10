#include "tokenizers/models.h"
#include "tokenizers/error.h"
#include "models/bpe/word.h"

#include <cstdio>
#include <string>

namespace tokenizers {
namespace models {

namespace {

/// Byte length of a UTF-8 character starting at byte b.
size_t utf8_char_len(uint8_t b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    return 4;
}

} // namespace

BPE::BPE(std::unordered_map<std::string, uint32_t> vocab,
         MergeMap merges,
         std::optional<std::string> unk_token,
         std::optional<std::string> continuing_subword_prefix,
         std::optional<std::string> end_of_word_suffix,
         bool fuse_unk,
         bool byte_fallback,
         bool ignore_merges)
    : vocab_(std::move(vocab)),
      merges_(std::move(merges)),
      unk_token_(std::move(unk_token)),
      continuing_subword_prefix_(std::move(continuing_subword_prefix)),
      end_of_word_suffix_(std::move(end_of_word_suffix)),
      fuse_unk_(fuse_unk),
      byte_fallback_(byte_fallback),
      ignore_merges_(ignore_merges) {
    for (const auto& [token, id] : vocab_) {
        vocab_r_[id] = token;
    }
}

Result<std::vector<Token>> BPE::merge_word(std::string_view sequence) const {
    if (sequence.empty()) return std::vector<Token>{};

    // Check cache using string_view (no allocation on hit)
    if (sequence.size() < MAX_CACHE_WORD_LEN) {
        auto it = cache_.find(sequence);
        if (it != cache_.end()) {
            return it->second;
        }
    }

    auto result = merge_word_uncached(sequence);
    if (!result) return result;

    // Store in cache
    if (sequence.size() < MAX_CACHE_WORD_LEN) {
        cache_.emplace(std::string(sequence), *result);
    }
    return result;
}

Result<std::vector<Token>> BPE::merge_word_uncached(std::string_view sequence) const {
    if (sequence.empty()) {
        return std::vector<Token>{};
    }

    // If ignore_merges and the whole word is in vocab, return directly
    if (ignore_merges_) {
        auto it = vocab_.find(std::string(sequence));
        if (it != vocab_.end()) {
            return std::vector<Token>{
                {it->second, std::string(sequence), {0, sequence.size()}}};
        }
    }

    BPEWord word;
    bool has_unk = false;
    std::vector<Token> unk_tokens; // for fuse_unk accumulation

    // Build character-level tokens
    size_t char_idx = 0;
    size_t num_chars = 0;
    // Count chars first
    for (size_t i = 0; i < sequence.size(); ) {
        i += utf8_char_len(static_cast<uint8_t>(sequence[i]));
        ++num_chars;
    }

    size_t pos = 0;
    size_t idx = 0;
    while (pos < sequence.size()) {
        size_t clen = utf8_char_len(static_cast<uint8_t>(sequence[pos]));
        std::string ch(sequence.substr(pos, clen));
        bool is_first = (idx == 0);
        bool is_last = (idx == num_chars - 1);

        // Build the token string with prefix/suffix
        std::string token_str;
        if (!is_first && continuing_subword_prefix_) {
            token_str = *continuing_subword_prefix_ + ch;
        } else {
            token_str = ch;
        }
        if (is_last && end_of_word_suffix_) {
            token_str += *end_of_word_suffix_;
        }

        auto vit = vocab_.find(token_str);
        if (vit != vocab_.end()) {
            word.add(vit->second, static_cast<uint32_t>(clen));
        } else if (byte_fallback_) {
            // Use byte-level fallback tokens like <0xHH>
            bool all_found = true;
            std::vector<std::pair<uint32_t, uint32_t>> byte_tokens;
            for (size_t b = 0; b < clen; ++b) {
                char hex[7];
                std::snprintf(hex, sizeof(hex), "<0x%02X>",
                              static_cast<uint8_t>(sequence[pos + b]));
                auto bit = vocab_.find(hex);
                if (bit == vocab_.end()) {
                    all_found = false;
                    break;
                }
                byte_tokens.emplace_back(bit->second, 1);
            }
            if (all_found) {
                for (auto& [id, len] : byte_tokens) {
                    word.add(id, len);
                }
            } else if (unk_token_) {
                auto uit = vocab_.find(*unk_token_);
                if (uit != vocab_.end()) {
                    word.add(uit->second, static_cast<uint32_t>(clen));
                }
            }
        } else if (unk_token_) {
            auto uit = vocab_.find(*unk_token_);
            if (uit != vocab_.end()) {
                word.add(uit->second, static_cast<uint32_t>(clen));
            }
        }

        pos += clen;
        ++idx;
    }

    // Apply BPE merges
    word.merge_all(merges_);

    auto tokens = word.to_tokens(vocab_r_);

    // Fuse consecutive UNK tokens if needed
    if (fuse_unk_ && unk_token_) {
        auto uit = vocab_.find(*unk_token_);
        if (uit != vocab_.end()) {
            uint32_t unk_id = uit->second;
            std::vector<Token> fused;
            for (auto& tok : tokens) {
                if (tok.id == unk_id && !fused.empty() && fused.back().id == unk_id) {
                    fused.back().offsets.second = tok.offsets.second;
                } else {
                    fused.push_back(std::move(tok));
                }
            }
            return fused;
        }
    }

    return tokens;
}

Result<std::vector<Token>> BPE::tokenize(std::string_view sequence) const {
    return merge_word(sequence);
}

std::optional<uint32_t> BPE::token_to_id(std::string_view token) const {
    auto it = vocab_.find(std::string(token));
    if (it != vocab_.end()) return it->second;
    return std::nullopt;
}

std::optional<std::string> BPE::id_to_token(uint32_t id) const {
    auto it = vocab_r_.find(id);
    if (it != vocab_r_.end()) return it->second;
    return std::nullopt;
}

std::unordered_map<std::string, uint32_t> BPE::get_vocab() const {
    return vocab_;
}

size_t BPE::get_vocab_size() const {
    return vocab_.size();
}

} // namespace models
} // namespace tokenizers
