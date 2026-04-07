#include "tokenizers/models.h"
#include "trie.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tokenizers {
namespace models {

static constexpr double K_UNK_PENALTY = 10.0;

struct Unigram::Impl {
    std::vector<std::pair<std::string, double>> vocab;
    std::unordered_map<std::string, TokenId> token_to_ids;
    Trie trie;
    double min_score = 0.0;
    std::optional<size_t> unk_id;
    bool byte_fallback = false;
    bool fuse_unk = true;

    std::vector<std::string> encode_optimized(std::string_view sentence) const;
};

// Compute first UTF-8 character byte length
static size_t utf8_char_len(uint8_t byte) {
    if ((byte & 0x80) == 0) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 1;
}

std::vector<std::string> Unigram::Impl::encode_optimized(std::string_view sentence) const {
    if (sentence.empty()) return {};

    size_t size = sentence.size();
    double unk_score = min_score - K_UNK_PENALTY;

    struct BestPathNode {
        size_t id = 0;
        double best_path_score = 0.0;
        bool has_starts_at = false;
        size_t starts_at = 0;
    };

    std::vector<BestPathNode> best_path_ends_at(size + 1);

    size_t starts_at = 0;
    while (starts_at < size) {
        double best_score_here = best_path_ends_at[starts_at].best_path_score;
        size_t mblen = utf8_char_len(static_cast<uint8_t>(sentence[starts_at]));
        bool has_single_node = false;

        auto prefix_lengths = trie.common_prefix_search(sentence.substr(starts_at));
        for (size_t match_len : prefix_lengths) {
            size_t key_pos = starts_at + match_len;
            std::string token(sentence.substr(starts_at, match_len));
            auto it = token_to_ids.find(token);
            if (it == token_to_ids.end()) continue;
            TokenId id = it->second;
            double score = vocab[id].second;
            double candidate = score + best_score_here;

            auto& target = best_path_ends_at[key_pos];
            if (!target.has_starts_at || candidate > target.best_path_score) {
                target.best_path_score = candidate;
                target.starts_at = starts_at;
                target.has_starts_at = true;
                target.id = id;
            }
            if (!has_single_node && match_len == mblen) {
                has_single_node = true;
            }
        }

        if (!has_single_node) {
            auto& target = best_path_ends_at[starts_at + mblen];
            double candidate = unk_score + best_score_here;
            if (!target.has_starts_at || candidate > target.best_path_score) {
                target.best_path_score = candidate;
                target.starts_at = starts_at;
                target.has_starts_at = true;
                target.id = unk_id.value_or(0);
            }
        }
        starts_at += mblen;
    }

    // Backtrack
    size_t ends_at = size;
    std::vector<std::string> results;
    std::vector<std::string> unk_parts;

    while (ends_at > 0) {
        auto& node = best_path_ends_at[ends_at];
        size_t sa = node.starts_at;

        if (fuse_unk && unk_id.has_value() && node.id == *unk_id) {
            unk_parts.push_back(std::string(sentence.substr(sa, ends_at - sa)));
        } else {
            if (!unk_parts.empty()) {
                std::reverse(unk_parts.begin(), unk_parts.end());
                std::string fused;
                for (auto& p : unk_parts) fused += p;
                results.push_back(std::move(fused));
                unk_parts.clear();
            }
            results.push_back(std::string(sentence.substr(sa, ends_at - sa)));
        }
        ends_at = sa;
    }

    if (!unk_parts.empty()) {
        std::reverse(unk_parts.begin(), unk_parts.end());
        std::string fused;
        for (auto& p : unk_parts) fused += p;
        results.push_back(std::move(fused));
    }

    std::reverse(results.begin(), results.end());
    return results;
}

Unigram::Unigram(std::vector<std::pair<std::string, double>> vocab,
                 std::optional<size_t> unk_id, bool byte_fallback)
    : impl_(std::make_shared<Impl>()) {
    impl_->unk_id = unk_id;
    impl_->byte_fallback = byte_fallback;

    double min_score = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < vocab.size(); ++i) {
        impl_->token_to_ids[vocab[i].first] = static_cast<TokenId>(i);
        impl_->trie.push(vocab[i].first);
        if (vocab[i].second < min_score) {
            min_score = vocab[i].second;
        }
    }
    impl_->min_score = min_score;
    impl_->vocab = std::move(vocab);
}

Result<std::vector<Token>> Unigram::tokenize(std::string_view sequence) const {
    if (sequence.empty()) return std::vector<Token>{};

    auto str_tokens = impl_->encode_optimized(sequence);
    std::vector<Token> tokens;
    tokens.reserve(str_tokens.size());

    size_t offset = 0;
    for (auto& s : str_tokens) {
        size_t len = s.size();
        Offsets offsets{offset, offset + len};

        auto it = impl_->token_to_ids.find(s);
        if (it != impl_->token_to_ids.end()) {
            tokens.emplace_back(it->second, std::move(s), offsets);
        } else if (impl_->byte_fallback) {
            // Try byte fallback: convert each byte to <0xNN>
            bool all_found = true;
            std::vector<Token> byte_tokens;
            for (uint8_t byte : s) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", byte);
                std::string byte_str(buf);
                auto bt_it = impl_->token_to_ids.find(byte_str);
                if (bt_it == impl_->token_to_ids.end()) {
                    all_found = false;
                    break;
                }
                byte_tokens.emplace_back(bt_it->second, std::move(byte_str), offsets);
            }
            if (all_found) {
                for (auto& bt : byte_tokens) tokens.push_back(std::move(bt));
            } else if (impl_->unk_id.has_value()) {
                TokenId uid = static_cast<TokenId>(*impl_->unk_id);
                tokens.emplace_back(uid, std::move(s), offsets);
            } else {
                return make_error("Unknown token and no unk_id set");
            }
        } else if (impl_->unk_id.has_value()) {
            TokenId uid = static_cast<TokenId>(*impl_->unk_id);
            tokens.emplace_back(uid, std::move(s), offsets);
        } else {
            return make_error("Unknown token and no unk_id set");
        }
        offset += len;
    }
    return tokens;
}

std::optional<TokenId> Unigram::token_to_id(std::string_view token) const {
    auto it = impl_->token_to_ids.find(std::string(token));
    if (it != impl_->token_to_ids.end()) return it->second;
    return std::nullopt;
}

std::optional<std::string> Unigram::id_to_token(TokenId id) const {
    if (id < impl_->vocab.size()) return impl_->vocab[id].first;
    return std::nullopt;
}

std::unordered_map<std::string, TokenId> Unigram::get_vocab() const {
    return impl_->token_to_ids;
}

size_t Unigram::get_vocab_size() const {
    return impl_->vocab.size();
}

const std::vector<std::pair<std::string, double>>& Unigram::get_vocab_scores() const {
    return impl_->vocab;
}

std::optional<size_t> Unigram::get_unk_id() const {
    return impl_->unk_id;
}

bool Unigram::get_byte_fallback() const {
    return impl_->byte_fallback;
}

} // namespace models
} // namespace tokenizers
