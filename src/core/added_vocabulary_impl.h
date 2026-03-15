#pragma once
/// Private header: defines Tokenizer::AddedVocabulary inner class.

#include "tokenizers/tokenizer.h"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tokenizers {

class Tokenizer::AddedVocabulary {
public:
    size_t add_tokens(const std::vector<AddedToken>& tokens, const Model* model) {
        size_t added = 0;
        for (const auto& token : tokens) {
            if (token_to_id_.contains(token.content)) continue;
            // If the token exists in the base model, register it with the model's
            // own ID for special-token matching (pre-tokenization splitting).
            // Don't allocate a new ID — this keeps vocab_size correct.
            if (model) {
                if (auto mid = model->token_to_id(token.content)) {
                    token_to_id_[token.content] = *mid;
                    id_to_token_[*mid] = token.content;
                    added_tokens_.push_back({token, *mid});
                    if (token.special) special_tokens_.insert(token.content);
                    continue;
                }
            }
            TokenId id = next_id(model);
            token_to_id_[token.content] = id;
            id_to_token_[id] = token.content;
            added_tokens_.push_back({token, id});
            if (token.special) special_tokens_.insert(token.content);
            ++added;
            ++num_new_tokens_;
        }
        return added;
    }

    size_t add_special_tokens(const std::vector<AddedToken>& tokens, const Model* model) {
        size_t added = 0;
        for (const auto& token : tokens) {
            special_tokens_.insert(token.content);
            if (token_to_id_.contains(token.content)) continue;
            if (model) {
                if (auto mid = model->token_to_id(token.content)) {
                    token_to_id_[token.content] = *mid;
                    id_to_token_[*mid] = token.content;
                    AddedToken stored = token;
                    stored.special = true;
                    added_tokens_.push_back({stored, *mid});
                    continue;
                }
            }
            TokenId id = next_id(model);
            token_to_id_[token.content] = id;
            id_to_token_[id] = token.content;
            AddedToken stored = token;
            stored.special = true;
            added_tokens_.push_back({stored, id});
            ++added;
            ++num_new_tokens_;
        }
        return added;
    }

    bool is_special_token(const std::string& token) const {
        return special_tokens_.count(token) > 0;
    }

    std::optional<TokenId> token_to_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) return it->second;
        return std::nullopt;
    }

    std::optional<std::string> id_to_token(TokenId id) const {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) return it->second;
        return std::nullopt;
    }

    /// Number of genuinely new tokens (not already in the base model).
    size_t size() const { return num_new_tokens_; }

    struct AddedTokenWithId {
        AddedToken token;
        TokenId id;
    };

    const std::vector<AddedTokenWithId>& get_added_tokens() const {
        return added_tokens_;
    }

    /// Split input into segments: (text, is_added_token_id_or_nullopt).
    /// Added tokens are matched greedily (longest first) and returned with their ID.
    /// Non-matching segments are returned with nullopt.
    struct Segment {
        std::string text;
        std::optional<TokenId> id;  // set if this segment is an added token
    };

    std::vector<Segment> split_on_added_tokens(std::string_view input) const {
        if (added_tokens_.empty()) return {{std::string(input), std::nullopt}};

        std::vector<Segment> segments;
        size_t pos = 0;
        while (pos < input.size()) {
            // Try to match an added token at current position (longest match)
            size_t best_len = 0;
            TokenId best_id = -1;
            for (const auto& at : added_tokens_) {
                auto& content = at.token.content;
                if (content.size() > best_len &&
                    pos + content.size() <= input.size() &&
                    input.substr(pos, content.size()) == content) {
                    best_len = content.size();
                    best_id = at.id;
                }
            }
            if (best_len > 0) {
                // Flush pending text
                segments.push_back({std::string(input.substr(pos, best_len)), best_id});
                pos += best_len;
            } else {
                // Accumulate non-special text
                if (segments.empty() || segments.back().id.has_value()) {
                    segments.push_back({"", std::nullopt});
                }
                segments.back().text += input[pos];
                ++pos;
            }
        }
        return segments;
    }

private:
    TokenId next_id(const Model* model) const {
        TokenId id = model ? static_cast<TokenId>(model->get_vocab_size()) : 0;
        return id + static_cast<TokenId>(num_new_tokens_);
    }

    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<TokenId, std::string> id_to_token_;
    std::unordered_set<std::string> special_tokens_;
    std::vector<AddedTokenWithId> added_tokens_;
    size_t num_new_tokens_ = 0;  // tokens not in base model
};

} // namespace tokenizers
