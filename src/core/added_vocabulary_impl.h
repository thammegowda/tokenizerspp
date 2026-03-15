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
            // Skip tokens that already exist in the base model — they're only
            // declared in added_tokens for special matching behavior, not new vocab.
            if (model && model->token_to_id(token.content).has_value()) {
                if (token.special) special_tokens_.insert(token.content);
                continue;
            }
            TokenId id = next_id(model);
            token_to_id_[token.content] = id;
            id_to_token_[id] = token.content;
            added_tokens_.push_back({token, id});
            if (token.special) {
                special_tokens_.insert(token.content);
            }
            ++added;
        }
        return added;
    }

    size_t add_special_tokens(const std::vector<AddedToken>& tokens, const Model* model) {
        size_t added = 0;
        for (const auto& token : tokens) {
            special_tokens_.insert(token.content);
            if (token_to_id_.contains(token.content)) continue;
            // Skip tokens that already exist in the base model
            if (model && model->token_to_id(token.content).has_value()) continue;
            TokenId id = next_id(model);
            token_to_id_[token.content] = id;
            id_to_token_[id] = token.content;
            AddedToken stored = token;
            stored.special = true;
            added_tokens_.push_back({stored, id});
            ++added;
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

    size_t size() const { return token_to_id_.size(); }

    struct AddedTokenWithId {
        AddedToken token;
        TokenId id;
    };

    const std::vector<AddedTokenWithId>& get_added_tokens() const {
        return added_tokens_;
    }

private:
    TokenId next_id(const Model* model) const {
        TokenId id = model ? static_cast<TokenId>(model->get_vocab_size()) : 0;
        return id + static_cast<TokenId>(token_to_id_.size());
    }

    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<TokenId, std::string> id_to_token_;
    std::unordered_set<std::string> special_tokens_;
    std::vector<AddedTokenWithId> added_tokens_;
};

} // namespace tokenizers
