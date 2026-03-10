#include "tokenizers/models.h"
#include "tokenizers/error.h"

namespace tokenizers {
namespace models {

WordLevel::WordLevel(std::unordered_map<std::string, uint32_t> vocab,
                     std::string unk_token)
    : vocab_(std::move(vocab)), unk_token_(std::move(unk_token)) {
    vocab_r_.reserve(vocab_.size());
    for (auto& [k, v] : vocab_) vocab_r_.emplace(v, k);
}

Result<std::vector<Token>> WordLevel::tokenize(std::string_view sequence) const {
    if (sequence.empty()) return std::vector<Token>{};

    auto it = vocab_.find(std::string(sequence));
    if (it != vocab_.end()) {
        return std::vector<Token>{
            Token{it->second, std::string(sequence), {0, sequence.size()}}};
    }
    // Look up UNK token
    auto unk_it = vocab_.find(unk_token_);
    if (unk_it != vocab_.end()) {
        return std::vector<Token>{
            Token{unk_it->second, unk_token_, {0, sequence.size()}}};
    }
    return make_error("WordLevel: unknown token and no UNK token in vocabulary");
}

std::optional<uint32_t> WordLevel::token_to_id(std::string_view token) const {
    auto it = vocab_.find(std::string(token));
    if (it != vocab_.end()) return it->second;
    return std::nullopt;
}

std::optional<std::string> WordLevel::id_to_token(uint32_t id) const {
    auto it = vocab_r_.find(id);
    if (it != vocab_r_.end()) return it->second;
    return std::nullopt;
}

std::unordered_map<std::string, uint32_t> WordLevel::get_vocab() const {
    return vocab_;
}

size_t WordLevel::get_vocab_size() const { return vocab_.size(); }

} // namespace models
} // namespace tokenizers
