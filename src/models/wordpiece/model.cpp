#include "tokenizers/models.h"
#include "tokenizers/error.h"

#include <string>

namespace tokenizers {
namespace models {

namespace {

// Count UTF-8 characters in a string
size_t char_count(std::string_view s) {
    size_t count = 0;
    for (size_t i = 0; i < s.size(); ) {
        auto b = static_cast<uint8_t>(s[i]);
        if (b < 0x80) i += 1;
        else if ((b & 0xE0) == 0xC0) i += 2;
        else if ((b & 0xF0) == 0xE0) i += 3;
        else i += 4;
        ++count;
    }
    return count;
}

// Get the byte length of the last UTF-8 character ending at `end` in `s`
size_t last_char_len(std::string_view s, size_t end) {
    if (end == 0) return 0;
    size_t i = end - 1;
    while (i > 0 && (static_cast<uint8_t>(s[i]) & 0xC0) == 0x80) --i;
    return end - i;
}

} // namespace

WordPiece::WordPiece(std::unordered_map<std::string, uint32_t> vocab,
                     std::string unk_token,
                     std::string continuing_subword_prefix,
                     size_t max_input_chars_per_word)
    : vocab_(std::move(vocab)),
      unk_token_(std::move(unk_token)),
      continuing_subword_prefix_(std::move(continuing_subword_prefix)),
      max_input_chars_per_word_(max_input_chars_per_word) {
    for (const auto& [token, id] : vocab_) {
        vocab_r_[id] = token;
    }
}

Result<std::vector<Token>> WordPiece::tokenize(std::string_view sequence) const {
    size_t char_len = char_count(sequence);
    if (char_len > max_input_chars_per_word_) {
        auto it = vocab_.find(unk_token_);
        if (it == vocab_.end()) {
            return make_error("WordPiece: Missing [UNK] token from vocabulary");
        }
        return std::vector<Token>{{it->second, unk_token_, {0, sequence.size()}}};
    }

    std::vector<Token> sub_tokens;
    bool is_bad = false;
    size_t start = 0;

    while (start < sequence.size()) {
        size_t end = sequence.size();
        std::optional<Token> cur_str;

        while (start < end) {
            std::string substr;
            if (start > 0) {
                substr = continuing_subword_prefix_ + std::string(sequence.substr(start, end - start));
            } else {
                substr = std::string(sequence.substr(start, end - start));
            }

            auto it = vocab_.find(substr);
            if (it != vocab_.end()) {
                cur_str = Token{it->second, std::move(substr), {start, end}};
                break;
            }
            // Remove last character
            end -= last_char_len(sequence, end);
        }

        if (!cur_str) {
            is_bad = true;
            break;
        }

        sub_tokens.push_back(std::move(*cur_str));
        start = end;
    }

    if (is_bad) {
        auto it = vocab_.find(unk_token_);
        if (it == vocab_.end()) {
            return make_error("WordPiece: Missing [UNK] token from vocabulary");
        }
        return std::vector<Token>{{it->second, unk_token_, {0, sequence.size()}}};
    }

    return sub_tokens;
}

std::optional<uint32_t> WordPiece::token_to_id(std::string_view token) const {
    auto it = vocab_.find(std::string(token));
    if (it != vocab_.end()) return it->second;
    return std::nullopt;
}

std::optional<std::string> WordPiece::id_to_token(uint32_t id) const {
    auto it = vocab_r_.find(id);
    if (it != vocab_r_.end()) return it->second;
    return std::nullopt;
}

std::unordered_map<std::string, uint32_t> WordPiece::get_vocab() const {
    return vocab_;
}

size_t WordPiece::get_vocab_size() const {
    return vocab_.size();
}

} // namespace models
} // namespace tokenizers
