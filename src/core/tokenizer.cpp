#include "tokenizers/tokenizer.h"
#include "tokenizers/chat_template.h"
#include "tokenizers/pre_tokenized_string.h"
#include "tokenizers/tokenizer_config.h"
#include "core/added_vocabulary_impl.h"

namespace tokenizers {

Tokenizer::Tokenizer(ModelPtr model)
    : model_(std::move(model)),
      added_vocabulary_(std::make_unique<AddedVocabulary>()) {}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer& Tokenizer::with_normalizer(NormalizerPtr normalizer) {
    normalizer_ = std::move(normalizer);
    return *this;
}

Tokenizer& Tokenizer::with_pre_tokenizer(PreTokenizerPtr pre_tokenizer) {
    pre_tokenizer_ = std::move(pre_tokenizer);
    return *this;
}

Tokenizer& Tokenizer::with_post_processor(PostProcessorPtr post_processor) {
    post_processor_ = std::move(post_processor);
    return *this;
}

Tokenizer& Tokenizer::with_decoder(DecoderPtr decoder) {
    decoder_ = std::move(decoder);
    return *this;
}

Tokenizer& Tokenizer::with_truncation(std::optional<TruncationParams> params) {
    truncation_ = std::move(params);
    return *this;
}

Tokenizer& Tokenizer::with_padding(std::optional<PaddingParams> params) {
    padding_ = std::move(params);
    return *this;
}

size_t Tokenizer::get_vocab_size() const {
    size_t size = model_ ? model_->get_vocab_size() : 0;
    if (added_vocabulary_) size += added_vocabulary_->size();
    return size;
}

std::optional<uint32_t> Tokenizer::token_to_id(std::string_view token) const {
    if (model_) {
        auto id = model_->token_to_id(token);
        if (id) return id;
    }
    if (added_vocabulary_) {
        return added_vocabulary_->token_to_id(std::string(token));
    }
    return std::nullopt;
}

std::optional<std::string> Tokenizer::id_to_token(uint32_t id) const {
    if (model_) {
        auto tok = model_->id_to_token(id);
        if (tok) return tok;
    }
    if (added_vocabulary_) {
        return added_vocabulary_->id_to_token(id);
    }
    return std::nullopt;
}

size_t Tokenizer::add_tokens(const std::vector<AddedToken>& tokens) {
    if (!added_vocabulary_) {
        added_vocabulary_ = std::make_unique<AddedVocabulary>();
    }
    return added_vocabulary_->add_tokens(tokens, model_.get());
}

size_t Tokenizer::add_special_tokens(const std::vector<AddedToken>& tokens) {
    if (!added_vocabulary_) {
        added_vocabulary_ = std::make_unique<AddedVocabulary>();
    }
    return added_vocabulary_->add_special_tokens(tokens, model_.get());
}

Result<Encoding> Tokenizer::encode_single(std::string_view input, bool add_special_tokens) const {
    if (!model_) {
        return make_error("No model set on Tokenizer");
    }

    // 1. Create PreTokenizedString
    PreTokenizedString pretokenized{std::string(input)};

    // 2. Normalize
    if (normalizer_) {
        auto result = pretokenized.normalize(
            [this](NormalizedString& ns) { return normalizer_->normalize(ns); });
        if (!result) return std::unexpected(result.error());
    }

    // 3. Pre-tokenize
    if (pre_tokenizer_) {
        auto result = pre_tokenizer_->pre_tokenize(pretokenized);
        if (!result) return std::unexpected(result.error());
    }

    // 4. Tokenize each split using the model
    auto tok_result = pretokenized.tokenize(
        [this](const NormalizedString& ns) -> Result<std::vector<Token>> {
            return model_->tokenize(ns.get());
        });
    if (!tok_result) return std::unexpected(tok_result.error());

    // 5. Build encoding
    auto encoding = pretokenized.into_encoding(std::nullopt, 0, OffsetType::Byte);
    if (!encoding) return std::unexpected(encoding.error());

    return std::move(*encoding);
}

Result<Encoding> Tokenizer::encode(std::string_view input, bool add_special_tokens) const {
    auto encoding = encode_single(input, add_special_tokens);
    if (!encoding) return std::unexpected(encoding.error());

    // Post-process
    if (post_processor_) {
        auto result = post_processor_->process(std::move(*encoding), std::nullopt, add_special_tokens);
        if (!result) return std::unexpected(result.error());
        encoding = std::move(*result);
    }

    // Truncation
    if (truncation_) {
        encoding->truncate(truncation_->max_length, truncation_->stride,
                           truncation_->direction == TruncationDirection::Right);
    }

    // Padding
    if (padding_) {
        size_t target = padding_->fixed_length;
        if (padding_->strategy == PaddingStrategy::BatchLongest) {
            target = encoding->len();
        }
        if (padding_->pad_to_multiple_of > 0) {
            size_t m = padding_->pad_to_multiple_of;
            target = ((target + m - 1) / m) * m;
        }
        if (target > 0) {
            encoding->pad(target, padding_->pad_id, padding_->pad_type_id,
                          padding_->pad_token,
                          padding_->direction == PaddingDirection::Right);
        }
    }

    return encoding;
}

Result<Encoding> Tokenizer::encode_pair(std::string_view input, std::string_view pair,
                                         bool add_special_tokens) const {
    auto enc1 = encode_single(input, add_special_tokens);
    if (!enc1) return std::unexpected(enc1.error());

    auto enc2 = encode_single(pair, add_special_tokens);
    if (!enc2) return std::unexpected(enc2.error());

    // Post-process with pair
    if (post_processor_) {
        auto result = post_processor_->process(std::move(*enc1), std::move(*enc2), add_special_tokens);
        if (!result) return std::unexpected(result.error());
        enc1 = std::move(*result);
    } else {
        enc1->merge_with(std::move(*enc2), false);
    }

    // Truncation
    if (truncation_) {
        enc1->truncate(truncation_->max_length, truncation_->stride,
                       truncation_->direction == TruncationDirection::Right);
    }

    // Padding
    if (padding_) {
        size_t target = padding_->fixed_length;
        if (padding_->strategy == PaddingStrategy::BatchLongest) {
            target = enc1->len();
        }
        if (padding_->pad_to_multiple_of > 0) {
            size_t m = padding_->pad_to_multiple_of;
            target = ((target + m - 1) / m) * m;
        }
        if (target > 0) {
            enc1->pad(target, padding_->pad_id, padding_->pad_type_id,
                      padding_->pad_token,
                      padding_->direction == PaddingDirection::Right);
        }
    }

    return enc1;
}

Result<std::vector<Encoding>> Tokenizer::encode_batch(
    const std::vector<std::string>& inputs, bool add_special_tokens) const {
    std::vector<Encoding> results;
    results.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto enc = encode(input, add_special_tokens);
        if (!enc) return std::unexpected(enc.error());
        results.push_back(std::move(*enc));
    }
    return results;
}

Result<std::string> Tokenizer::decode(const std::vector<uint32_t>& ids,
                                       bool skip_special_tokens) const {
    if (!model_) {
        return make_error("No model set on Tokenizer");
    }

    std::vector<std::string> tokens;
    tokens.reserve(ids.size());

    for (uint32_t id : ids) {
        auto tok = model_->id_to_token(id);
        if (!tok) {
            // Check added vocabulary
            if (added_vocabulary_) {
                tok = added_vocabulary_->id_to_token(id);
            }
            if (!tok) continue;
        }

        if (skip_special_tokens && added_vocabulary_ &&
            added_vocabulary_->is_special_token(*tok)) {
            continue;
        }

        tokens.push_back(std::move(*tok));
    }

    if (decoder_) {
        return decoder_->decode(std::move(tokens));
    }

    // Default: join with space
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += ' ';
        result += tokens[i];
    }
    return result;
}

Result<std::vector<std::string>> Tokenizer::decode_batch(
    const std::vector<std::vector<uint32_t>>& batch_ids,
    bool skip_special_tokens) const {
    std::vector<std::string> results;
    results.reserve(batch_ids.size());
    for (const auto& ids : batch_ids) {
        auto result = decode(ids, skip_special_tokens);
        if (!result) return std::unexpected(result.error());
        results.push_back(std::move(*result));
    }
    return results;
}

// Serialization: from_file, from_string, from_directory, to_string, save are in serialization.cpp

// ── Config & Chat Template ──────────────────────────────────────────────────

Tokenizer& Tokenizer::with_config(TokenizerConfig config) {
    config_ = std::move(config);
    return *this;
}

const TokenizerConfig* Tokenizer::get_config() const {
    return config_ ? &*config_ : nullptr;
}

std::string Tokenizer::bos_token() const {
    return config_ ? config_->bos_token.value_or("") : "";
}

std::string Tokenizer::eos_token() const {
    return config_ ? config_->eos_token.value_or("") : "";
}

bool Tokenizer::has_chat_template() const {
    return config_ && config_->has_chat_template();
}

std::string Tokenizer::chat_template_str(const std::string& name) const {
    if (!config_) return "";
    return config_->get_chat_template(name).value_or("");
}

Result<std::string> Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt,
    const std::string& template_name) const {
    auto tmpl_str = chat_template_str(template_name);
    if (tmpl_str.empty()) {
        return make_error("No chat template available");
    }
    return apply_chat_template(tmpl_str, messages, add_generation_prompt);
}

Result<std::string> Tokenizer::apply_chat_template(
    const std::string& template_str,
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt) const {
    auto it = template_cache_.find(template_str);
    if (it == template_cache_.end()) {
        std::string mutated = template_str;
        // Gemma3 hack: multimodal content access → plain content
        size_t pos;
        while ((pos = mutated.find("messages[0]['content'][0]['text']")) != std::string::npos)
            mutated.replace(pos, 32, "messages[0]['content']");
        // Qwen3 hack: Python slice → Jinja reverse filter
        while ((pos = mutated.find("[::-1]")) != std::string::npos)
            mutated.replace(pos, 6, "|reverse");
        // Strip generation markers
        while ((pos = mutated.find("{% generation %}")) != std::string::npos)
            mutated.erase(pos, 16);
        while ((pos = mutated.find("{% endgeneration %}")) != std::string::npos)
            mutated.erase(pos, 19);

        std::optional<std::string> bos = config_ ? config_->bos_token : std::nullopt;
        std::optional<std::string> eos = config_ ? config_->eos_token : std::nullopt;

        try {
            auto [inserted_it, _] = template_cache_.emplace(
                template_str, ChatTemplate(mutated, bos, eos));
            it = inserted_it;
        } catch (const std::exception& e) {
            return make_error(std::string("Chat template error: ") + e.what());
        }
    }

    return it->second.apply(messages, add_generation_prompt);
}

Result<Encoding> Tokenizer::encode_chat(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt,
    bool add_special_tokens) const {
    auto formatted = apply_chat_template(messages, add_generation_prompt);
    if (!formatted) return std::unexpected(formatted.error());
    return encode(*formatted, add_special_tokens);
}

} // namespace tokenizers
