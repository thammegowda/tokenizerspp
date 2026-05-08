#pragma once
/// @file tokenizers/tokenizer.h
/// Main Tokenizer class orchestrating the full pipeline.

#include "tokenizers/added_token.h"
#include "tokenizers/chat_template.h"
#include "tokenizers/common.h"
#include "tokenizers/decoder.h"
#include "tokenizers/encoding.h"
#include "tokenizers/error.h"
#include "tokenizers/model.h"
#include "tokenizers/normalizer.h"
#include "tokenizers/post_processor.h"
#include "tokenizers/pre_tokenizer.h"
#include "tokenizers/tokenizer_config.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace tokenizers {

/// Padding strategy.
enum class PaddingStrategy { BatchLongest, Fixed };
/// Padding direction.
enum class PaddingDirection { Left, Right };

/// Padding parameters.
struct PaddingParams {
    PaddingStrategy strategy = PaddingStrategy::BatchLongest;
    PaddingDirection direction = PaddingDirection::Right;
    size_t pad_to_multiple_of = 0;
    TokenId pad_id = 0;
    TokenId pad_type_id = 0;
    std::string pad_token = "[PAD]";
    size_t fixed_length = 0;
};

/// Truncation strategy.
enum class TruncationStrategy { LongestFirst, OnlyFirst, OnlySecond };
/// Truncation direction.
enum class TruncationDirection { Left, Right };

/// Truncation parameters.
struct TruncationParams {
    size_t max_length = 512;
    size_t stride = 0;
    TruncationStrategy strategy = TruncationStrategy::LongestFirst;
    TruncationDirection direction = TruncationDirection::Right;
};

/// The main Tokenizer class.
class Tokenizer {
public:
    Tokenizer();
    explicit Tokenizer(ModelPtr model);
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // Component setters (builder-style)
    Tokenizer& with_normalizer(NormalizerPtr normalizer);
    Tokenizer& with_pre_tokenizer(PreTokenizerPtr pre_tokenizer);
    Tokenizer& with_post_processor(PostProcessorPtr post_processor);
    Tokenizer& with_decoder(DecoderPtr decoder);
    Tokenizer& with_truncation(std::optional<TruncationParams> params);
    Tokenizer& with_padding(std::optional<PaddingParams> params);

    // Component getters
    [[nodiscard]] const Model* get_model() const { return model_.get(); }
    [[nodiscard]] const Normalizer* get_normalizer() const { return normalizer_.get(); }
    [[nodiscard]] const PreTokenizer* get_pre_tokenizer() const { return pre_tokenizer_.get(); }
    [[nodiscard]] const PostProcessor* get_post_processor() const { return post_processor_.get(); }
    [[nodiscard]] const Decoder* get_decoder() const { return decoder_.get(); }

    // Vocabulary
    [[nodiscard]] size_t get_vocab_size() const;
    [[nodiscard]] std::optional<TokenId> token_to_id(std::string_view token) const;
    [[nodiscard]] std::optional<std::string> id_to_token(TokenId id) const;

    // Add tokens
    size_t add_tokens(const std::vector<AddedToken>& tokens);
    size_t add_special_tokens(const std::vector<AddedToken>& tokens);

    // Encode
    [[nodiscard]] Result<Encoding> encode(std::string_view input, bool add_special_tokens) const;
    [[nodiscard]] Result<Encoding> encode_pair(std::string_view input, std::string_view pair,
                                                bool add_special_tokens) const;
    [[nodiscard]] Result<std::vector<Encoding>>
    encode_batch(const std::vector<std::string>& inputs, bool add_special_tokens) const;

    // Decode
    [[nodiscard]] Result<std::string> decode(const std::vector<TokenId>& ids,
                                              bool skip_special_tokens) const;
    [[nodiscard]] Result<std::vector<std::string>>
    decode_batch(const std::vector<std::vector<TokenId>>& batch_ids,
                 bool skip_special_tokens) const;

    // Serialization
    [[nodiscard]] static Result<Tokenizer> from_file(const std::string& path);
    [[nodiscard]] static Result<Tokenizer> from_string(std::string_view json);
    [[nodiscard]] static Result<Tokenizer> from_json(const nlohmann::json& j);
    [[nodiscard]] static Result<Tokenizer> from_directory(const std::string& dir);
    [[nodiscard]] Result<std::string> to_string(bool pretty = false) const;
    [[nodiscard]] Result<void> save(const std::string& path, bool pretty = true) const;

    // Config
    Tokenizer& with_config(TokenizerConfig config);
    [[nodiscard]] const TokenizerConfig* get_config() const;
    [[nodiscard]] std::string bos_token() const;
    [[nodiscard]] std::string eos_token() const;

    // Chat template
    [[nodiscard]] bool has_chat_template() const;
    [[nodiscard]] std::string chat_template_str(const std::string& name = "default") const;
    [[nodiscard]] Result<std::string> apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true,
        const std::string& template_name = "default") const;
    [[nodiscard]] Result<std::string> apply_chat_template(
        const std::string& template_str,
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true) const;
    /// Apply the chat template to pre-built JSON messages (e.g. structured
    /// multimodal content with `content[]` parts). Uses the same template
    /// cache as the flat-message overload.
    [[nodiscard]] Result<std::string> apply_chat_template_json(
        const nlohmann::json& messages_json,
        bool add_generation_prompt = true,
        const std::string& template_name = "default") const;
    [[nodiscard]] Result<Encoding> encode_chat(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true,
        bool add_special_tokens = true) const;

private:
    ModelPtr model_;
    NormalizerPtr normalizer_;
    PreTokenizerPtr pre_tokenizer_;
    PostProcessorPtr post_processor_;
    DecoderPtr decoder_;

    std::optional<TruncationParams> truncation_;
    std::optional<PaddingParams> padding_;

    // AddedVocabulary managed internally
    class AddedVocabulary;
    std::unique_ptr<AddedVocabulary> added_vocabulary_;

    std::optional<TokenizerConfig> config_;
    mutable std::unordered_map<std::string, ChatTemplate> template_cache_;
    mutable std::unique_ptr<std::mutex> template_cache_mutex_;  ///< protects lazy insertions into template_cache_

    // Internal encode helpers
    Result<Encoding> encode_single(std::string_view input, bool add_special_tokens) const;
    Result<Encoding> encode_segment(const std::string& text) const;
};

} // namespace tokenizers
