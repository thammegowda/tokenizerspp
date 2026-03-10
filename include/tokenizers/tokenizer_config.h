#pragma once
/// @file tokenizers/tokenizer_config.h
/// Configuration from tokenizer_config.json (separate from tokenizer.json).

#include "tokenizers/common.h"
#include "tokenizers/error.h"

#include <optional>
#include <string>
#include <unordered_map>

namespace tokenizers {

/// Configuration from tokenizer_config.json (separate from tokenizer.json).
/// Contains special tokens, chat templates, and model-specific settings.
struct TokenizerConfig {
    std::optional<std::string> bos_token;
    std::optional<std::string> eos_token;
    std::optional<std::string> pad_token;
    std::optional<std::string> unk_token;
    bool add_bos_token = false;
    bool add_eos_token = false;

    // Chat template: either a single default template or named templates
    std::optional<std::string> default_chat_template;
    std::unordered_map<std::string, std::string> named_chat_templates;

    /// Parse from a JSON string (tokenizer_config.json content)
    static Result<TokenizerConfig> from_json(std::string_view json);

    /// Load from a file path
    static Result<TokenizerConfig> from_file(const std::string& path);

    /// Get a chat template by name ("default" returns default_chat_template)
    std::optional<std::string> get_chat_template(const std::string& name = "default") const;

    /// Whether any chat template is available
    bool has_chat_template() const;
};

} // namespace tokenizers
