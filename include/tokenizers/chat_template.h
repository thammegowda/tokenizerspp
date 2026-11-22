#pragma once
/// @file tokenizers/chat_template.h
/// Chat template support using a minimal Jinja2 engine.

#include "tokenizers/common.h"
#include "tokenizers/error.h"

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace tokenizers {

/// A single message in a chat conversation.
struct ChatMessage {
    std::string role;
    std::string content;
};

/// Applies a Jinja2-style chat template to a list of messages.
class ChatTemplate {
public:
    /// Construct from a Jinja2 template string and optional special tokens.
    explicit ChatTemplate(const std::string& template_str,
                          std::optional<std::string> bos_token = std::nullopt,
                          std::optional<std::string> eos_token = std::nullopt);

    /// Apply the template to messages, returning the formatted string.
    ///
    /// `template_args` is a JSON object of extra variables placed in the render
    /// context, for templates that branch on model-specific runtime settings
    /// (`enable_thinking`, `tools`, ...). The set is per-model: it is whatever
    /// that model's template reads. Keys the engine supplies itself
    /// (`messages`, `add_generation_prompt`, `bos_token`, `eos_token`) are
    /// rejected rather than overridden.
    [[nodiscard]] Result<std::string> apply(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = false,
        const nlohmann::ordered_json& template_args = nlohmann::ordered_json::object()) const;

    /// Apply structured messages, including tool_calls, tool_call_id, name,
    /// reasoning_content, and content parts. Parse inputs as ordered_json to
    /// preserve key order; converting an already sorted json cannot restore it.
    [[nodiscard]] Result<std::string> apply_json(
        const nlohmann::ordered_json& messages_json,
        bool add_generation_prompt = false,
        const nlohmann::ordered_json& template_args = nlohmann::ordered_json::object()) const;

private:
    std::string template_str_;
    std::optional<std::string> bos_token_;
    std::optional<std::string> eos_token_;
};

} // namespace tokenizers
