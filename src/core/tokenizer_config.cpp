/// @file core/tokenizer_config.cpp
/// TokenizerConfig parsing from tokenizer_config.json.

#include "tokenizers/tokenizer_config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

namespace tokenizers {

using json = nlohmann::json;

/// Extract a token string from a JSON value that may be a string or an object
/// with a "content" field (HuggingFace format).
static std::optional<std::string> parse_token_value(const json& j, const std::string& key) {
    if (!j.contains(key) || j[key].is_null()) return std::nullopt;
    const auto& val = j[key];
    if (val.is_string()) return val.get<std::string>();
    if (val.is_object() && val.contains("content")) return val["content"].get<std::string>();
    return std::nullopt;
}

Result<TokenizerConfig> TokenizerConfig::from_json(std::string_view json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        return make_error(std::string("tokenizer_config.json parse error: ") + e.what());
    }

    TokenizerConfig config;

    config.bos_token = parse_token_value(j, "bos_token");
    config.eos_token = parse_token_value(j, "eos_token");
    config.pad_token = parse_token_value(j, "pad_token");
    config.unk_token = parse_token_value(j, "unk_token");

    if (j.contains("add_bos_token") && j["add_bos_token"].is_boolean()) {
        config.add_bos_token = j["add_bos_token"].get<bool>();
    }
    if (j.contains("add_eos_token") && j["add_eos_token"].is_boolean()) {
        config.add_eos_token = j["add_eos_token"].get<bool>();
    }

    // Chat template: string or array of {name, template}
    if (j.contains("chat_template") && !j["chat_template"].is_null()) {
        const auto& ct = j["chat_template"];
        if (ct.is_string()) {
            config.default_chat_template = ct.get<std::string>();
        } else if (ct.is_array()) {
            for (const auto& entry : ct) {
                if (!entry.contains("name") || !entry.contains("template")) continue;
                auto name = entry["name"].get<std::string>();
                auto tmpl = entry["template"].get<std::string>();
                config.named_chat_templates[name] = tmpl;
                if (name == "default") {
                    config.default_chat_template = tmpl;
                }
            }
        }
    }

    return config;
}

Result<TokenizerConfig> TokenizerConfig::from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return make_error("Cannot open file: " + path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return from_json(ss.str());
}

std::optional<std::string> TokenizerConfig::get_chat_template(const std::string& name) const {
    if (name == "default") {
        return default_chat_template;
    }
    auto it = named_chat_templates.find(name);
    if (it != named_chat_templates.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool TokenizerConfig::has_chat_template() const {
    return default_chat_template.has_value() || !named_chat_templates.empty();
}

} // namespace tokenizers
