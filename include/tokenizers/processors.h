#pragma once
/// @file tokenizers/processors.h
/// Concrete post-processor implementations.

#include "tokenizers/post_processor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokenizers {
namespace processors {

/// BERT post-processor: adds [CLS] at beginning and [SEP] at end.
class BertProcessing : public PostProcessor {
public:
    std::pair<std::string, uint32_t> sep;
    std::pair<std::string, uint32_t> cls;

    BertProcessing(std::pair<std::string, uint32_t> sep, std::pair<std::string, uint32_t> cls)
        : sep(std::move(sep)), cls(std::move(cls)) {}

    [[nodiscard]] size_t added_tokens(bool is_pair) const override;
    [[nodiscard]] Result<std::vector<Encoding>>
    process_encodings(std::vector<Encoding> encodings, bool add_special_tokens) const override;
};

/// RoBERTa post-processor: similar to BERT but with different type_id handling.
class RobertaProcessing : public PostProcessor {
public:
    std::pair<std::string, uint32_t> sep;
    std::pair<std::string, uint32_t> cls;
    bool trim_offsets = true;
    bool add_prefix_space = true;

    RobertaProcessing(std::pair<std::string, uint32_t> sep,
                      std::pair<std::string, uint32_t> cls,
                      bool trim_offsets = true, bool add_prefix_space = true)
        : sep(std::move(sep)), cls(std::move(cls)),
          trim_offsets(trim_offsets), add_prefix_space(add_prefix_space) {}

    [[nodiscard]] size_t added_tokens(bool is_pair) const override;
    [[nodiscard]] Result<std::vector<Encoding>>
    process_encodings(std::vector<Encoding> encodings, bool add_special_tokens) const override;
};

/// Sequence post-processor: chains multiple post-processors.
class SequenceProcessing : public PostProcessor {
public:
    std::vector<PostProcessorPtr> processors;

    explicit SequenceProcessing(std::vector<PostProcessorPtr> processors)
        : processors(std::move(processors)) {}

    [[nodiscard]] size_t added_tokens(bool is_pair) const override;
    [[nodiscard]] Result<std::vector<Encoding>>
    process_encodings(std::vector<Encoding> encodings, bool add_special_tokens) const override;
};

/// Template-based post-processor: applies templates with special tokens.
enum class TemplateSequence { A, B };

struct TemplatePiece {
    enum Kind { Sequence, SpecialToken } kind;
    TemplateSequence sequence = TemplateSequence::A;
    std::string special_token;
    uint32_t type_id = 0;
};

struct SpecialTokenDef {
    std::string id;
    uint32_t token_id = 0;
    std::vector<uint32_t> ids;
    std::vector<std::string> tokens;
};

class TemplateProcessing : public PostProcessor {
public:
    std::vector<TemplatePiece> single_template;
    std::vector<TemplatePiece> pair_template;
    std::unordered_map<std::string, SpecialTokenDef> special_tokens;

    TemplateProcessing() = default;
    TemplateProcessing(std::vector<TemplatePiece> single_tmpl,
                       std::vector<TemplatePiece> pair_tmpl,
                       std::vector<SpecialTokenDef> special_tokens);

    [[nodiscard]] size_t added_tokens(bool is_pair) const override;
    [[nodiscard]] Result<std::vector<Encoding>>
    process_encodings(std::vector<Encoding> encodings, bool add_special_tokens) const override;

private:
    size_t count_added(const std::vector<TemplatePiece>& tmpl) const;
};

} // namespace processors
} // namespace tokenizers
