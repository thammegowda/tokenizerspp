#include "tokenizers/processors.h"

namespace tokenizers {
namespace processors {

TemplateProcessing::TemplateProcessing(std::vector<TemplatePiece> single_tmpl,
                                       std::vector<TemplatePiece> pair_tmpl,
                                       std::vector<SpecialTokenDef> special_tokens_vec)
    : single_template(std::move(single_tmpl)), pair_template(std::move(pair_tmpl)) {
    for (auto& st : special_tokens_vec) {
        std::string id = st.id;
        special_tokens.emplace(std::move(id), std::move(st));
    }
}

size_t TemplateProcessing::count_added(const std::vector<TemplatePiece>& tmpl) const {
    size_t count = 0;
    for (const auto& piece : tmpl) {
        if (piece.kind == TemplatePiece::SpecialToken) {
            auto it = special_tokens.find(piece.special_token);
            if (it != special_tokens.end()) {
                count += it->second.ids.size();
            }
        }
    }
    return count;
}

size_t TemplateProcessing::added_tokens(bool is_pair) const {
    return count_added(is_pair ? pair_template : single_template);
}

Result<std::vector<Encoding>> TemplateProcessing::process_encodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {

    const auto& tmpl = (encodings.size() >= 2) ? pair_template : single_template;

    std::vector<Encoding> result;
    for (const auto& piece : tmpl) {
        if (piece.kind == TemplatePiece::Sequence) {
            size_t idx = (piece.sequence == TemplateSequence::B) ? 1 : 0;
            if (idx < encodings.size()) {
                Encoding enc = encodings[idx];
                // Set type_ids to the piece's type_id
                std::vector<TokenId> type_ids(enc.len(), piece.type_id);
                enc.set_type_ids(std::move(type_ids));
                enc.set_sequence_id(idx);
                result.push_back(std::move(enc));
            }
        } else if (piece.kind == TemplatePiece::SpecialToken) {
            if (add_special_tokens) {
                auto it = special_tokens.find(piece.special_token);
                if (it != special_tokens.end()) {
                    const auto& st = it->second;
                    size_t len = st.ids.size();
                    Encoding enc(
                        st.ids,
                        std::vector<TokenId>(len, piece.type_id),
                        st.tokens,
                        std::vector<std::optional<TokenId>>(len, std::nullopt),
                        std::vector<Offsets>(len, {0, 0}),
                        std::vector<TokenId>(len, 1),  // special_tokens_mask
                        std::vector<TokenId>(len, 1),  // attention_mask
                        {},
                        {}
                    );
                    result.push_back(std::move(enc));
                }
            }
        }
    }
    return result;
}

} // namespace processors
} // namespace tokenizers
