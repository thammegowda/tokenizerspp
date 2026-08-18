#include "tokenizers/tokenizer.h"
#include "tokenizers/decoders.h"
#include "tokenizers/models.h"
#include "tokenizers/normalizers.h"
#include "tokenizers/pattern.h"
#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/processors.h"
#include "tokenizers/tokenizer_config.h"
#include "core/added_vocabulary_impl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace tokenizers {

using json = nlohmann::json;

// ── helpers ──────────────────────────────────────────────────────────────────

static bool is_null_or_missing(const json& j, const std::string& key) {
    return !j.contains(key) || j[key].is_null();
}

template <typename T>
static T get_or(const json& j, const std::string& key, T default_val) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<T>();
    return default_val;
}

template <typename T>
static std::optional<T> get_opt(const json& j, const std::string& key) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<T>();
    return std::nullopt;
}

// ── normalizer ───────────────────────────────────────────────────────────────

static Result<NormalizerPtr> parse_normalizer(const json& j);

static Result<NormalizerPtr> parse_normalizer_obj(const json& j) {
    std::string type = j.at("type").get<std::string>();

    if (type == "BertNormalizer") {
        auto n = std::make_unique<normalizers::BertNormalizer>();
        n->clean_text = get_or(j, "clean_text", true);
        n->handle_chinese_chars = get_or(j, "handle_chinese_chars", true);
        n->strip_accents = get_opt<bool>(j, "strip_accents");
        n->lowercase = get_or(j, "lowercase", true);
        return n;
    }
    if (type == "NFC") return std::make_unique<normalizers::NFCNormalizer>();
    if (type == "NFKC") return std::make_unique<normalizers::NFKCNormalizer>();
    if (type == "NFD") return std::make_unique<normalizers::NFDNormalizer>();
    if (type == "NFKD") return std::make_unique<normalizers::NFKDNormalizer>();
    if (type == "Nmt") return std::make_unique<normalizers::NmtNormalizer>();
    if (type == "Lowercase") return std::make_unique<normalizers::LowercaseNormalizer>();
    if (type == "StripAccents") return std::make_unique<normalizers::StripAccentsNormalizer>();
    if (type == "Strip") {
        bool left = get_or(j, "strip_left", true);
        bool right = get_or(j, "strip_right", true);
        return std::make_unique<normalizers::StripNormalizer>(left, right);
    }
    if (type == "Prepend") {
        std::string prepend = j.at("prepend").get<std::string>();
        return std::make_unique<normalizers::PrependNormalizer>(std::move(prepend));
    }
    if (type == "Replace") {
        std::string content = j.at("content").get<std::string>();
        auto& pat = j.at("pattern");
        std::unique_ptr<Pattern> pattern;
        if (pat.contains("String")) {
            pattern = std::make_unique<StringPattern>(pat["String"].get<std::string>());
        } else if (pat.contains("Regex")) {
            pattern = std::make_unique<RegexPattern>(pat["Regex"].get<std::string>());
        } else {
            return make_error("Replace: unsupported pattern type");
        }
        return std::make_unique<normalizers::ReplaceNormalizer>(std::move(pattern),
                                                                 std::move(content));
    }
    if (type == "Sequence") {
        std::vector<NormalizerPtr> children;
        for (auto& child : j.at("normalizers")) {
            auto r = parse_normalizer(child);
            if (!r) return std::unexpected(r.error());
            children.push_back(std::move(*r));
        }
        return std::make_unique<normalizers::SequenceNormalizer>(std::move(children));
    }
    if (type == "ByteLevel") {
        return std::make_unique<normalizers::ByteLevelNormalizer>();
    }
    if (type == "Precompiled") {
        return std::make_unique<normalizers::PrecompiledNormalizer>();
    }
    return make_error("Unknown normalizer type: " + type);
}

static Result<NormalizerPtr> parse_normalizer(const json& j) {
    if (j.is_null()) return NormalizerPtr{nullptr};
    return parse_normalizer_obj(j);
}

// ── pre_tokenizer ────────────────────────────────────────────────────────────

static Result<PreTokenizerPtr> parse_pre_tokenizer(const json& j);

static Result<PreTokenizerPtr> parse_pre_tokenizer_obj(const json& j) {
    std::string type = j.at("type").get<std::string>();

    if (type == "BertPreTokenizer")
        return std::make_unique<pre_tokenizers::BertPreTokenizer>();
    if (type == "Whitespace")
        return std::make_unique<pre_tokenizers::Whitespace>();
    if (type == "WhitespaceSplit")
        return std::make_unique<pre_tokenizers::WhitespaceSplit>();
    if (type == "Punctuation")
        return std::make_unique<pre_tokenizers::Punctuation>();
    if (type == "Digits") {
        bool individual = get_or(j, "individual_digits", false);
        return std::make_unique<pre_tokenizers::Digits>(individual);
    }
    if (type == "Sequence") {
        std::vector<PreTokenizerPtr> children;
        for (auto& child : j.at("pretokenizers")) {
            auto r = parse_pre_tokenizer(child);
            if (!r) return std::unexpected(r.error());
            children.push_back(std::move(*r));
        }
        return std::make_unique<pre_tokenizers::SequencePreTokenizer>(std::move(children));
    }
    if (type == "Metaspace") {
        // Parse replacement character
        std::string repl_str = get_or<std::string>(j, "replacement", "\xE2\x96\x81");
        // Decode first UTF-8 codepoint from repl_str
        char32_t repl_char = U'\u2581';
        if (!repl_str.empty()) {
            auto ch = static_cast<unsigned char>(repl_str[0]);
            if ((ch & 0x80) == 0) repl_char = ch;
            else {
                if ((ch & 0xE0) == 0xC0) repl_char = (ch & 0x1F);
                else if ((ch & 0xF0) == 0xE0) repl_char = (ch & 0x0F);
                else repl_char = (ch & 0x07);
                for (size_t i = 1; i < repl_str.size() && (static_cast<unsigned char>(repl_str[i]) & 0xC0) == 0x80; ++i)
                    repl_char = (repl_char << 6) | (static_cast<unsigned char>(repl_str[i]) & 0x3F);
            }
        }

        // Parse prepend_scheme
        auto scheme = pre_tokenizers::Metaspace::PrependScheme::Always;
        auto scheme_str = get_or<std::string>(j, "prepend_scheme", "always");
        if (scheme_str == "never") scheme = pre_tokenizers::Metaspace::PrependScheme::Never;
        else if (scheme_str == "first") scheme = pre_tokenizers::Metaspace::PrependScheme::First;

        // Handle legacy add_prefix_space
        if (j.contains("add_prefix_space") && !j["add_prefix_space"].is_null()) {
            bool prefix = j["add_prefix_space"].get<bool>();
            if (!prefix) scheme = pre_tokenizers::Metaspace::PrependScheme::Never;
        }

        bool do_split = get_or(j, "split", true);
        return std::make_unique<pre_tokenizers::Metaspace>(repl_char, scheme, do_split);
    }
    if (type == "Split") {
        // Parse pattern
        auto& pat = j.at("pattern");
        std::unique_ptr<Pattern> pattern;
        if (pat.contains("String")) {
            pattern = std::make_unique<StringPattern>(pat["String"].get<std::string>());
        } else if (pat.contains("Regex")) {
            pattern = std::make_unique<RegexPattern>(pat["Regex"].get<std::string>());
        } else {
            return make_error("Split: unsupported pattern type");
        }

        // Parse behavior
        auto behavior_str = j.at("behavior").get<std::string>();
        SplitDelimiterBehavior behavior = SplitDelimiterBehavior::Isolated;
        if (behavior_str == "Removed") behavior = SplitDelimiterBehavior::Removed;
        else if (behavior_str == "Isolated") behavior = SplitDelimiterBehavior::Isolated;
        else if (behavior_str == "MergedWithPrevious") behavior = SplitDelimiterBehavior::MergedWithPrevious;
        else if (behavior_str == "MergedWithNext") behavior = SplitDelimiterBehavior::MergedWithNext;
        else if (behavior_str == "Contiguous") behavior = SplitDelimiterBehavior::Contiguous;

        bool invert = get_or(j, "invert", false);
        return std::make_unique<pre_tokenizers::SplitPreTokenizer>(
            std::move(pattern), behavior, invert);
    }
    if (type == "ByteLevel") {
        bool prefix_space = get_or(j, "add_prefix_space", true);
        bool trim = get_or(j, "trim_offsets", true);
        bool regex = get_or(j, "use_regex", true);
        return std::make_unique<pre_tokenizers::ByteLevel>(prefix_space, trim, regex);
    }
    if (type == "UnicodeScripts") {
        return std::make_unique<pre_tokenizers::UnicodeScripts>();
    }
    if (type == "FixedLength") {
        size_t len = j.at("length").get<size_t>();
        return std::make_unique<pre_tokenizers::FixedLength>(len);
    }

    return make_error("Unknown pre-tokenizer type: " + type);
}

static Result<PreTokenizerPtr> parse_pre_tokenizer(const json& j) {
    if (j.is_null()) return PreTokenizerPtr{nullptr};
    return parse_pre_tokenizer_obj(j);
}

// ── post_processor ───────────────────────────────────────────────────────────

static Result<PostProcessorPtr> parse_post_processor(const json& j);

static std::pair<std::string, TokenId> parse_token_pair(const json& j) {
    return {j[0].get<std::string>(), j[1].get<TokenId>()};
}

static Result<PostProcessorPtr> parse_post_processor_obj(const json& j) {
    std::string type = j.at("type").get<std::string>();

    if (type == "BertProcessing") {
        auto sep = parse_token_pair(j.at("sep"));
        auto cls = parse_token_pair(j.at("cls"));
        return std::make_unique<processors::BertProcessing>(std::move(sep), std::move(cls));
    }
    if (type == "RobertaProcessing") {
        auto sep = parse_token_pair(j.at("sep"));
        auto cls = parse_token_pair(j.at("cls"));
        bool trim = get_or(j, "trim_offsets", true);
        bool prefix_space = get_or(j, "add_prefix_space", true);
        return std::make_unique<processors::RobertaProcessing>(std::move(sep), std::move(cls),
                                                                trim, prefix_space);
    }
    if (type == "Sequence") {
        std::vector<PostProcessorPtr> children;
        for (auto& child : j.at("processors")) {
            auto r = parse_post_processor(child);
            if (!r) return std::unexpected(r.error());
            if (!*r) {
                return make_error("Sequence post-processor contains a null child");
            }
            children.push_back(std::move(*r));
        }
        return std::make_unique<processors::SequenceProcessing>(std::move(children));
    }
    if (type == "TemplateProcessing") {
        // Parse template pieces
        auto parse_template = [](const json& arr) -> std::vector<processors::TemplatePiece> {
            std::vector<processors::TemplatePiece> pieces;
            for (auto& item : arr) {
                processors::TemplatePiece piece;
                if (item.contains("Sequence")) {
                    piece.kind = processors::TemplatePiece::Sequence;
                    auto id = item["Sequence"]["id"].get<std::string>();
                    piece.sequence = (id == "B") ? processors::TemplateSequence::B
                                                 : processors::TemplateSequence::A;
                    piece.type_id = get_or<TokenId>(item["Sequence"], "type_id", 0);
                } else if (item.contains("SpecialToken")) {
                    piece.kind = processors::TemplatePiece::SpecialToken;
                    piece.special_token = item["SpecialToken"]["id"].get<std::string>();
                    piece.type_id = get_or<TokenId>(item["SpecialToken"], "type_id", 0);
                }
                pieces.push_back(std::move(piece));
            }
            return pieces;
        };

        auto single_tmpl = parse_template(j.at("single"));
        auto pair_tmpl = parse_template(j.at("pair"));

        std::vector<processors::SpecialTokenDef> special_tokens_vec;
        if (j.contains("special_tokens") && !j["special_tokens"].is_null()) {
            for (auto& [key, val] : j["special_tokens"].items()) {
                processors::SpecialTokenDef st;
                st.id = val.at("id").get<std::string>();
                for (auto& id : val.at("ids")) st.ids.push_back(id.get<TokenId>());
                for (auto& tok : val.at("tokens")) st.tokens.push_back(tok.get<std::string>());
                if (!st.ids.empty()) st.token_id = st.ids[0];
                special_tokens_vec.push_back(std::move(st));
            }
        }
        return std::make_unique<processors::TemplateProcessing>(
            std::move(single_tmpl), std::move(pair_tmpl), std::move(special_tokens_vec));
    }
    if (type == "ByteLevel") {
        return std::make_unique<processors::ByteLevelProcessing>(
            get_or(j, "add_prefix_space", true),
            get_or(j, "trim_offsets", true),
            get_or(j, "use_regex", true));
    }
    return make_error("Unknown post-processor type: " + type);
}

static Result<PostProcessorPtr> parse_post_processor(const json& j) {
    if (j.is_null()) return PostProcessorPtr{nullptr};
    return parse_post_processor_obj(j);
}

// ── decoder ──────────────────────────────────────────────────────────────────

static Result<DecoderPtr> parse_decoder(const json& j);

static Result<DecoderPtr> parse_decoder_obj(const json& j) {
    std::string type = j.at("type").get<std::string>();

    if (type == "WordPiece") {
        std::string prefix = get_or<std::string>(j, "prefix", "##");
        bool cleanup = get_or(j, "cleanup", true);
        return std::make_unique<decoders::WordPieceDecoder>(std::move(prefix), cleanup);
    }
    if (type == "ByteLevel") {
        return std::make_unique<decoders::ByteLevelDecoder>();
    }
    if (type == "BPE") {
        std::string suffix = get_or<std::string>(j, "suffix", "</w>");
        return std::make_unique<decoders::BPEDecoder>(std::move(suffix));
    }
    if (type == "ByteFallback") {
        return std::make_unique<decoders::ByteFallbackDecoder>();
    }
    if (type == "CTC") {
        std::string pad = get_or<std::string>(j, "pad_token", "<pad>");
        std::string delim = get_or<std::string>(j, "word_delimiter_token", "|");
        bool cleanup = get_or(j, "cleanup", true);
        return std::make_unique<decoders::CTCDecoder>(std::move(pad), std::move(delim), cleanup);
    }
    if (type == "Fuse") {
        return std::make_unique<decoders::FuseDecoder>();
    }
    if (type == "Strip") {
        std::string content_str = get_or<std::string>(j, "content", " ");
        char32_t content_char = ' ';
        if (!content_str.empty()) {
            // Decode first UTF-8 code point
            auto ch = static_cast<unsigned char>(content_str[0]);
            if ((ch & 0x80) == 0) content_char = ch;
            else if ((ch & 0xE0) == 0xC0) content_char = (ch & 0x1F);
            else if ((ch & 0xF0) == 0xE0) content_char = (ch & 0x0F);
            else content_char = (ch & 0x07);
            for (size_t i = 1; i < content_str.size() && (static_cast<unsigned char>(content_str[i]) & 0xC0) == 0x80; ++i)
                content_char = (content_char << 6) | (static_cast<unsigned char>(content_str[i]) & 0x3F);
        }
        size_t start = get_or<size_t>(j, "start", 0);
        size_t stop = get_or<size_t>(j, "stop", 0);
        return std::make_unique<decoders::StripDecoder>(content_char, start, stop);
    }
    if (type == "Sequence") {
        std::vector<DecoderPtr> children;
        for (auto& child : j.at("decoders")) {
            auto r = parse_decoder(child);
            if (!r) return std::unexpected(r.error());
            children.push_back(std::move(*r));
        }
        return std::make_unique<decoders::SequenceDecoder>(std::move(children));
    }
    if (type == "Metaspace") {
        std::string repl = get_or<std::string>(j, "replacement", "\xE2\x96\x81");
        auto scheme_str = get_or<std::string>(j, "prepend_scheme", "always");
        bool prepend_always = (scheme_str == "always");
        // Handle legacy add_prefix_space
        if (j.contains("add_prefix_space") && !j["add_prefix_space"].is_null()) {
            prepend_always = j["add_prefix_space"].get<bool>();
        }
        return std::make_unique<decoders::MetaspaceDecoder>(std::move(repl), prepend_always);
    }
    if (type == "Replace") {
        // Pattern can be {"String":"..."} or a plain string
        std::string pat;
        if (j.contains("pattern")) {
            auto& p = j["pattern"];
            if (p.is_object() && p.contains("String"))
                pat = p["String"].get<std::string>();
            else if (p.is_string())
                pat = p.get<std::string>();
        }
        std::string content = get_or<std::string>(j, "content", "");
        return std::make_unique<decoders::ReplaceDecoder>(std::move(pat), std::move(content));
    }
    return make_error("Unknown decoder type: " + type);
}

static Result<DecoderPtr> parse_decoder(const json& j) {
    if (j.is_null()) return DecoderPtr{nullptr};
    return parse_decoder_obj(j);
}

// ── model ────────────────────────────────────────────────────────────────────

static std::unordered_map<std::string, TokenId> parse_vocab(const json& j) {
    std::unordered_map<std::string, TokenId> vocab;
    vocab.reserve(j.size());
    for (auto& [key, val] : j.items()) {
        vocab.emplace(key, val.get<TokenId>());
    }
    return vocab;
}

static Result<ModelPtr> parse_model(const json& j) {
    // Handle missing "type" field — infer from available keys
    std::string type;
    if (j.contains("type") && !j["type"].is_null()) {
        type = j["type"].get<std::string>();
    } else if (j.contains("merges")) {
        type = "BPE";
    } else if (j.contains("continuing_subword_prefix")) {
        type = "WordPiece";
    } else if (j.contains("vocab") && j["vocab"].is_array()) {
        type = "Unigram";
    } else {
        type = "BPE"; // default fallback
    }

    if (type == "WordPiece") {
        auto vocab = parse_vocab(j.at("vocab"));
        std::string unk = get_or<std::string>(j, "unk_token", "[UNK]");
        std::string prefix = get_or<std::string>(j, "continuing_subword_prefix", "##");
        size_t max_chars = get_or<size_t>(j, "max_input_chars_per_word", 100);
        return std::make_unique<models::WordPiece>(std::move(vocab), std::move(unk),
                                                    std::move(prefix), max_chars);
    }
    if (type == "BPE") {
        auto vocab = parse_vocab(j.at("vocab"));
        auto unk_token = get_opt<std::string>(j, "unk_token");
        auto continuing_prefix = get_opt<std::string>(j, "continuing_subword_prefix");
        auto eof_suffix = get_opt<std::string>(j, "end_of_word_suffix");
        bool fuse_unk = get_or(j, "fuse_unk", false);
        bool byte_fallback = get_or(j, "byte_fallback", false);
        bool ignore_merges = get_or(j, "ignore_merges", false);

        size_t prefix_len = continuing_prefix ? continuing_prefix->size() : 0;

        models::MergeMap merges;
        if (j.contains("merges") && !j["merges"].is_null()) {
            auto& arr = j["merges"];
            merges.reserve(arr.size());
            TokenId rank = 0;
            for (auto& merge_entry : arr) {
                std::string a, b;
                if (merge_entry.is_string()) {
                    // "a b" format
                    std::string s = merge_entry.get<std::string>();
                    auto space_pos = s.find(' ');
                    if (space_pos == std::string::npos) {
                        return make_error("Invalid merge string (no space): " + s);
                    }
                    a = s.substr(0, space_pos);
                    b = s.substr(space_pos + 1);
                } else if (merge_entry.is_array() && merge_entry.size() == 2) {
                    // ["a", "b"] format
                    a = merge_entry[0].get<std::string>();
                    b = merge_entry[1].get<std::string>();
                } else {
                    ++rank;
                    continue;
                }

                auto a_it = vocab.find(a);
                auto b_it = vocab.find(b);
                if (a_it == vocab.end() || b_it == vocab.end()) {
                    ++rank;
                    continue; // skip merges with unknown tokens
                }
                TokenId a_id = a_it->second;
                TokenId b_id = b_it->second;

                // Compute merged token: a + b[prefix_len..]
                std::string merged = a + b.substr(prefix_len);
                auto m_it = vocab.find(merged);
                if (m_it == vocab.end()) {
                    ++rank;
                    continue; // skip if merged token not in vocab
                }
                TokenId new_id = m_it->second;
                merges.emplace(models::Pair{a_id, b_id},
                               std::pair<TokenId, TokenId>{rank, new_id});
                ++rank;
            }
        }

        return std::make_unique<models::BPE>(std::move(vocab), std::move(merges),
                                              std::move(unk_token), std::move(continuing_prefix),
                                              std::move(eof_suffix), fuse_unk, byte_fallback,
                                              ignore_merges);
    }
    if (type == "WordLevel") {
        auto vocab = parse_vocab(j.at("vocab"));
        std::string unk = get_or<std::string>(j, "unk_token", "[UNK]");
        return std::make_unique<models::WordLevel>(std::move(vocab), std::move(unk));
    }
    if (type == "Unigram") {
        std::vector<std::pair<std::string, double>> vocab;
        if (j.contains("vocab") && !j["vocab"].is_null()) {
            for (auto& item : j["vocab"]) {
                std::string token = item[0].get<std::string>();
                double score = item[1].get<double>();
                vocab.emplace_back(std::move(token), score);
            }
        }
        auto unk_id = get_opt<size_t>(j, "unk_id");
        bool byte_fallback = get_or(j, "byte_fallback", false);
        return std::make_unique<models::Unigram>(std::move(vocab), unk_id, byte_fallback);
    }
    return make_error("Unknown model type: " + type);
}

// ── truncation / padding ─────────────────────────────────────────────────────

static std::optional<TruncationParams> parse_truncation(const json& j) {
    if (j.is_null()) return std::nullopt;
    TruncationParams p;
    p.max_length = get_or<size_t>(j, "max_length", 512);
    p.stride = get_or<size_t>(j, "stride", 0);
    auto strat = get_or<std::string>(j, "strategy", "LongestFirst");
    if (strat == "OnlyFirst") p.strategy = TruncationStrategy::OnlyFirst;
    else if (strat == "OnlySecond") p.strategy = TruncationStrategy::OnlySecond;
    else p.strategy = TruncationStrategy::LongestFirst;
    auto dir = get_or<std::string>(j, "direction", "Right");
    p.direction = (dir == "Left") ? TruncationDirection::Left : TruncationDirection::Right;
    return p;
}

static std::optional<PaddingParams> parse_padding(const json& j) {
    if (j.is_null()) return std::nullopt;
    PaddingParams p;
    p.pad_id = get_or<TokenId>(j, "pad_id", 0);
    p.pad_type_id = get_or<TokenId>(j, "pad_type_id", 0);
    p.pad_token = get_or<std::string>(j, "pad_token", "[PAD]");
    auto dir = get_or<std::string>(j, "direction", "Right");
    p.direction = (dir == "Left") ? PaddingDirection::Left : PaddingDirection::Right;
    p.pad_to_multiple_of = get_or<size_t>(j, "pad_to_multiple_of", 0);
    if (j.contains("length") && !j["length"].is_null()) {
        p.strategy = PaddingStrategy::Fixed;
        p.fixed_length = j["length"].get<size_t>();
    }
    return p;
}

// ── added_tokens ─────────────────────────────────────────────────────────────

static AddedToken parse_added_token(const json& j) {
    AddedToken t;
    t.content = j.at("content").get<std::string>();
    t.single_word = get_or(j, "single_word", false);
    t.lstrip = get_or(j, "lstrip", false);
    t.rstrip = get_or(j, "rstrip", false);
    t.normalized = get_or(j, "normalized", true);
    t.special = get_or(j, "special", false);
    return t;
}

// ── from_string / from_file ──────────────────────────────────────────────────

Result<Tokenizer> Tokenizer::from_string(std::string_view json_str) {
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::parse_error& e) {
        return make_error(std::string("JSON parse error: ") + e.what());
    }
    return from_json(j);
}

Result<Tokenizer> Tokenizer::from_json(const json& j) {
    // model (required)
    if (!j.contains("model") || j["model"].is_null()) {
        return make_error("tokenizer.json missing 'model' field");
    }
    auto model_result = parse_model(j["model"]);
    if (!model_result) return std::unexpected(model_result.error());

    Tokenizer tokenizer(std::move(*model_result));

    // normalizer
    if (j.contains("normalizer")) {
        auto r = parse_normalizer(j["normalizer"]);
        if (!r) return std::unexpected(r.error());
        if (*r) tokenizer.with_normalizer(std::move(*r));
    }

    // pre_tokenizer
    if (j.contains("pre_tokenizer")) {
        auto r = parse_pre_tokenizer(j["pre_tokenizer"]);
        if (!r) return std::unexpected(r.error());
        if (*r) tokenizer.with_pre_tokenizer(std::move(*r));
    }

    // post_processor
    if (j.contains("post_processor")) {
        auto r = parse_post_processor(j["post_processor"]);
        if (!r) return std::unexpected(r.error());
        if (*r) tokenizer.with_post_processor(std::move(*r));
    }

    // decoder
    if (j.contains("decoder")) {
        auto r = parse_decoder(j["decoder"]);
        if (!r) return std::unexpected(r.error());
        if (*r) tokenizer.with_decoder(std::move(*r));
    }

    // truncation
    if (j.contains("truncation")) {
        tokenizer.with_truncation(parse_truncation(j["truncation"]));
    }

    // padding
    if (j.contains("padding")) {
        tokenizer.with_padding(parse_padding(j["padding"]));
    }

    // added_tokens
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        std::vector<AddedToken> special_tokens;
        std::vector<AddedToken> normal_tokens;
        for (auto& tok_json : j["added_tokens"]) {
            auto tok = parse_added_token(tok_json);
            if (tok.special) {
                special_tokens.push_back(std::move(tok));
            } else {
                normal_tokens.push_back(std::move(tok));
            }
        }
        if (!special_tokens.empty()) tokenizer.add_special_tokens(special_tokens);
        if (!normal_tokens.empty()) tokenizer.add_tokens(normal_tokens);
    }

    return tokenizer;
}

Result<Tokenizer> Tokenizer::from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return make_error("Cannot open file: " + path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    auto tok = from_string(ss.str());
    if (!tok) return tok;

    // Try loading sibling tokenizer_config.json
    auto slash = path.rfind('/');
    if (slash != std::string::npos) {
        auto dir = path.substr(0, slash);
        auto config = TokenizerConfig::from_file(dir + "/tokenizer_config.json");
        if (config) {
            tok->with_config(std::move(*config));
        }
    }

    return tok;
}

Result<Tokenizer> Tokenizer::from_directory(const std::string& dir) {
    auto tok = from_file(dir + "/tokenizer.json");
    if (!tok) return tok;

    // If from_file didn't already load the config (e.g. dir has no slash),
    // try explicitly
    if (!tok->get_config()) {
        auto config = TokenizerConfig::from_file(dir + "/tokenizer_config.json");
        if (config) {
            tok->with_config(std::move(*config));
        }
    }

    return tok;
}

// ── Serialization helpers ────────────────────────────────────────────────────

static std::string char32_to_utf8(char32_t ch) {
    std::string result;
    if (ch <= 0x7F) {
        result += static_cast<char>(ch);
    } else if (ch <= 0x7FF) {
        result += static_cast<char>(0xC0 | (ch >> 6));
        result += static_cast<char>(0x80 | (ch & 0x3F));
    } else if (ch <= 0xFFFF) {
        result += static_cast<char>(0xE0 | (ch >> 12));
        result += static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (ch & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (ch >> 18));
        result += static_cast<char>(0x80 | ((ch >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (ch & 0x3F));
    }
    return result;
}

static json serialize_normalizer(const Normalizer* n);
static json serialize_pre_tokenizer(const PreTokenizer* pt);
static json serialize_post_processor(const PostProcessor* pp);
static json serialize_decoder(const Decoder* d);
static json serialize_model(const Model* m);

static json serialize_normalizer(const Normalizer* n) {
    if (!n) return nullptr;

    if (auto* p = dynamic_cast<const normalizers::BertNormalizer*>(n)) {
        json j;
        j["type"] = "BertNormalizer";
        j["clean_text"] = p->clean_text;
        j["handle_chinese_chars"] = p->handle_chinese_chars;
        j["strip_accents"] = p->strip_accents.has_value() ? json(*p->strip_accents) : json(nullptr);
        j["lowercase"] = p->lowercase;
        return j;
    }
    if (dynamic_cast<const normalizers::NFCNormalizer*>(n)) return json{{"type", "NFC"}};
    if (dynamic_cast<const normalizers::NFKCNormalizer*>(n)) return json{{"type", "NFKC"}};
    if (dynamic_cast<const normalizers::NFDNormalizer*>(n)) return json{{"type", "NFD"}};
    if (dynamic_cast<const normalizers::NFKDNormalizer*>(n)) return json{{"type", "NFKD"}};
    if (dynamic_cast<const normalizers::NmtNormalizer*>(n)) return json{{"type", "Nmt"}};
    if (dynamic_cast<const normalizers::LowercaseNormalizer*>(n)) return json{{"type", "Lowercase"}};
    if (dynamic_cast<const normalizers::StripAccentsNormalizer*>(n)) return json{{"type", "StripAccents"}};
    if (auto* p = dynamic_cast<const normalizers::StripNormalizer*>(n)) {
        return json{{"type", "Strip"}, {"strip_left", p->strip_left}, {"strip_right", p->strip_right}};
    }
    if (auto* p = dynamic_cast<const normalizers::PrependNormalizer*>(n)) {
        return json{{"type", "Prepend"}, {"prepend", p->prepend_str}};
    }
    if (auto* p = dynamic_cast<const normalizers::ReplaceNormalizer*>(n)) {
        json j;
        j["type"] = "Replace";
        j["content"] = p->content;
        if (auto* sp = dynamic_cast<const StringPattern*>(p->pattern.get()))
            j["pattern"] = json{{"String", sp->get_pattern()}};
        else if (auto* rp = dynamic_cast<const RegexPattern*>(p->pattern.get()))
            j["pattern"] = json{{"Regex", rp->get_pattern()}};
        return j;
    }
    if (auto* p = dynamic_cast<const normalizers::SequenceNormalizer*>(n)) {
        json arr = json::array();
        for (auto& child : p->normalizers)
            arr.push_back(serialize_normalizer(child.get()));
        return json{{"type", "Sequence"}, {"normalizers", arr}};
    }
    if (dynamic_cast<const normalizers::ByteLevelNormalizer*>(n)) return json{{"type", "ByteLevel"}};
    if (dynamic_cast<const normalizers::PrecompiledNormalizer*>(n)) return json{{"type", "Precompiled"}};

    return nullptr;
}

static json serialize_pre_tokenizer(const PreTokenizer* pt) {
    if (!pt) return nullptr;

    if (dynamic_cast<const pre_tokenizers::BertPreTokenizer*>(pt))
        return json{{"type", "BertPreTokenizer"}};
    if (dynamic_cast<const pre_tokenizers::Whitespace*>(pt))
        return json{{"type", "Whitespace"}};
    if (dynamic_cast<const pre_tokenizers::WhitespaceSplit*>(pt))
        return json{{"type", "WhitespaceSplit"}};
    if (dynamic_cast<const pre_tokenizers::Punctuation*>(pt))
        return json{{"type", "Punctuation"}};
    if (auto* p = dynamic_cast<const pre_tokenizers::Digits*>(pt))
        return json{{"type", "Digits"}, {"individual_digits", p->individual_digits}};
    if (auto* p = dynamic_cast<const pre_tokenizers::CharDelimiterSplit*>(pt))
        return json{{"type", "CharDelimiterSplit"}, {"delimiter", char32_to_utf8(p->delimiter)}};
    if (auto* p = dynamic_cast<const pre_tokenizers::ByteLevel*>(pt)) {
        return json{{"type", "ByteLevel"},
                     {"add_prefix_space", p->add_prefix_space},
                     {"trim_offsets", p->trim_offsets},
                     {"use_regex", p->use_regex}};
    }
    if (auto* p = dynamic_cast<const pre_tokenizers::Metaspace*>(pt)) {
        json j;
        j["type"] = "Metaspace";
        j["replacement"] = p->replacement;
        j["str_rep"] = p->replacement;
        switch (p->prepend_scheme) {
            case pre_tokenizers::Metaspace::PrependScheme::Always:
                j["prepend_scheme"] = "always"; break;
            case pre_tokenizers::Metaspace::PrependScheme::First:
                j["prepend_scheme"] = "first"; break;
            case pre_tokenizers::Metaspace::PrependScheme::Never:
                j["prepend_scheme"] = "never"; break;
        }
        j["split"] = p->split;
        return j;
    }
    if (auto* p = dynamic_cast<const pre_tokenizers::SplitPreTokenizer*>(pt)) {
        json j;
        j["type"] = "Split";
        if (auto* sp = dynamic_cast<const StringPattern*>(p->pattern.get()))
            j["pattern"] = json{{"String", sp->get_pattern()}};
        else if (auto* rp = dynamic_cast<const RegexPattern*>(p->pattern.get()))
            j["pattern"] = json{{"Regex", rp->get_pattern()}};
        switch (p->behavior) {
            case SplitDelimiterBehavior::Removed: j["behavior"] = "Removed"; break;
            case SplitDelimiterBehavior::Isolated: j["behavior"] = "Isolated"; break;
            case SplitDelimiterBehavior::MergedWithPrevious: j["behavior"] = "MergedWithPrevious"; break;
            case SplitDelimiterBehavior::MergedWithNext: j["behavior"] = "MergedWithNext"; break;
            case SplitDelimiterBehavior::Contiguous: j["behavior"] = "Contiguous"; break;
        }
        j["invert"] = p->invert;
        return j;
    }
    if (auto* p = dynamic_cast<const pre_tokenizers::SequencePreTokenizer*>(pt)) {
        json arr = json::array();
        for (auto& child : p->pre_tokenizers)
            arr.push_back(serialize_pre_tokenizer(child.get()));
        return json{{"type", "Sequence"}, {"pretokenizers", arr}};
    }
    if (dynamic_cast<const pre_tokenizers::UnicodeScripts*>(pt))
        return json{{"type", "UnicodeScripts"}};
    if (auto* p = dynamic_cast<const pre_tokenizers::FixedLength*>(pt))
        return json{{"type", "FixedLength"}, {"length", p->length}};

    return nullptr;
}

static json serialize_post_processor(const PostProcessor* pp) {
    if (!pp) return nullptr;

    if (auto* p = dynamic_cast<const processors::BertProcessing*>(pp)) {
        return json{{"type", "BertProcessing"},
                     {"sep", json::array({p->sep.first, p->sep.second})},
                     {"cls", json::array({p->cls.first, p->cls.second})}};
    }
    if (auto* p = dynamic_cast<const processors::RobertaProcessing*>(pp)) {
        return json{{"type", "RobertaProcessing"},
                     {"sep", json::array({p->sep.first, p->sep.second})},
                     {"cls", json::array({p->cls.first, p->cls.second})},
                     {"trim_offsets", p->trim_offsets},
                     {"add_prefix_space", p->add_prefix_space}};
    }
    if (auto* p = dynamic_cast<const processors::ByteLevelProcessing*>(pp)) {
        return json{{"type", "ByteLevel"},
                    {"add_prefix_space", p->add_prefix_space},
                    {"trim_offsets", p->trim_offsets},
                    {"use_regex", p->use_regex}};
    }
    if (auto* p = dynamic_cast<const processors::TemplateProcessing*>(pp)) {
        auto serialize_template = [](const std::vector<processors::TemplatePiece>& tmpl) {
            json arr = json::array();
            for (auto& piece : tmpl) {
                if (piece.kind == processors::TemplatePiece::Sequence) {
                    std::string id = (piece.sequence == processors::TemplateSequence::A) ? "A" : "B";
                    arr.push_back(json{{"Sequence", {{"id", id}, {"type_id", piece.type_id}}}});
                } else {
                    arr.push_back(json{{"SpecialToken", {{"id", piece.special_token}, {"type_id", piece.type_id}}}});
                }
            }
            return arr;
        };
        json j;
        j["type"] = "TemplateProcessing";
        j["single"] = serialize_template(p->single_template);
        j["pair"] = serialize_template(p->pair_template);
        json st = json::object();
        for (auto& [key, def] : p->special_tokens) {
            st[key] = json{{"id", def.id}, {"ids", def.ids}, {"tokens", def.tokens}};
        }
        j["special_tokens"] = st;
        return j;
    }
    if (auto* p = dynamic_cast<const processors::SequenceProcessing*>(pp)) {
        json arr = json::array();
        for (auto& child : p->processors)
            arr.push_back(serialize_post_processor(child.get()));
        return json{{"type", "Sequence"}, {"processors", arr}};
    }

    return nullptr;
}

static json serialize_decoder(const Decoder* d) {
    if (!d) return nullptr;

    if (auto* p = dynamic_cast<const decoders::WordPieceDecoder*>(d)) {
        return json{{"type", "WordPiece"}, {"prefix", p->prefix}, {"cleanup", p->cleanup}};
    }
    if (dynamic_cast<const decoders::ByteLevelDecoder*>(d))
        return json{{"type", "ByteLevel"}};
    if (auto* p = dynamic_cast<const decoders::BPEDecoder*>(d))
        return json{{"type", "BPE"}, {"suffix", p->suffix}};
    if (dynamic_cast<const decoders::ByteFallbackDecoder*>(d))
        return json{{"type", "ByteFallback"}};
    if (auto* p = dynamic_cast<const decoders::CTCDecoder*>(d)) {
        return json{{"type", "CTC"},
                     {"pad_token", p->pad_token},
                     {"word_delimiter_token", p->word_delimiter_token},
                     {"cleanup", p->cleanup}};
    }
    if (dynamic_cast<const decoders::FuseDecoder*>(d))
        return json{{"type", "Fuse"}};
    if (auto* p = dynamic_cast<const decoders::StripDecoder*>(d)) {
        return json{{"type", "Strip"},
                     {"content", char32_to_utf8(p->content)},
                     {"start", p->start},
                     {"stop", p->stop}};
    }
    if (auto* p = dynamic_cast<const decoders::MetaspaceDecoder*>(d)) {
        json j;
        j["type"] = "Metaspace";
        j["replacement"] = p->replacement;
        j["prepend_scheme"] = p->prepend_scheme_always ? "always" : "first";
        j["add_prefix_space"] = p->prepend_scheme_always;
        return j;
    }
    if (auto* p = dynamic_cast<const decoders::ReplaceDecoder*>(d)) {
        return json{{"type", "Replace"},
                     {"pattern", json{{"String", p->pattern}}},
                     {"content", p->content}};
    }
    if (auto* p = dynamic_cast<const decoders::SequenceDecoder*>(d)) {
        json arr = json::array();
        for (auto& child : p->decoders)
            arr.push_back(serialize_decoder(child.get()));
        return json{{"type", "Sequence"}, {"decoders", arr}};
    }

    return nullptr;
}

static json serialize_model(const Model* m) {
    if (!m) return nullptr;

    if (auto* p = dynamic_cast<const models::WordPiece*>(m)) {
        // Build ordered vocab (sorted by ID)
        auto vocab_map = p->get_vocab();
        std::vector<std::pair<std::string, TokenId>> sorted_vocab(vocab_map.begin(), vocab_map.end());
        std::sort(sorted_vocab.begin(), sorted_vocab.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        json vocab = json::object();
        for (auto& [tok, id] : sorted_vocab)
            vocab[tok] = id;

        return json{{"type", "WordPiece"},
                     {"unk_token", p->get_unk_token()},
                     {"continuing_subword_prefix", p->get_continuing_subword_prefix()},
                     {"max_input_chars_per_word", p->get_max_input_chars_per_word()},
                     {"vocab", vocab}};
    }
    if (auto* p = dynamic_cast<const models::BPE*>(m)) {
        auto vocab_map = p->get_vocab();
        // Build reverse vocab (id → token)
        std::unordered_map<TokenId, std::string> id_to_tok;
        for (auto& [tok, id] : vocab_map)
            id_to_tok[id] = tok;

        // Build ordered vocab
        std::vector<std::pair<std::string, TokenId>> sorted_vocab(vocab_map.begin(), vocab_map.end());
        std::sort(sorted_vocab.begin(), sorted_vocab.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        json vocab = json::object();
        for (auto& [tok, id] : sorted_vocab)
            vocab[tok] = id;

        // Build merges sorted by rank
        auto& merge_map = p->get_merges();
        std::vector<std::pair<models::Pair, TokenId>> sorted_merges;
        sorted_merges.reserve(merge_map.size());
        for (auto& [pair, rank_newid] : merge_map)
            sorted_merges.push_back({pair, rank_newid.first});
        std::sort(sorted_merges.begin(), sorted_merges.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        json merges = json::array();
        for (auto& [pair, rank] : sorted_merges) {
            auto a_it = id_to_tok.find(pair.first);
            auto b_it = id_to_tok.find(pair.second);
            if (a_it != id_to_tok.end() && b_it != id_to_tok.end())
                merges.push_back(a_it->second + " " + b_it->second);
        }

        json j;
        j["type"] = "BPE";
        j["dropout"] = nullptr;
        j["unk_token"] = p->get_unk_token().has_value() ? json(*p->get_unk_token()) : json(nullptr);
        j["continuing_subword_prefix"] = p->get_continuing_subword_prefix().has_value()
            ? json(*p->get_continuing_subword_prefix()) : json(nullptr);
        j["end_of_word_suffix"] = p->get_end_of_word_suffix().has_value()
            ? json(*p->get_end_of_word_suffix()) : json(nullptr);
        j["fuse_unk"] = p->get_fuse_unk();
        j["byte_fallback"] = p->get_byte_fallback();
        j["ignore_merges"] = p->get_ignore_merges();
        j["vocab"] = vocab;
        j["merges"] = merges;
        return j;
    }
    if (auto* p = dynamic_cast<const models::WordLevel*>(m)) {
        auto vocab_map = p->get_vocab();
        std::vector<std::pair<std::string, TokenId>> sorted_vocab(vocab_map.begin(), vocab_map.end());
        std::sort(sorted_vocab.begin(), sorted_vocab.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        json vocab = json::object();
        for (auto& [tok, id] : sorted_vocab)
            vocab[tok] = id;

        return json{{"type", "WordLevel"},
                     {"unk_token", p->get_unk_token()},
                     {"vocab", vocab}};
    }
    if (auto* p = dynamic_cast<const models::Unigram*>(m)) {
        auto& vocab_scores = p->get_vocab_scores();
        json vocab = json::array();
        for (auto& [tok, score] : vocab_scores)
            vocab.push_back(json::array({tok, score}));

        json j;
        j["type"] = "Unigram";
        j["unk_id"] = p->get_unk_id().has_value() ? json(*p->get_unk_id()) : json(nullptr);
        j["byte_fallback"] = p->get_byte_fallback();
        j["vocab"] = vocab;
        return j;
    }

    return nullptr;
}

// ── to_string / save ─────────────────────────────────────────────────────────

Result<std::string> Tokenizer::to_string(bool pretty) const {
    if (!model_) {
        return make_error("No model set on Tokenizer");
    }

    json j;
    j["version"] = "1.0";

    // truncation
    if (truncation_) {
        json t;
        t["max_length"] = truncation_->max_length;
        t["stride"] = truncation_->stride;
        switch (truncation_->strategy) {
            case TruncationStrategy::LongestFirst: t["strategy"] = "LongestFirst"; break;
            case TruncationStrategy::OnlyFirst: t["strategy"] = "OnlyFirst"; break;
            case TruncationStrategy::OnlySecond: t["strategy"] = "OnlySecond"; break;
        }
        t["direction"] = (truncation_->direction == TruncationDirection::Left) ? "Left" : "Right";
        j["truncation"] = t;
    } else {
        j["truncation"] = nullptr;
    }

    // padding
    if (padding_) {
        json p;
        if (padding_->strategy == PaddingStrategy::Fixed) {
            p["length"] = padding_->fixed_length;
        }
        p["pad_id"] = padding_->pad_id;
        p["pad_type_id"] = padding_->pad_type_id;
        p["pad_token"] = padding_->pad_token;
        p["direction"] = (padding_->direction == PaddingDirection::Left) ? "Left" : "Right";
        p["pad_to_multiple_of"] = padding_->pad_to_multiple_of > 0
            ? json(padding_->pad_to_multiple_of) : json(nullptr);
        j["padding"] = p;
    } else {
        j["padding"] = nullptr;
    }

    // added_tokens
    json added = json::array();
    if (added_vocabulary_) {
        for (auto& at : added_vocabulary_->get_added_tokens()) {
            json t;
            t["id"] = at.id;
            t["content"] = at.token.content;
            t["single_word"] = at.token.single_word;
            t["lstrip"] = at.token.lstrip;
            t["rstrip"] = at.token.rstrip;
            t["normalized"] = at.token.normalized;
            t["special"] = at.token.special;
            added.push_back(t);
        }
    }
    j["added_tokens"] = added;

    // components
    j["normalizer"] = serialize_normalizer(normalizer_.get());
    j["pre_tokenizer"] = serialize_pre_tokenizer(pre_tokenizer_.get());
    j["post_processor"] = serialize_post_processor(post_processor_.get());
    j["decoder"] = serialize_decoder(decoder_.get());
    j["model"] = serialize_model(model_.get());

    return pretty ? j.dump(2) : j.dump();
}

Result<void> Tokenizer::save(const std::string& path, bool pretty) const {
    auto str = to_string(pretty);
    if (!str) return std::unexpected(str.error());

    std::ofstream file(path);
    if (!file.is_open()) {
        return make_error("Cannot open file for writing: " + path);
    }
    file << *str;
    if (!file.good()) {
        return make_error("Error writing to file: " + path);
    }
    return {};
}

} // namespace tokenizers
