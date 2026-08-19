#include <gtest/gtest.h>
#include "tokenizers/tokenizer.h"
#include "tokenizers/tokenizer_config.h"
#include "tokenizers/chat_template.h"
#include "tokenizers/models.h"
#include "tokenizers/normalizers.h"
#include "tokenizers/normalized_string.h"
#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/pre_tokenized_string.h"
#include "tokenizers/processors.h"
#include "tokenizers/decoders.h"

#include <nlohmann/json.hpp>

#include <cstdio>

namespace tokenizers {
namespace {

// Helper: create a BERT-like vocabulary
std::unordered_map<std::string, TokenId> make_bert_vocab() {
    return {
        {"[PAD]", 0}, {"[UNK]", 1}, {"[CLS]", 2}, {"[SEP]", 3}, {"[MASK]", 4},
        {"hello", 5}, {"world", 6}, {"##lo", 7}, {"hel", 8},
        {"the", 9}, {"a", 10}, {"is", 11},
    };
}

// === BertProcessing tests ===

TEST(BertProcessingTest, AddedTokensCount) {
    processors::BertProcessing proc({"[SEP]", 3}, {"[CLS]", 2});
    EXPECT_EQ(proc.added_tokens(false), 2u);
    EXPECT_EQ(proc.added_tokens(true), 3u);
}

TEST(BertProcessingTest, SingleEncoding) {
    processors::BertProcessing proc({"[SEP]", 3}, {"[CLS]", 2});

    // Create a simple encoding
    Encoding enc({5, 6}, {0, 0}, {"hello", "world"},
                 {std::nullopt, std::nullopt}, {{0, 5}, {6, 11}},
                 {0, 0}, {1, 1}, {}, {});

    auto result = proc.process(std::move(enc), std::nullopt, true);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->get_ids().size(), 4u);  // CLS + hello + world + SEP
    EXPECT_EQ(result->get_ids()[0], 2u);  // CLS
    EXPECT_EQ(result->get_ids()[1], 5u);  // hello
    EXPECT_EQ(result->get_ids()[2], 6u);  // world
    EXPECT_EQ(result->get_ids()[3], 3u);  // SEP
}

TEST(BertProcessingTest, PairEncoding) {
    processors::BertProcessing proc({"[SEP]", 3}, {"[CLS]", 2});

    Encoding enc1({5}, {0}, {"hello"}, {std::nullopt}, {{0, 5}}, {0}, {1}, {}, {});
    Encoding enc2({6}, {0}, {"world"}, {std::nullopt}, {{0, 5}}, {0}, {1}, {}, {});

    auto result = proc.process(std::move(enc1), std::move(enc2), true);
    ASSERT_TRUE(result.has_value());

    // CLS + hello + SEP + world + SEP = 5 tokens
    EXPECT_EQ(result->get_ids().size(), 5u);
    EXPECT_EQ(result->get_ids()[0], 2u);  // CLS
    EXPECT_EQ(result->get_ids()[1], 5u);  // hello
    EXPECT_EQ(result->get_ids()[2], 3u);  // SEP
    EXPECT_EQ(result->get_ids()[3], 6u);  // world
    EXPECT_EQ(result->get_ids()[4], 3u);  // SEP
}

TEST(BertProcessingTest, NoSpecialTokens) {
    processors::BertProcessing proc({"[SEP]", 3}, {"[CLS]", 2});

    Encoding enc({5, 6}, {0, 0}, {"hello", "world"},
                 {std::nullopt, std::nullopt}, {{0, 5}, {6, 11}},
                 {0, 0}, {1, 1}, {}, {});

    auto result = proc.process(std::move(enc), std::nullopt, false);
    ASSERT_TRUE(result.has_value());

    // No special tokens added
    EXPECT_EQ(result->get_ids().size(), 2u);
    EXPECT_EQ(result->get_ids()[0], 5u);
    EXPECT_EQ(result->get_ids()[1], 6u);
}

// === SequenceProcessing tests ===

TEST(SequenceProcessingTest, ChainedProcessors) {
    std::vector<PostProcessorPtr> procs;
    procs.push_back(std::make_unique<processors::BertProcessing>(
        std::pair<std::string, TokenId>{"[SEP]", 3},
        std::pair<std::string, TokenId>{"[CLS]", 2}));
    processors::SequenceProcessing seq(std::move(procs));

    EXPECT_EQ(seq.added_tokens(false), 2u);

    Encoding enc({5}, {0}, {"hello"}, {std::nullopt}, {{0, 5}}, {0}, {1}, {}, {});
    auto result = seq.process(std::move(enc), std::nullopt, true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->get_ids().size(), 3u);  // CLS + hello + SEP
}

// === End-to-end WordPiece test ===

TEST(TokenizerE2E, WordPieceEncodeDecode) {
    auto vocab = make_bert_vocab();
    auto model = std::make_unique<models::WordPiece>(vocab);

    Tokenizer tokenizer(std::move(model));
    tokenizer.with_normalizer(std::make_unique<normalizers::BertNormalizer>(
        true, true, std::nullopt, true));
    tokenizer.with_pre_tokenizer(std::make_unique<pre_tokenizers::BertPreTokenizer>());
    tokenizer.with_post_processor(std::make_unique<processors::BertProcessing>(
        std::pair<std::string, TokenId>{"[SEP]", 3},
        std::pair<std::string, TokenId>{"[CLS]", 2}));
    tokenizer.with_decoder(std::make_unique<decoders::WordPieceDecoder>());

    // Encode
    auto enc = tokenizer.encode("hello world", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();

    auto& ids = enc->get_ids();
    // Should be [CLS] hello world [SEP]
    ASSERT_GE(ids.size(), 4u);
    EXPECT_EQ(ids.front(), 2u);  // CLS
    EXPECT_EQ(ids.back(), 3u);   // SEP

    // Decode (without special tokens)
    auto decoded = tokenizer.decode(ids, false);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
}

TEST(TokenizerE2E, WordPieceEncodeNoSpecialTokens) {
    auto vocab = make_bert_vocab();
    auto model = std::make_unique<models::WordPiece>(vocab);

    Tokenizer tokenizer(std::move(model));
    tokenizer.with_pre_tokenizer(std::make_unique<pre_tokenizers::BertPreTokenizer>());
    tokenizer.with_post_processor(std::make_unique<processors::BertProcessing>(
        std::pair<std::string, TokenId>{"[SEP]", 3},
        std::pair<std::string, TokenId>{"[CLS]", 2}));

    auto enc = tokenizer.encode("hello world", false);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();

    auto& ids = enc->get_ids();
    // No CLS/SEP since add_special_tokens=false
    for (TokenId id : ids) {
        EXPECT_NE(id, 2u);  // no CLS
        EXPECT_NE(id, 3u);  // no SEP
    }
}

// === End-to-end BPE test ===

TEST(TokenizerE2E, BPEEncodeDecode) {
    using models::MergeMap;
    std::unordered_map<std::string, TokenId> vocab = {
        {"h", 0}, {"e", 1}, {"l", 2}, {"o", 3}, {" ", 4},
        {"w", 5}, {"r", 6}, {"d", 7},
        {"he", 8}, {"ll", 9}, {"wo", 10}, {"rl", 11},
    };
    MergeMap merges = {
        {{0, 1}, {0, 8}},   // h+e -> he
        {{2, 2}, {1, 9}},   // l+l -> ll
        {{5, 3}, {2, 10}},  // w+o -> wo
        {{6, 2}, {3, 11}},  // r+l -> rl
    };

    auto model = std::make_unique<models::BPE>(vocab, merges);
    Tokenizer tokenizer(std::move(model));

    auto enc = tokenizer.encode("hello", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    EXPECT_FALSE(enc->get_ids().empty());

    // Decode
    auto decoded = tokenizer.decode(enc->get_ids(), false);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    // Default decode joins with spaces
    EXPECT_FALSE(decoded->empty());
}

// === Encode/Decode round-trip ===

TEST(TokenizerE2E, RoundTrip) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    auto model = std::make_unique<models::WordPiece>(vocab);
    Tokenizer tokenizer(std::move(model));
    tokenizer.with_pre_tokenizer(std::make_unique<pre_tokenizers::WhitespaceSplit>());
    tokenizer.with_decoder(std::make_unique<decoders::WordPieceDecoder>());

    auto enc = tokenizer.encode("hello world", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();

    auto& ids = enc->get_ids();
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 2u);

    auto decoded = tokenizer.decode(ids, false);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    EXPECT_EQ(*decoded, "hello world");
}

// === Vocabulary delegation ===

TEST(TokenizerTest, VocabDelegation) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    auto model = std::make_unique<models::WordPiece>(vocab);
    Tokenizer tokenizer(std::move(model));

    EXPECT_EQ(tokenizer.get_vocab_size(), 3u);
    EXPECT_EQ(tokenizer.token_to_id("hello"), 1u);
    EXPECT_EQ(tokenizer.token_to_id("xyz"), std::nullopt);
    EXPECT_EQ(tokenizer.id_to_token(2), "world");
    EXPECT_EQ(tokenizer.id_to_token(99), std::nullopt);
}

// === Encode batch ===

TEST(TokenizerE2E, EncodeBatch) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    auto model = std::make_unique<models::WordPiece>(vocab);
    Tokenizer tokenizer(std::move(model));
    tokenizer.with_pre_tokenizer(std::make_unique<pre_tokenizers::WhitespaceSplit>());

    auto result = tokenizer.encode_batch({"hello", "world"}, true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].get_ids(), std::vector<TokenId>{1});
    EXPECT_EQ((*result)[1].get_ids(), std::vector<TokenId>{2});
}

// === Decode batch ===

TEST(TokenizerE2E, DecodeBatch) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    auto model = std::make_unique<models::WordPiece>(vocab);
    Tokenizer tokenizer(std::move(model));

    auto result = tokenizer.decode_batch({{1}, {2}}, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0], "hello");
    EXPECT_EQ((*result)[1], "world");
}

// === Pair encoding ===

TEST(TokenizerE2E, PairEncoding) {
    auto vocab = make_bert_vocab();
    auto model = std::make_unique<models::WordPiece>(vocab);
    Tokenizer tokenizer(std::move(model));
    tokenizer.with_pre_tokenizer(std::make_unique<pre_tokenizers::BertPreTokenizer>());
    tokenizer.with_post_processor(std::make_unique<processors::BertProcessing>(
        std::pair<std::string, TokenId>{"[SEP]", 3},
        std::pair<std::string, TokenId>{"[CLS]", 2}));

    auto enc = tokenizer.encode_pair("hello", "world", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();

    auto& ids = enc->get_ids();
    // CLS + hello + SEP + world + SEP = 5
    ASSERT_EQ(ids.size(), 5u);
    EXPECT_EQ(ids[0], 2u);  // CLS
    EXPECT_EQ(ids[2], 3u);  // SEP
    EXPECT_EQ(ids[4], 3u);  // SEP
}

// === Truncation ===

TEST(TokenizerE2E, Truncation) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"a", 1}, {"b", 2}, {"c", 3},
    };
    auto model = std::make_unique<models::WordPiece>(vocab);
    Tokenizer tokenizer(std::move(model));

    TruncationParams trunc;
    trunc.max_length = 2;
    tokenizer.with_truncation(trunc);

    auto enc = tokenizer.encode("abc", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    EXPECT_LE(enc->get_ids().size(), 2u);
}

// === Add special tokens and skip in decode ===

TEST(TokenizerE2E, SkipSpecialTokensInDecode) {
    auto vocab = make_bert_vocab();
    auto model = std::make_unique<models::WordPiece>(vocab);
    Tokenizer tokenizer(std::move(model));
    tokenizer.with_decoder(std::make_unique<decoders::WordPieceDecoder>());

    // Add [CLS] and [SEP] as special tokens
    tokenizer.add_special_tokens({
        AddedToken("[CLS]", true),
        AddedToken("[SEP]", true),
    });

    // Decode with skip_special_tokens=true
    auto decoded = tokenizer.decode({2, 5, 6, 3}, true);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    // Should not contain [CLS] or [SEP]
    EXPECT_EQ(decoded->find("[CLS]"), std::string::npos);
    EXPECT_EQ(decoded->find("[SEP]"), std::string::npos);
}

// === Deserialization (from_string) tests ===

static const char* kWordPieceJson = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2]},
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5, "hel": 6, "##lo": 7}
  }
})";

TEST(DeserializationTest, WordPieceFromString) {
    auto result = Tokenizer::from_string(kWordPieceJson);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto& tok = *result;

    EXPECT_EQ(tok.token_to_id("hello"), 4u);
    EXPECT_EQ(tok.token_to_id("[UNK]"), 1u);

    auto enc = tok.encode("hello world", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    auto& ids = enc->get_ids();
    ASSERT_GE(ids.size(), 4u);
    EXPECT_EQ(ids.front(), 2u);  // CLS
    EXPECT_EQ(ids.back(), 3u);   // SEP

    auto decoded = tok.decode(ids, true);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    EXPECT_EQ(decoded->find("[CLS]"), std::string::npos);
}

static const char* kBPEJson = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hel": 6, "lo": 7, "hello": 8},
    "merges": ["h e", "l l", "he l", "l o", "hel lo"]
  }
})";

TEST(DeserializationTest, BPEFromString) {
    auto result = Tokenizer::from_string(kBPEJson);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto& tok = *result;

    EXPECT_EQ(tok.token_to_id("hello"), 8u);
    auto enc = tok.encode("hello", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    // BPE should merge h+e→he, l+l→ll, he+l→hel, l+o→lo, hel+lo→hello
    EXPECT_FALSE(enc->get_ids().empty());
}

TEST(DeserializationTest, PreservesMixedAddedTokenIds) {
    constexpr auto json = R"({
        "model": {
            "type": "WordPiece",
            "vocab": {"[UNK]": 0},
            "unk_token": "[UNK]"
        },
        "added_tokens": [
            {"id": 1, "content": "<special-a>", "special": true},
            {"id": 2, "content": "<normal-a>", "special": false},
            {"id": 3, "content": "<special-b>", "special": true},
            {"id": 4, "content": "<normal-b>", "special": false}
        ]
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    EXPECT_EQ(result->token_to_id("<special-a>"), 1u);
    EXPECT_EQ(result->token_to_id("<normal-a>"), 2u);
    EXPECT_EQ(result->token_to_id("<special-b>"), 3u);
    EXPECT_EQ(result->token_to_id("<normal-b>"), 4u);
}

TEST(DeserializationTest, NullComponents) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1}}
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(result->get_normalizer(), nullptr);
    EXPECT_EQ(result->get_pre_tokenizer(), nullptr);
    EXPECT_EQ(result->get_post_processor(), nullptr);
    EXPECT_EQ(result->get_decoder(), nullptr);
}

TEST(DeserializationTest, InvalidJson) {
    auto result = Tokenizer::from_string("not json");
    ASSERT_FALSE(result.has_value());
}

TEST(DeserializationTest, MissingModel) {
    auto result = Tokenizer::from_string(R"({"model": null})");
    ASSERT_FALSE(result.has_value());
}

TEST(DeserializationTest, SequenceNormalizer) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0, "hello": 1}},
      "normalizer": {"type": "Sequence", "normalizers": [{"type": "Lowercase"}, {"type": "StripAccents"}]},
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_NE(result->get_normalizer(), nullptr);
}

TEST(DeserializationTest, RobertaPostProcessor) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0}},
      "post_processor": {"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0], "trim_offsets": true, "add_prefix_space": true}
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_NE(result->get_post_processor(), nullptr);
}

TEST(DeserializationTest, ByteLevelSequenceIsConcreteAndRoundTrips) {
        const char* json = R"({
            "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0, "hello": 1}},
            "post_processor": {
                "type": "Sequence",
                "processors": [{
                    "type": "ByteLevel",
                    "add_prefix_space": true,
                    "trim_offsets": false,
                    "use_regex": false
                }]
            }
        })";

        auto tokenizer = Tokenizer::from_string(json);
        ASSERT_TRUE(tokenizer.has_value()) << tokenizer.error().message();
        auto* sequence = dynamic_cast<const processors::SequenceProcessing*>(
                tokenizer->get_post_processor());
        ASSERT_NE(sequence, nullptr);
        ASSERT_EQ(sequence->processors.size(), 1u);
        auto* byte_level = dynamic_cast<const processors::ByteLevelProcessing*>(
                sequence->processors.front().get());
        ASSERT_NE(byte_level, nullptr);
        EXPECT_TRUE(byte_level->add_prefix_space);
        EXPECT_FALSE(byte_level->trim_offsets);
        EXPECT_FALSE(byte_level->use_regex);

        auto encoding = tokenizer->encode("hello", true);
        ASSERT_TRUE(encoding.has_value()) << encoding.error().message();
        EXPECT_EQ(encoding->get_ids(), std::vector<TokenId>{1});

        auto serialized = tokenizer->to_string(false);
        ASSERT_TRUE(serialized.has_value()) << serialized.error().message();
        auto document = nlohmann::json::parse(*serialized);
        const auto& saved = document["post_processor"]["processors"][0];
        EXPECT_EQ(saved["type"], "ByteLevel");
        EXPECT_EQ(saved["add_prefix_space"], true);
        EXPECT_EQ(saved["trim_offsets"], false);
        EXPECT_EQ(saved["use_regex"], false);
        EXPECT_TRUE(Tokenizer::from_string(*serialized).has_value());

        document["post_processor"]["processors"].push_back(nullptr);
        auto invalid = Tokenizer::from_string(document.dump());
        ASSERT_FALSE(invalid.has_value());
        EXPECT_NE(invalid.error().message().find("null child"), std::string::npos);

        std::vector<PostProcessorPtr> null_children;
        null_children.push_back(nullptr);
        EXPECT_THROW(processors::SequenceProcessing(std::move(null_children)),
                                 std::invalid_argument);
}

TEST(DeserializationTest, TruncationAndPadding) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0, "a": 1, "b": 2, "c": 3}},
      "truncation": {"max_length": 2, "stride": 0, "strategy": "LongestFirst"},
      "padding": null
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto enc = result->encode("abc", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    EXPECT_LE(enc->get_ids().size(), 2u);
}

// === TemplateProcessing tests ===

TEST(TemplateProcessingTest, AddedTokensCount) {
    // BERT-style: single=[CLS] $A [SEP], pair=[CLS] $A [SEP] $B [SEP]
    processors::SpecialTokenDef cls_def{"[CLS]", 2, {2}, {"[CLS]"}};
    processors::SpecialTokenDef sep_def{"[SEP]", 3, {3}, {"[SEP]"}};

    std::vector<processors::TemplatePiece> single_tmpl = {
        {processors::TemplatePiece::SpecialToken, {}, "[CLS]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::A, {}, 0},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 0},
    };
    std::vector<processors::TemplatePiece> pair_tmpl = {
        {processors::TemplatePiece::SpecialToken, {}, "[CLS]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::A, {}, 0},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::B, {}, 1},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 1},
    };

    processors::TemplateProcessing proc(single_tmpl, pair_tmpl, {cls_def, sep_def});
    EXPECT_EQ(proc.added_tokens(false), 2u); // [CLS] + [SEP]
    EXPECT_EQ(proc.added_tokens(true), 3u);  // [CLS] + [SEP] + [SEP]
}

TEST(TemplateProcessingTest, SingleEncoding) {
    processors::SpecialTokenDef cls_def{"[CLS]", 2, {2}, {"[CLS]"}};
    processors::SpecialTokenDef sep_def{"[SEP]", 3, {3}, {"[SEP]"}};

    std::vector<processors::TemplatePiece> single_tmpl = {
        {processors::TemplatePiece::SpecialToken, {}, "[CLS]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::A, {}, 0},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 0},
    };
    std::vector<processors::TemplatePiece> pair_tmpl = {
        {processors::TemplatePiece::SpecialToken, {}, "[CLS]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::A, {}, 0},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::B, {}, 1},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 1},
    };

    processors::TemplateProcessing proc(single_tmpl, pair_tmpl, {cls_def, sep_def});

    Encoding enc({5, 6}, {0, 0}, {"hello", "world"},
                 {std::nullopt, std::nullopt}, {{0, 5}, {6, 11}},
                 {0, 0}, {1, 1}, {}, {});

    auto result = proc.process(std::move(enc), std::nullopt, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    EXPECT_EQ(result->get_ids().size(), 4u);  // CLS + hello + world + SEP
    EXPECT_EQ(result->get_ids()[0], 2u);  // CLS
    EXPECT_EQ(result->get_ids()[1], 5u);  // hello
    EXPECT_EQ(result->get_ids()[2], 6u);  // world
    EXPECT_EQ(result->get_ids()[3], 3u);  // SEP

    // type_ids should all be 0 for single
    for (auto tid : result->get_type_ids()) EXPECT_EQ(tid, 0u);
}

TEST(TemplateProcessingTest, PairEncoding) {
    processors::SpecialTokenDef cls_def{"[CLS]", 2, {2}, {"[CLS]"}};
    processors::SpecialTokenDef sep_def{"[SEP]", 3, {3}, {"[SEP]"}};

    std::vector<processors::TemplatePiece> single_tmpl = {
        {processors::TemplatePiece::SpecialToken, {}, "[CLS]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::A, {}, 0},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 0},
    };
    std::vector<processors::TemplatePiece> pair_tmpl = {
        {processors::TemplatePiece::SpecialToken, {}, "[CLS]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::A, {}, 0},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::B, {}, 1},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 1},
    };

    processors::TemplateProcessing proc(single_tmpl, pair_tmpl, {cls_def, sep_def});

    Encoding enc1({5}, {0}, {"hello"}, {std::nullopt}, {{0, 5}}, {0}, {1}, {}, {});
    Encoding enc2({6}, {0}, {"world"}, {std::nullopt}, {{0, 5}}, {0}, {1}, {}, {});

    auto result = proc.process(std::move(enc1), std::move(enc2), true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // CLS + hello + SEP + world + SEP = 5 tokens
    EXPECT_EQ(result->get_ids().size(), 5u);
    EXPECT_EQ(result->get_ids()[0], 2u);  // CLS
    EXPECT_EQ(result->get_ids()[1], 5u);  // hello
    EXPECT_EQ(result->get_ids()[2], 3u);  // SEP
    EXPECT_EQ(result->get_ids()[3], 6u);  // world
    EXPECT_EQ(result->get_ids()[4], 3u);  // SEP

    // type_ids: 0,0,0,1,1
    EXPECT_EQ(result->get_type_ids()[0], 0u);
    EXPECT_EQ(result->get_type_ids()[1], 0u);
    EXPECT_EQ(result->get_type_ids()[2], 0u);
    EXPECT_EQ(result->get_type_ids()[3], 1u);
    EXPECT_EQ(result->get_type_ids()[4], 1u);
}

TEST(TemplateProcessingTest, NoSpecialTokens) {
    processors::SpecialTokenDef cls_def{"[CLS]", 2, {2}, {"[CLS]"}};
    processors::SpecialTokenDef sep_def{"[SEP]", 3, {3}, {"[SEP]"}};

    std::vector<processors::TemplatePiece> single_tmpl = {
        {processors::TemplatePiece::SpecialToken, {}, "[CLS]", 0},
        {processors::TemplatePiece::Sequence, processors::TemplateSequence::A, {}, 0},
        {processors::TemplatePiece::SpecialToken, {}, "[SEP]", 0},
    };

    processors::TemplateProcessing proc(single_tmpl, {}, {cls_def, sep_def});

    Encoding enc({5, 6}, {0, 0}, {"hello", "world"},
                 {std::nullopt, std::nullopt}, {{0, 5}, {6, 11}},
                 {0, 0}, {1, 1}, {}, {});

    auto result = proc.process(std::move(enc), std::nullopt, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // No special tokens, just the encoding
    EXPECT_EQ(result->get_ids().size(), 2u);
    EXPECT_EQ(result->get_ids()[0], 5u);
    EXPECT_EQ(result->get_ids()[1], 6u);
}

TEST(TemplateProcessingTest, JsonDeserialization) {
    std::string json = R"({
        "model": {"type": "WordPiece", "vocab": {"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":5,"world":6}, "unk_token":"[UNK]"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],
            "pair": [{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],
            "special_tokens": {"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}
        },
        "added_tokens": [
            {"id":2,"content":"[CLS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":3,"content":"[SEP]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ]
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto enc = result->encode("hello world", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    // Should have CLS + hello + world + SEP = 4 tokens
    ASSERT_GE(enc->get_ids().size(), 3u);
    EXPECT_EQ(enc->get_ids().front(), 2u); // CLS
    EXPECT_EQ(enc->get_ids().back(), 3u);  // SEP
}

// === JSON Serialization tests ===

TEST(SerializationTest, WordPieceRoundTrip) {
    auto result1 = Tokenizer::from_string(kWordPieceJson);
    ASSERT_TRUE(result1.has_value()) << result1.error().message();

    auto json_str = result1->to_string(true);
    ASSERT_TRUE(json_str.has_value()) << json_str.error().message();

    auto result2 = Tokenizer::from_string(*json_str);
    ASSERT_TRUE(result2.has_value()) << result2.error().message();

    // Verify same encoding behavior
    auto enc1 = result1->encode("hello world", true);
    auto enc2 = result2->encode("hello world", true);
    ASSERT_TRUE(enc1.has_value()) << enc1.error().message();
    ASSERT_TRUE(enc2.has_value()) << enc2.error().message();
    EXPECT_EQ(enc1->get_ids(), enc2->get_ids());
}

TEST(SerializationTest, BPERoundTrip) {
    auto result1 = Tokenizer::from_string(kBPEJson);
    ASSERT_TRUE(result1.has_value()) << result1.error().message();

    auto json_str = result1->to_string(false);
    ASSERT_TRUE(json_str.has_value()) << json_str.error().message();

    auto result2 = Tokenizer::from_string(*json_str);
    ASSERT_TRUE(result2.has_value()) << result2.error().message();

    auto enc1 = result1->encode("hello", true);
    auto enc2 = result2->encode("hello", true);
    ASSERT_TRUE(enc1.has_value()) << enc1.error().message();
    ASSERT_TRUE(enc2.has_value()) << enc2.error().message();
    EXPECT_EQ(enc1->get_ids(), enc2->get_ids());
}

TEST(SerializationTest, NullComponentsRoundTrip) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1}}
    })";
    auto result1 = Tokenizer::from_string(json);
    ASSERT_TRUE(result1.has_value()) << result1.error().message();

    auto json_str = result1->to_string(true);
    ASSERT_TRUE(json_str.has_value()) << json_str.error().message();

    auto result2 = Tokenizer::from_string(*json_str);
    ASSERT_TRUE(result2.has_value()) << result2.error().message();
    EXPECT_EQ(result2->get_normalizer(), nullptr);
    EXPECT_EQ(result2->get_pre_tokenizer(), nullptr);
    EXPECT_EQ(result2->get_post_processor(), nullptr);
    EXPECT_EQ(result2->get_decoder(), nullptr);

    EXPECT_EQ(result2->token_to_id("a"), 1u);
}

TEST(SerializationTest, SaveToFile) {
    auto result = Tokenizer::from_string(kWordPieceJson);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    std::string path = "/tmp/tokenizerpp_test_save.json";
    auto save_result = result->save(path, true);
    ASSERT_TRUE(save_result.has_value()) << save_result.error().message();

    auto loaded = Tokenizer::from_file(path);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message();

    auto enc1 = result->encode("hello world", true);
    auto enc2 = loaded->encode("hello world", true);
    ASSERT_TRUE(enc1.has_value()) << enc1.error().message();
    ASSERT_TRUE(enc2.has_value()) << enc2.error().message();
    EXPECT_EQ(enc1->get_ids(), enc2->get_ids());

    // Cleanup
    std::remove(path.c_str());
}

TEST(SerializationTest, TruncationRoundTrip) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0, "a": 1, "b": 2, "c": 3}},
      "truncation": {"max_length": 2, "stride": 0, "strategy": "LongestFirst", "direction": "Right"},
      "padding": null
    })";
    auto result1 = Tokenizer::from_string(json);
    ASSERT_TRUE(result1.has_value()) << result1.error().message();

    auto json_str = result1->to_string(true);
    ASSERT_TRUE(json_str.has_value()) << json_str.error().message();

    auto result2 = Tokenizer::from_string(*json_str);
    ASSERT_TRUE(result2.has_value()) << result2.error().message();

    auto enc = result2->encode("abc", true);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    EXPECT_LE(enc->get_ids().size(), 2u);
}

// === Metaspace decoder tests ===

TEST(MetaspaceDecoderTest, Basic) {
    decoders::MetaspaceDecoder dec;

    // Simulate tokens produced by Metaspace pre-tokenizer
    std::vector<std::string> tokens = {
        "\xE2\x96\x81Hello",  // ▁Hello
        "\xE2\x96\x81world",  // ▁world
    };
    auto result = dec.decode_chain(std::move(tokens));
    ASSERT_TRUE(result.has_value()) << result.error().message();
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0], "Hello");   // leading space stripped
    EXPECT_EQ((*result)[1], " world");
}

TEST(MetaspaceDecoderTest, EmptyTokens) {
    decoders::MetaspaceDecoder dec;
    auto result = dec.decode_chain({});
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST(MetaspaceDecoderTest, NoReplacements) {
    decoders::MetaspaceDecoder dec;
    std::vector<std::string> tokens = {"Hello", "world"};
    auto result = dec.decode_chain(std::move(tokens));
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0], "Hello");
    EXPECT_EQ((*result)[1], "world");
}

TEST(MetaspaceDecoderTest, FullDecode) {
    decoders::MetaspaceDecoder dec;
    std::vector<std::string> tokens = {
        "\xE2\x96\x81Hello",  // ▁Hello
        "\xE2\x96\x81world",  // ▁world
    };
    auto result = dec.decode(std::move(tokens));
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "Hello world");
}

TEST(MetaspaceDecoderTest, JsonDeserialization) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0}},
      "decoder": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always"}
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_NE(result->get_decoder(), nullptr);
}

TEST(MetaspaceDecoderTest, SerializationRoundTrip) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0}},
      "decoder": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always"}
    })";
    auto result1 = Tokenizer::from_string(json);
    ASSERT_TRUE(result1.has_value()) << result1.error().message();

    auto json_str = result1->to_string(true);
    ASSERT_TRUE(json_str.has_value()) << json_str.error().message();

    auto result2 = Tokenizer::from_string(*json_str);
    ASSERT_TRUE(result2.has_value()) << result2.error().message();
    EXPECT_NE(result2->get_decoder(), nullptr);
}

// === FixedLength pre-tokenizer tests ===

TEST(FixedLengthTest, BasicSplit) {
    pre_tokenizers::FixedLength pt(3);
    PreTokenizedString pts(std::string("abcdefgh"));
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = pts.get_splits(OffsetReferential::Original, OffsetType::Byte);
    // "abc", "def", "gh"
    ASSERT_EQ(splits.size(), 3u);
    EXPECT_EQ(splits[0].text, "abc");
    EXPECT_EQ(splits[1].text, "def");
    EXPECT_EQ(splits[2].text, "gh");
}

TEST(FixedLengthTest, ExactMultiple) {
    pre_tokenizers::FixedLength pt(3);
    PreTokenizedString pts(std::string("abcdef"));
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = pts.get_splits(OffsetReferential::Original, OffsetType::Byte);
    ASSERT_EQ(splits.size(), 2u);
    EXPECT_EQ(splits[0].text, "abc");
    EXPECT_EQ(splits[1].text, "def");
}

TEST(FixedLengthTest, SingleChar) {
    pre_tokenizers::FixedLength pt(1);
    PreTokenizedString pts(std::string("abc"));
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = pts.get_splits(OffsetReferential::Original, OffsetType::Byte);
    ASSERT_EQ(splits.size(), 3u);
    EXPECT_EQ(splits[0].text, "a");
    EXPECT_EQ(splits[1].text, "b");
    EXPECT_EQ(splits[2].text, "c");
}

TEST(FixedLengthTest, JsonDeserialization) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]", "vocab": {"[UNK]": 0, "ab": 1, "cd": 2}},
      "pre_tokenizer": {"type": "FixedLength", "length": 2}
    })";
    auto result = Tokenizer::from_string(json);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_NE(result->get_pre_tokenizer(), nullptr);
}

// ============================================================================
// TokenizerConfig tests
// ============================================================================

TEST(TokenizerConfigTest, ParseStringTokens) {
    auto config = TokenizerConfig::from_json(R"({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    })");
    ASSERT_TRUE(config.has_value()) << config.error().message();
    EXPECT_EQ(config->bos_token, "<s>");
    EXPECT_EQ(config->eos_token, "</s>");
    EXPECT_TRUE(config->has_chat_template());
    EXPECT_TRUE(config->default_chat_template.has_value());
}

TEST(TokenizerConfigTest, ParseObjectTokens) {
    auto config = TokenizerConfig::from_json(R"({
        "bos_token": {"content": "<|begin|>", "lstrip": false, "rstrip": false},
        "eos_token": {"content": "<|end|>"}
    })");
    ASSERT_TRUE(config.has_value()) << config.error().message();
    EXPECT_EQ(config->bos_token, "<|begin|>");
    EXPECT_EQ(config->eos_token, "<|end|>");
}

TEST(TokenizerConfigTest, ParseArrayTemplates) {
    auto config = TokenizerConfig::from_json(R"({
        "chat_template": [
            {"name": "default", "template": "default tmpl"},
            {"name": "tool_use", "template": "tool tmpl"}
        ]
    })");
    ASSERT_TRUE(config.has_value()) << config.error().message();
    EXPECT_EQ(config->get_chat_template("default"), "default tmpl");
    EXPECT_EQ(config->get_chat_template("tool_use"), "tool tmpl");
    EXPECT_TRUE(config->has_chat_template());
    // "default" entry also populates default_chat_template
    EXPECT_EQ(config->default_chat_template, "default tmpl");
}

TEST(TokenizerConfigTest, NullChatTemplate) {
    auto config = TokenizerConfig::from_json(R"({
        "bos_token": "<s>",
        "chat_template": null
    })");
    ASSERT_TRUE(config.has_value()) << config.error().message();
    EXPECT_FALSE(config->has_chat_template());
}

TEST(TokenizerConfigTest, MissingFields) {
    auto config = TokenizerConfig::from_json(R"({})");
    ASSERT_TRUE(config.has_value()) << config.error().message();
    EXPECT_FALSE(config->bos_token.has_value());
    EXPECT_FALSE(config->eos_token.has_value());
    EXPECT_FALSE(config->has_chat_template());
    EXPECT_FALSE(config->add_bos_token);
    EXPECT_FALSE(config->add_eos_token);
}

TEST(TokenizerConfigTest, BooleanFlags) {
    auto config = TokenizerConfig::from_json(R"({
        "add_bos_token": true,
        "add_eos_token": true
    })");
    ASSERT_TRUE(config.has_value()) << config.error().message();
    EXPECT_TRUE(config->add_bos_token);
    EXPECT_TRUE(config->add_eos_token);
}

// ============================================================================
// Tokenizer chat template integration tests
// ============================================================================

TEST(TokenizerChatTest, ApplyCustomTemplate) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1, "hi": 2, "user": 3, "assistant": 4}}
    })";
    auto tok = Tokenizer::from_string(json);
    ASSERT_TRUE(tok.has_value()) << tok.error().message();

    auto result = tok->apply_chat_template(
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}",
        {{"user", "Hello"}, {"assistant", "Hi!"}},
        false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_NE(result->find("user: Hello"), std::string::npos);
    EXPECT_NE(result->find("assistant: Hi!"), std::string::npos);
}

TEST(TokenizerChatTest, ApplyBuiltinTemplate) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1}}
    })";
    auto tok = Tokenizer::from_string(json);
    ASSERT_TRUE(tok.has_value()) << tok.error().message();

    TokenizerConfig config;
    config.default_chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}";
    config.bos_token = "<s>";
    config.eos_token = "</s>";
    tok->with_config(std::move(config));

    EXPECT_TRUE(tok->has_chat_template());

    auto result = tok->apply_chat_template({{"user", "Hello"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "Hello");
}

TEST(TokenizerChatTest, SpecialTokensInTemplate) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1}}
    })";
    auto tok = Tokenizer::from_string(json);
    ASSERT_TRUE(tok.has_value()) << tok.error().message();

    TokenizerConfig config;
    config.bos_token = "<s>";
    config.eos_token = "</s>";
    config.default_chat_template =
        "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}{{ eos_token }}";
    tok->with_config(std::move(config));

    EXPECT_EQ(tok->bos_token(), "<s>");
    EXPECT_EQ(tok->eos_token(), "</s>");

    auto result = tok->apply_chat_template({{"user", "Hello"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "<s>Hello</s>");
}

TEST(TokenizerChatTest, NoChatTemplateError) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0}}
    })";
    auto tok = Tokenizer::from_string(json);
    ASSERT_TRUE(tok.has_value()) << tok.error().message();

    // No config set, should error
    auto result = tok->apply_chat_template({{"user", "Hi"}}, false);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().message().find("No chat template"), std::string::npos);
}

TEST(TokenizerChatTest, EncodeChat) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2}}
    })";
    auto tok = Tokenizer::from_string(json);
    ASSERT_TRUE(tok.has_value()) << tok.error().message();

    TokenizerConfig config;
    config.default_chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}";
    tok->with_config(std::move(config));

    auto enc = tok->encode_chat({{"user", "hello"}}, false);
    ASSERT_TRUE(enc.has_value()) << enc.error().message();
    EXPECT_GT(enc->len(), 0u);
}

TEST(TokenizerChatTest, ConfigAccessorsWithoutConfig) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0}}
    })";
    auto tok = Tokenizer::from_string(json);
    ASSERT_TRUE(tok.has_value()) << tok.error().message();

    EXPECT_EQ(tok->bos_token(), "");
    EXPECT_EQ(tok->eos_token(), "");
    EXPECT_FALSE(tok->has_chat_template());
    EXPECT_EQ(tok->chat_template_str(), "");
    EXPECT_EQ(tok->get_config(), nullptr);
}

TEST(TokenizerChatTest, NamedTemplates) {
    const char* json = R"({
      "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0}}
    })";
    auto tok = Tokenizer::from_string(json);
    ASSERT_TRUE(tok.has_value()) << tok.error().message();

    TokenizerConfig config;
    config.default_chat_template = "default: {% for m in messages %}{{ m['content'] }}{% endfor %}";
    config.named_chat_templates["tool_use"] = "tool: {% for m in messages %}{{ m['content'] }}{% endfor %}";
    tok->with_config(std::move(config));

    auto r1 = tok->apply_chat_template({{"user", "Hi"}}, false, "default");
    ASSERT_TRUE(r1.has_value()) << r1.error().message();
    EXPECT_NE(r1->find("default: Hi"), std::string::npos);

    auto r2 = tok->apply_chat_template({{"user", "Hi"}}, false, "tool_use");
    ASSERT_TRUE(r2.has_value()) << r2.error().message();
    EXPECT_NE(r2->find("tool: Hi"), std::string::npos);
}

// ============================================================================
// Added token splitting during encode
//
// When input text contains literal added-token strings (e.g. from chat template
// rendering), the tokenizer must match them as single tokens — not feed them
// into the BPE/Unigram model as regular sub-words.
// ============================================================================

TEST(AddedTokenEncodingTest, SpecialTokensEncodedAsSingleIds) {
    // Build a minimal WordLevel tokenizer with added tokens
    // (WordLevel maps whole words to IDs without merging/splitting)
    std::unordered_map<std::string, TokenId> vocab = {
        {"hello", 0}, {"world", 1}, {"<bos>", 2}, {"<eos>", 3},
        {"<start>", 4}, {"<end>", 5},
    };
    auto model = std::make_unique<models::WordLevel>(vocab);

    Tokenizer tok(std::move(model));

    // Register added tokens (same as what happens when loading tokenizer.json)
    tok.add_special_tokens({
        AddedToken{"<bos>", true},
        AddedToken{"<eos>", true},
        AddedToken{"<start>", true},
        AddedToken{"<end>", true},
    });

    // Encode text with embedded special tokens
    auto result = tok.encode("<bos><start>hello<end><eos>", false);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto& ids = result->get_ids();
    // Should be: <bos>=2, <start>=4, hello=0, <end>=5, <eos>=3
    ASSERT_EQ(ids.size(), 5u) << "Expected 5 tokens (3 special + 'hello' + 1 special)";
    EXPECT_EQ(ids[0], 2);  // <bos>
    EXPECT_EQ(ids[1], 4);  // <start>
    EXPECT_EQ(ids[2], 0);  // hello
    EXPECT_EQ(ids[3], 5);  // <end>
    EXPECT_EQ(ids[4], 3);  // <eos>
}

TEST(AddedTokenEncodingTest, NoAddedTokensPassThrough) {
    // Without added tokens, angle-bracket strings go through the model normally
    std::unordered_map<std::string, TokenId> vocab = {
        {"hello", 0}, {"world", 1}, {"<", 2}, {">", 3}, {"b", 4}, {"o", 5}, {"s", 6},
    };
    models::MergeMap merges;
    auto model = std::make_unique<models::BPE>(vocab, merges);
    Tokenizer tok(std::move(model));

    // No added tokens registered — "<bos>" will be split by BPE
    auto result = tok.encode("<bos>hello", false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    // "<bos>" would be split into subwords, not a single token
    EXPECT_GT(result->get_ids().size(), 2u);
}

} // namespace
} // namespace tokenizers
