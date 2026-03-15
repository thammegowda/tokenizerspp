#include <gtest/gtest.h>
#include "tokenizers/models.h"
#include "tokenizers/decoders.h"

namespace tokenizers {
namespace {

TEST(WordPieceTest, BasicTokenize) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"[CLS]", 1}, {"[SEP]", 2},
        {"want", 3}, {"##want", 4}, {"##ed", 5}, {"wa", 6}, {"un", 7},
        {"runn", 8}, {"##ing", 9},
    };

    models::WordPiece wp(vocab);

    auto result = wp.tokenize("unwanted");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0].value, "un");
    EXPECT_EQ((*result)[0].id, 7u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 2}));
    EXPECT_EQ((*result)[1].value, "##want");
    EXPECT_EQ((*result)[1].id, 4u);
    EXPECT_EQ((*result)[1].offsets, (Offsets{2, 6}));
    EXPECT_EQ((*result)[2].value, "##ed");
    EXPECT_EQ((*result)[2].id, 5u);
    EXPECT_EQ((*result)[2].offsets, (Offsets{6, 8}));
}

TEST(WordPieceTest, UnknownToken) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1},
    };

    models::WordPiece wp(vocab);

    auto result = wp.tokenize("xyz");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "[UNK]");
    EXPECT_EQ((*result)[0].id, 0u);
}

TEST(WordPieceTest, WholeWord) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };

    models::WordPiece wp(vocab);

    auto result = wp.tokenize("hello");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "hello");
    EXPECT_EQ((*result)[0].id, 1u);
}

TEST(WordPieceTest, MaxInputChars) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"a", 1},
    };

    models::WordPiece wp(vocab, "[UNK]", "##", 5);

    // String with more than 5 chars → UNK
    auto result = wp.tokenize("abcdefgh");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "[UNK]");
}

TEST(WordPieceTest, TokenToId) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"##world", 2},
    };
    models::WordPiece wp(vocab);

    EXPECT_EQ(wp.token_to_id("hello"), 1u);
    EXPECT_EQ(wp.token_to_id("##world"), 2u);
    EXPECT_EQ(wp.token_to_id("[UNK]"), 0u);
    EXPECT_EQ(wp.token_to_id("notfound"), std::nullopt);
}

TEST(WordPieceTest, IdToToken) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1},
    };
    models::WordPiece wp(vocab);

    EXPECT_EQ(wp.id_to_token(0), "[UNK]");
    EXPECT_EQ(wp.id_to_token(1), "hello");
    EXPECT_EQ(wp.id_to_token(99), std::nullopt);
}

TEST(WordPieceTest, VocabSize) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    models::WordPiece wp(vocab);

    EXPECT_EQ(wp.get_vocab_size(), 3u);
}

// ===== WordPiece Decoder =====

TEST(WordPieceDecoderTest, Basic) {
    decoders::WordPieceDecoder dec;
    auto result = dec.decode({"Hello", "##llo", "world"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Hellollo world");
}

TEST(WordPieceDecoderTest, SingleToken) {
    decoders::WordPieceDecoder dec;
    auto result = dec.decode({"Hello"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Hello");
}

TEST(WordPieceDecoderTest, AllSubwords) {
    decoders::WordPieceDecoder dec;
    auto result = dec.decode({"un", "##want", "##ed"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "unwanted");
}

// ===== BPE Model =====

using models::MergeMap;

TEST(BPETest, BasicMerges) {
    // vocab: a=0, b=1, ab=2
    // merges: (0,1) → (rank=0, new_id=2)
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b", 1}, {"ab", 2}, {"<unk>", 3},
    };
    MergeMap merges = {{{0, 1}, {0, 2}}};
    models::BPE bpe(vocab, merges, "<unk>");
    auto result = bpe.tokenize("ab");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "ab");
    EXPECT_EQ((*result)[0].id, 2u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 2}));
}

TEST(BPETest, UnknownToken) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"<unk>", 1},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges, "<unk>");
    auto result = bpe.tokenize("b");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "<unk>");
    EXPECT_EQ((*result)[0].id, 1u);
}

TEST(BPETest, MultipleMerges) {
    // vocab: a=0, b=1, c=2, ab=3, abc=4
    // merges: (0,1)→(rank=0, 3), (3,2)→(rank=1, 4)
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b", 1}, {"c", 2}, {"ab", 3}, {"abc", 4},
    };
    MergeMap merges = {{{0, 1}, {0, 3}}, {{3, 2}, {1, 4}}};
    models::BPE bpe(vocab, merges);
    auto result = bpe.tokenize("abc");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "abc");
    EXPECT_EQ((*result)[0].id, 4u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 3}));
}

TEST(BPETest, ByteOffsets) {
    // vocab: a=0, b=1, c=2, ab=3
    // merges: (0,1)→(0,3) — "ab" merges but "c" stays separate
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b", 1}, {"c", 2}, {"ab", 3},
    };
    MergeMap merges = {{{0, 1}, {0, 3}}};
    models::BPE bpe(vocab, merges);
    auto result = bpe.tokenize("abc");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].value, "ab");
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 2}));
    EXPECT_EQ((*result)[1].value, "c");
    EXPECT_EQ((*result)[1].offsets, (Offsets{2, 3}));
}

TEST(BPETest, NoMerges) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"h", 0}, {"e", 1}, {"l", 2}, {"o", 3},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges);
    auto result = bpe.tokenize("hello");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 5u);
    EXPECT_EQ((*result)[0].value, "h");
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 1}));
    EXPECT_EQ((*result)[4].value, "o");
    EXPECT_EQ((*result)[4].offsets, (Offsets{4, 5}));
}

TEST(BPETest, EmptyInput) {
    std::unordered_map<std::string, TokenId> vocab = {{"a", 0}};
    MergeMap merges;
    models::BPE bpe(vocab, merges);
    auto result = bpe.tokenize("");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST(BPETest, TokenToId) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b", 1}, {"ab", 2},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges);
    EXPECT_EQ(bpe.token_to_id("a"), 0u);
    EXPECT_EQ(bpe.token_to_id("ab"), 2u);
    EXPECT_EQ(bpe.token_to_id("xyz"), std::nullopt);
}

TEST(BPETest, IdToToken) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b", 1},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges);
    EXPECT_EQ(bpe.id_to_token(0), "a");
    EXPECT_EQ(bpe.id_to_token(1), "b");
    EXPECT_EQ(bpe.id_to_token(99), std::nullopt);
}

TEST(BPETest, VocabSize) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b", 1}, {"ab", 2},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges);
    EXPECT_EQ(bpe.get_vocab_size(), 3u);
}

TEST(BPETest, FuseUnk) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"<unk>", 1},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges, "<unk>", std::nullopt, std::nullopt,
                    /*fuse_unk=*/true);
    // "xyz" → three unknowns fused into one
    auto result = bpe.tokenize("xyz");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].id, 1u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 3}));
}

TEST(BPETest, ContinuingSubwordPrefix) {
    // With continuing_subword_prefix="##", non-first chars look up "##<char>"
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"##b", 1}, {"##c", 2},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges, std::nullopt, "##");
    auto result = bpe.tokenize("abc");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0].value, "a");
    EXPECT_EQ((*result)[0].id, 0u);
    EXPECT_EQ((*result)[1].value, "##b");
    EXPECT_EQ((*result)[1].id, 1u);
    EXPECT_EQ((*result)[2].value, "##c");
    EXPECT_EQ((*result)[2].id, 2u);
}

TEST(BPETest, EndOfWordSuffix) {
    // Last char gets suffix appended
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b</w>", 1},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges, std::nullopt, std::nullopt, "</w>");
    auto result = bpe.tokenize("ab");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].value, "a");
    EXPECT_EQ((*result)[0].id, 0u);
    EXPECT_EQ((*result)[1].value, "b</w>");
    EXPECT_EQ((*result)[1].id, 1u);
}

TEST(BPETest, ByteFallback) {
    // Char 'ñ' (U+00F1) = 0xC3 0xB1 in UTF-8
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"<0xC3>", 1}, {"<0xB1>", 2},
    };
    MergeMap merges;
    models::BPE bpe(vocab, merges, std::nullopt, std::nullopt, std::nullopt,
                    false, /*byte_fallback=*/true);
    auto result = bpe.tokenize("ñ");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].id, 1u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 1}));
    EXPECT_EQ((*result)[1].id, 2u);
    EXPECT_EQ((*result)[1].offsets, (Offsets{1, 2}));
}

TEST(BPETest, IgnoreMerges) {
    // If ignore_merges=true and whole word is in vocab, skip merging
    std::unordered_map<std::string, TokenId> vocab = {
        {"a", 0}, {"b", 1}, {"ab", 2},
    };
    MergeMap merges = {{{0, 1}, {0, 2}}};
    models::BPE bpe(vocab, merges, std::nullopt, std::nullopt, std::nullopt,
                    false, false, /*ignore_merges=*/true);
    auto result = bpe.tokenize("ab");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "ab");
    EXPECT_EQ((*result)[0].id, 2u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 2}));
}

// ===== WordLevel Model =====

TEST(WordLevelTest, BasicLookup) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    models::WordLevel wl(vocab);

    auto result = wl.tokenize("hello");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "hello");
    EXPECT_EQ((*result)[0].id, 1u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 5}));
}

TEST(WordLevelTest, UnknownToken) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1},
    };
    models::WordLevel wl(vocab);

    auto result = wl.tokenize("xyz");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "[UNK]");
    EXPECT_EQ((*result)[0].id, 0u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 3}));
}

TEST(WordLevelTest, MissingUnkToken) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"hello", 0},
    };
    models::WordLevel wl(vocab);

    auto result = wl.tokenize("xyz");
    ASSERT_FALSE(result.has_value());
}

TEST(WordLevelTest, EmptyInput) {
    std::unordered_map<std::string, TokenId> vocab = {{"a", 0}};
    models::WordLevel wl(vocab);

    auto result = wl.tokenize("");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST(WordLevelTest, TokenToId) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    models::WordLevel wl(vocab);

    EXPECT_EQ(wl.token_to_id("hello"), 1u);
    EXPECT_EQ(wl.token_to_id("world"), 2u);
    EXPECT_EQ(wl.token_to_id("notfound"), std::nullopt);
}

TEST(WordLevelTest, IdToToken) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1},
    };
    models::WordLevel wl(vocab);

    EXPECT_EQ(wl.id_to_token(0), "[UNK]");
    EXPECT_EQ(wl.id_to_token(1), "hello");
    EXPECT_EQ(wl.id_to_token(99), std::nullopt);
}

TEST(WordLevelTest, VocabSize) {
    std::unordered_map<std::string, TokenId> vocab = {
        {"[UNK]", 0}, {"hello", 1}, {"world", 2},
    };
    models::WordLevel wl(vocab);
    EXPECT_EQ(wl.get_vocab_size(), 3u);
}

// ===== BPE Decoder =====

TEST(BPEDecoderTest, BasicSuffix) {
    decoders::BPEDecoder dec("</w>");
    auto result = dec.decode({"Hello</w>", "World</w>"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Hello World");
}

TEST(BPEDecoderTest, LastTokenNoSuffix) {
    decoders::BPEDecoder dec("</w>");
    auto result = dec.decode({"Hello</w>", "World"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Hello World");
}

TEST(BPEDecoderTest, LastTokenWithSuffix) {
    decoders::BPEDecoder dec("</w>");
    // Last token's suffix is replaced with "" (not space)
    auto result = dec.decode({"Hello</w>"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Hello");
}

TEST(BPEDecoderTest, NoSuffix) {
    decoders::BPEDecoder dec("</w>");
    auto result = dec.decode({"Hello", "World"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "HelloWorld");
}

// ===== ByteFallback Decoder =====

TEST(ByteFallbackDecoderTest, NormalTokens) {
    decoders::ByteFallbackDecoder dec;
    auto result = dec.decode({"Hey", "friend!"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Heyfriend!");
}

TEST(ByteFallbackDecoderTest, SingleByte) {
    decoders::ByteFallbackDecoder dec;
    auto result = dec.decode({"<0x61>"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "a");
}

TEST(ByteFallbackDecoderTest, InvalidUtf8SingleByte) {
    decoders::ByteFallbackDecoder dec;
    auto result = dec.decode({"<0xE5>"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "\xEF\xBF\xBD"); // U+FFFD
}

TEST(ByteFallbackDecoderTest, InvalidUtf8TwoBytes) {
    decoders::ByteFallbackDecoder dec;
    auto result = dec.decode({"<0xE5>", "<0x8f>"});
    ASSERT_TRUE(result.has_value());
    // Two invalid bytes → two U+FFFD
    EXPECT_EQ(*result, "\xEF\xBF\xBD\xEF\xBF\xBD");
}

TEST(ByteFallbackDecoderTest, ValidUtf8ThreeBytes) {
    decoders::ByteFallbackDecoder dec;
    // 叫 = E5 8F AB
    auto result = dec.decode({"<0xE5>", "<0x8f>", "<0xab>"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "叫");
}

TEST(ByteFallbackDecoderTest, BytesThenNormal) {
    decoders::ByteFallbackDecoder dec;
    auto result = dec.decode({"<0xE5>", "<0x8f>", "<0xab>", "a"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "叫a");
}

TEST(ByteFallbackDecoderTest, InvalidThenNormal) {
    decoders::ByteFallbackDecoder dec;
    auto result = dec.decode({"<0xE5>", "<0x8f>", "a"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "\xEF\xBF\xBD" "\xEF\xBF\xBD" "a");
}

// ===== CTC Decoder =====

TEST(CTCDecoderTest, HandmadeSample) {
    decoders::CTCDecoder dec;
    auto result = dec.decode_chain(
        {"<pad>", "<pad>", "h", "e", "e", "l", "l", "<pad>", "l", "o", "o", "o", "<pad>"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<std::string>{"h", "e", "l", "l", "o"}));
}

TEST(CTCDecoderTest, WithDelimiter) {
    decoders::CTCDecoder dec;
    auto result = dec.decode_chain(
        {"<pad>", "<pad>", "h", "e", "e", "l", "l", "<pad>", "l", "o", "o", "o",
         "<pad>", "<pad>", "|", "<pad>", "w", "o", "o", "o", "r", "<pad>", "<pad>",
         "l", "l", "d", "<pad>", "<pad>", "<pad>", "<pad>"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result,
              (std::vector<std::string>{"h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"}));
}

// ===== Fuse Decoder =====

TEST(FuseDecoderTest, Basic) {
    decoders::FuseDecoder dec;
    auto result = dec.decode_chain({"Hey", " friend!"});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0], "Hey friend!");
}

TEST(FuseDecoderTest, Empty) {
    decoders::FuseDecoder dec;
    auto result = dec.decode_chain({});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0], "");
}

// ===== Strip Decoder =====

TEST(StripDecoderTest, StripLeft) {
    decoders::StripDecoder dec('H', 1, 0);
    auto result = dec.decode_chain({"Hey", " friend!", "HHH"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<std::string>{"ey", " friend!", "HH"}));
}

TEST(StripDecoderTest, StripRight) {
    decoders::StripDecoder dec('y', 0, 1);
    auto result = dec.decode_chain({"Hey", " friend!"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<std::string>{"He", " friend!"}));
}

TEST(StripDecoderTest, StripBoth) {
    decoders::StripDecoder dec('_', 2, 2);
    auto result = dec.decode_chain({"__hello__", "_world_"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<std::string>{"hello", "world"}));
}

// ===== Unigram Model =====

TEST(UnigramTest, BasicEncode) {
    // Mirrors Rust doctest: encode("abcdacdxx") == ["abcd", "a", "cd", "xx"]
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0},
        {"a", 0.0},
        {"b", 0.0},
        {"c", 0.0},
        {"d", 0.0},
        {"cd", 1.0},
        {"ab", 2.0},
        {"abc", 5.0},
        {"abcd", 10.0},
    };

    models::Unigram model(vocab, 0);
    auto result = model.tokenize("abcdacdxx");
    ASSERT_TRUE(result.has_value()) << result.error().message();
    ASSERT_EQ(result->size(), 4u);
    EXPECT_EQ((*result)[0].value, "abcd");
    EXPECT_EQ((*result)[1].value, "a");
    EXPECT_EQ((*result)[2].value, "cd");
    EXPECT_EQ((*result)[3].value, "xx");
}

TEST(UnigramTest, SingleToken) {
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0},
        {"abcd", 10.0},
    };
    models::Unigram model(vocab, 0);
    auto result = model.tokenize("abcd");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].value, "abcd");
    EXPECT_EQ((*result)[0].id, 1u);
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 4}));
}

TEST(UnigramTest, UnknownTokenFused) {
    // UNK tokens should be fused by default
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0},
        {"a", 0.0},
    };
    models::Unigram model(vocab, 0);
    auto result = model.tokenize("axyz");
    ASSERT_TRUE(result.has_value());
    // 'a' is known, 'xyz' are all unknown and should be fused
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].value, "a");
    EXPECT_EQ((*result)[0].id, 1u);
    EXPECT_EQ((*result)[1].value, "xyz");
    EXPECT_EQ((*result)[1].id, 0u);
}

TEST(UnigramTest, EmptyInput) {
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0},
    };
    models::Unigram model(vocab, 0);
    auto result = model.tokenize("");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST(UnigramTest, ByteFallback) {
    // 'ñ' (U+00F1) = 0xC3 0xB1 in UTF-8
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0},
        {"a", 0.0},
        {"<0xC3>", 0.0},
        {"<0xB1>", 0.0},
    };
    models::Unigram model(vocab, 0, true);
    auto result = model.tokenize("añ");
    ASSERT_TRUE(result.has_value());
    // 'a' → direct match, 'ñ' → byte fallback <0xC3> + <0xB1>
    ASSERT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0].value, "a");
    EXPECT_EQ((*result)[0].id, 1u);
    EXPECT_EQ((*result)[1].value, "<0xC3>");
    EXPECT_EQ((*result)[1].id, 2u);
    EXPECT_EQ((*result)[2].value, "<0xB1>");
    EXPECT_EQ((*result)[2].id, 3u);
}

TEST(UnigramTest, TokenToId) {
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0}, {"hello", 1.0}, {"world", 2.0},
    };
    models::Unigram model(vocab, 0);
    EXPECT_EQ(model.token_to_id("hello"), 1u);
    EXPECT_EQ(model.token_to_id("world"), 2u);
    EXPECT_EQ(model.token_to_id("<unk>"), 0u);
    EXPECT_EQ(model.token_to_id("notfound"), std::nullopt);
}

TEST(UnigramTest, IdToToken) {
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0}, {"hello", 1.0},
    };
    models::Unigram model(vocab, 0);
    EXPECT_EQ(model.id_to_token(0), "<unk>");
    EXPECT_EQ(model.id_to_token(1), "hello");
    EXPECT_EQ(model.id_to_token(99), std::nullopt);
}

TEST(UnigramTest, VocabSize) {
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0}, {"a", 1.0}, {"b", 2.0},
    };
    models::Unigram model(vocab, 0);
    EXPECT_EQ(model.get_vocab_size(), 3u);
}

TEST(UnigramTest, Encode2) {
    // Mirrors Rust test_encode2
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0},
        {"ab", 0.0},
        {"cd", -0.1},
        {"abc", -0.2},
        {"a", -0.3},
        {"b", -0.4},
        {"c", -0.5},
        {"ABC", -0.5},
        {"abcdabcd", 20.0},
        {"q", 20.5},
        {"r", 20.5},
        {"qr", -0.5},
    };

    models::Unigram model(vocab, 0);
    {
        auto r = model.tokenize("abc");
        ASSERT_TRUE(r.has_value());
        ASSERT_EQ(r->size(), 1u);
        EXPECT_EQ((*r)[0].value, "abc");
    }
    {
        auto r = model.tokenize("abcd");
        ASSERT_TRUE(r.has_value());
        ASSERT_EQ(r->size(), 2u);
        EXPECT_EQ((*r)[0].value, "ab");
        EXPECT_EQ((*r)[1].value, "cd");
    }
    {
        auto r = model.tokenize("abcc");
        ASSERT_TRUE(r.has_value());
        ASSERT_EQ(r->size(), 2u);
        EXPECT_EQ((*r)[0].value, "abc");
        EXPECT_EQ((*r)[1].value, "c");
    }
    {
        // UNK fused: "AB" are both unknown, fuse to "AB"
        auto r = model.tokenize("AB");
        ASSERT_TRUE(r.has_value());
        ASSERT_EQ(r->size(), 1u);
        EXPECT_EQ((*r)[0].value, "AB");
    }
}

TEST(UnigramTest, CorrectOffsets) {
    std::vector<std::pair<std::string, double>> vocab = {
        {"<unk>", 0.0},
        {"a", 0.0},
        {"b", 0.0},
        {"ab", 2.0},
    };
    models::Unigram model(vocab, 0);
    auto result = model.tokenize("aba");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].value, "ab");
    EXPECT_EQ((*result)[0].offsets, (Offsets{0, 2}));
    EXPECT_EQ((*result)[1].value, "a");
    EXPECT_EQ((*result)[1].offsets, (Offsets{2, 3}));
}

} // namespace
} // namespace tokenizers
