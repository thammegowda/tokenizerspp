#include <gtest/gtest.h>
#include "tokenizers/encoding.h"
#include "tokenizers/token.h"

namespace tokenizers {
namespace {

TEST(EncodingTest, DefaultConstruction) {
    Encoding enc;
    EXPECT_TRUE(enc.is_empty());
    EXPECT_EQ(enc.len(), 0u);
    EXPECT_EQ(enc.n_sequences(), 1u);
}

TEST(EncodingTest, WithCapacity) {
    auto enc = Encoding::with_capacity(100);
    EXPECT_TRUE(enc.is_empty());
    EXPECT_EQ(enc.len(), 0u);
}

TEST(EncodingTest, FromTokens) {
    std::vector<Token> tokens = {
        {0, "Hello", {0, 5}},
        {1, "World", {6, 11}},
    };
    auto enc = Encoding::from_tokens(tokens, 0);
    EXPECT_EQ(enc.len(), 2u);
    EXPECT_FALSE(enc.is_empty());
    EXPECT_EQ(enc.get_ids(), (std::vector<TokenId>{0, 1}));
    EXPECT_EQ(enc.get_tokens()[0], "Hello");
    EXPECT_EQ(enc.get_tokens()[1], "World");
    EXPECT_EQ(enc.get_offsets()[0], (Offsets{0, 5}));
    EXPECT_EQ(enc.get_offsets()[1], (Offsets{6, 11}));
    EXPECT_EQ(enc.get_type_ids(), (std::vector<TokenId>{0, 0}));
    EXPECT_EQ(enc.get_attention_mask(), (std::vector<TokenId>{1, 1}));
    EXPECT_EQ(enc.get_special_tokens_mask(), (std::vector<TokenId>{0, 0}));
}

TEST(EncodingTest, FullConstruction) {
    Encoding enc(
        {1, 2, 3},           // ids
        {0, 0, 0},           // type_ids
        {"a", "b", "c"},     // tokens
        {0u, 0u, 0u},        // words
        {{0,1},{1,2},{2,3}},  // offsets
        {0, 0, 0},           // special_tokens_mask
        {1, 1, 1},           // attention_mask
        {},                   // overflowing
        {}                    // sequence_ranges
    );
    EXPECT_EQ(enc.len(), 3u);
    EXPECT_EQ(enc.get_ids(), (std::vector<TokenId>{1, 2, 3}));
}

TEST(EncodingTest, MergeTwo) {
    auto enc1 = Encoding::from_tokens({{0, "Hello", {0, 5}}}, 0);
    auto enc2 = Encoding::from_tokens({{1, "World", {0, 5}}}, 0);
    enc1.merge_with(std::move(enc2), false);
    EXPECT_EQ(enc1.len(), 2u);
    EXPECT_EQ(enc1.get_ids(), (std::vector<TokenId>{0, 1}));
    EXPECT_EQ(enc1.get_offsets()[0], (Offsets{0, 5}));
    EXPECT_EQ(enc1.get_offsets()[1], (Offsets{0, 5}));
}

TEST(EncodingTest, MergeGrowingOffsets) {
    auto enc1 = Encoding::from_tokens({{0, "Hello", {0, 5}}}, 0);
    auto enc2 = Encoding::from_tokens({{1, "World", {0, 5}}}, 0);
    enc1.merge_with(std::move(enc2), true);
    EXPECT_EQ(enc1.len(), 2u);
    EXPECT_EQ(enc1.get_offsets()[0], (Offsets{0, 5}));
    EXPECT_EQ(enc1.get_offsets()[1], (Offsets{5, 10}));
}

TEST(EncodingTest, MergeMultiple) {
    std::vector<Encoding> encs;
    encs.push_back(Encoding::from_tokens({{0, "a", {0, 1}}}, 0));
    encs.push_back(Encoding::from_tokens({{1, "b", {0, 1}}}, 0));
    encs.push_back(Encoding::from_tokens({{2, "c", {0, 1}}}, 0));
    auto merged = Encoding::merge(std::move(encs), false);
    EXPECT_EQ(merged.len(), 3u);
    EXPECT_EQ(merged.get_ids(), (std::vector<TokenId>{0, 1, 2}));
}

TEST(EncodingTest, TruncateRight) {
    auto enc = Encoding::from_tokens({
        {0, "a", {0, 1}}, {1, "b", {1, 2}}, {2, "c", {2, 3}},
        {3, "d", {3, 4}}, {4, "e", {4, 5}},
    }, 0);
    enc.truncate(3, 0, true);
    EXPECT_EQ(enc.len(), 3u);
    EXPECT_EQ(enc.get_ids(), (std::vector<TokenId>{0, 1, 2}));
}

TEST(EncodingTest, TruncateLeft) {
    auto enc = Encoding::from_tokens({
        {0, "a", {0, 1}}, {1, "b", {1, 2}}, {2, "c", {2, 3}},
        {3, "d", {3, 4}}, {4, "e", {4, 5}},
    }, 0);
    enc.truncate(3, 0, false);
    EXPECT_EQ(enc.len(), 3u);
    EXPECT_EQ(enc.get_ids(), (std::vector<TokenId>{2, 3, 4}));
}

TEST(EncodingTest, TruncateNoOp) {
    auto enc = Encoding::from_tokens({{0, "a", {0, 1}}}, 0);
    enc.truncate(10, 0);
    EXPECT_EQ(enc.len(), 1u);
}

TEST(EncodingTest, PadRight) {
    auto enc = Encoding::from_tokens({{0, "Hello", {0, 5}}}, 0);
    enc.pad(4, 0, 0, "[PAD]", true);
    EXPECT_EQ(enc.len(), 4u);
    EXPECT_EQ(enc.get_ids(), (std::vector<TokenId>{0, 0, 0, 0}));
    EXPECT_EQ(enc.get_tokens()[0], "Hello");
    EXPECT_EQ(enc.get_tokens()[1], "[PAD]");
    EXPECT_EQ(enc.get_attention_mask(), (std::vector<TokenId>{1, 0, 0, 0}));
}

TEST(EncodingTest, PadLeft) {
    auto enc = Encoding::from_tokens({{5, "Hello", {0, 5}}}, 0);
    enc.pad(3, 0, 0, "[PAD]", false);
    EXPECT_EQ(enc.len(), 3u);
    EXPECT_EQ(enc.get_ids(), (std::vector<TokenId>{0, 0, 5}));
    EXPECT_EQ(enc.get_tokens()[0], "[PAD]");
    EXPECT_EQ(enc.get_tokens()[2], "Hello");
    EXPECT_EQ(enc.get_attention_mask(), (std::vector<TokenId>{0, 0, 1}));
}

TEST(EncodingTest, PadNoOp) {
    auto enc = Encoding::from_tokens({{0, "a", {0, 1}}, {1, "b", {1, 2}}}, 0);
    enc.pad(2, 0, 0, "[PAD]");
    EXPECT_EQ(enc.len(), 2u);
}

TEST(EncodingTest, SetSequenceId) {
    auto enc = Encoding::from_tokens({{0, "a", {0, 1}}, {1, "b", {1, 2}}}, 0);
    enc.set_sequence_id(0);
    EXPECT_EQ(enc.n_sequences(), 1u);
}

} // namespace
} // namespace tokenizers
