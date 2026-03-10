#include <gtest/gtest.h>
#include "tokenizers/pre_tokenizers.h"
#include "tokenizers/decoders.h"
#include "tokenizers/pre_tokenized_string.h"
#include "tokenizers/normalized_string.h"
#include "tokenizers/pattern.h"

namespace tokenizers {
namespace {

// Helper to get splits as (text, offsets) pairs
auto get_splits_text(const PreTokenizedString& pts) {
    auto splits = pts.get_splits(OffsetReferential::Original, OffsetType::Byte);
    std::vector<std::pair<std::string, Offsets>> result;
    for (const auto& s : splits) {
        result.emplace_back(std::string(s.text), s.offsets);
    }
    return result;
}

// ===== BertPreTokenizer =====

TEST(BertPreTokenizerTest, Basic) {
    // Mirrors Rust test: bert.rs basic()
    pre_tokenizers::BertPreTokenizer pt;
    PreTokenizedString pts("Hey friend!     How are you?!?");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value());

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 9u);
    EXPECT_EQ(splits[0], (std::pair<std::string, Offsets>{"Hey", {0, 3}}));
    EXPECT_EQ(splits[1], (std::pair<std::string, Offsets>{"friend", {4, 10}}));
    EXPECT_EQ(splits[2], (std::pair<std::string, Offsets>{"!", {10, 11}}));
    EXPECT_EQ(splits[3], (std::pair<std::string, Offsets>{"How", {16, 19}}));
    EXPECT_EQ(splits[4], (std::pair<std::string, Offsets>{"are", {20, 23}}));
    EXPECT_EQ(splits[5], (std::pair<std::string, Offsets>{"you", {24, 27}}));
    EXPECT_EQ(splits[6], (std::pair<std::string, Offsets>{"?", {27, 28}}));
    EXPECT_EQ(splits[7], (std::pair<std::string, Offsets>{"!", {28, 29}}));
    EXPECT_EQ(splits[8], (std::pair<std::string, Offsets>{"?", {29, 30}}));
}

// ===== WhitespaceSplit =====

TEST(WhitespaceSplitTest, Basic) {
    // Mirrors Rust test: whitespace.rs whitespace_split()
    pre_tokenizers::WhitespaceSplit pt;

    {
        PreTokenizedString pts("Hey man!");
        ASSERT_TRUE(pt.pre_tokenize(pts).has_value());
        auto splits = get_splits_text(pts);
        ASSERT_EQ(splits.size(), 2u);
        EXPECT_EQ(splits[0], (std::pair<std::string, Offsets>{"Hey", {0, 3}}));
        EXPECT_EQ(splits[1], (std::pair<std::string, Offsets>{"man!", {4, 8}}));
    }

    {
        PreTokenizedString pts("Hey, man, Good?");
        ASSERT_TRUE(pt.pre_tokenize(pts).has_value());
        auto splits = get_splits_text(pts);
        ASSERT_EQ(splits.size(), 3u);
        EXPECT_EQ(splits[0], (std::pair<std::string, Offsets>{"Hey,", {0, 4}}));
        EXPECT_EQ(splits[1], (std::pair<std::string, Offsets>{"man,", {5, 9}}));
        EXPECT_EQ(splits[2], (std::pair<std::string, Offsets>{"Good?", {10, 15}}));
    }
}

// ===== Whitespace (regex-based) =====

TEST(WhitespaceTest, Basic) {
    // Mirrors Rust test: whitespace.rs basic()
    pre_tokenizers::Whitespace pt;

    {
        PreTokenizedString pts("Hey man!");
        ASSERT_TRUE(pt.pre_tokenize(pts).has_value());
        auto splits = get_splits_text(pts);
        ASSERT_EQ(splits.size(), 3u);
        EXPECT_EQ(splits[0], (std::pair<std::string, Offsets>{"Hey", {0, 3}}));
        EXPECT_EQ(splits[1], (std::pair<std::string, Offsets>{"man", {4, 7}}));
        EXPECT_EQ(splits[2], (std::pair<std::string, Offsets>{"!", {7, 8}}));
    }

    {
        PreTokenizedString pts("How are you doing?");
        ASSERT_TRUE(pt.pre_tokenize(pts).has_value());
        auto splits = get_splits_text(pts);
        ASSERT_EQ(splits.size(), 5u);
        EXPECT_EQ(splits[0], (std::pair<std::string, Offsets>{"How", {0, 3}}));
        EXPECT_EQ(splits[1], (std::pair<std::string, Offsets>{"are", {4, 7}}));
        EXPECT_EQ(splits[2], (std::pair<std::string, Offsets>{"you", {8, 11}}));
        EXPECT_EQ(splits[3], (std::pair<std::string, Offsets>{"doing", {12, 17}}));
        EXPECT_EQ(splits[4], (std::pair<std::string, Offsets>{"?", {17, 18}}));
    }

    {
        PreTokenizedString pts("\n");
        ASSERT_TRUE(pt.pre_tokenize(pts).has_value());
        auto splits = get_splits_text(pts);
        EXPECT_EQ(splits.size(), 0u);
    }
}

// ===== CharDelimiterSplit =====

TEST(CharDelimiterSplitTest, Basic) {
    pre_tokenizers::CharDelimiterSplit pt('-');
    PreTokenizedString pts("hello-world-foo");
    ASSERT_TRUE(pt.pre_tokenize(pts).has_value());
    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 3u);
    EXPECT_EQ(splits[0].first, "hello");
    EXPECT_EQ(splits[1].first, "world");
    EXPECT_EQ(splits[2].first, "foo");
}

// ===== ByteLevel =====

TEST(ByteLevelTest, BytesCharMapSize) {
    // All 256 byte values must be mapped
    auto& b2c = pre_tokenizers::ByteLevel::bytes_char_array();
    // Flat array always has 256 entries; check all are non-zero or that byte 0 maps to >= 256
    size_t count = 0;
    for (size_t i = 0; i < 256; ++i) {
        if (b2c[i] != 0 || i == 0) ++count;
    }
    // All 256 bytes should be mapped (byte 0 maps to >= 256)
    EXPECT_GE(static_cast<uint32_t>(b2c[0]), 256u);
    auto& c2b_valid = pre_tokenizers::ByteLevel::char_bytes_valid();
    size_t valid_count = 0;
    for (size_t i = 0; i < c2b_valid.size(); ++i) {
        if (c2b_valid[i]) ++valid_count;
    }
    EXPECT_EQ(valid_count, 256u);
}

TEST(ByteLevelTest, BytesCharPrintableIdentity) {
    // Printable ASCII bytes 0x21-0x7E map to themselves
    auto& b2c = pre_tokenizers::ByteLevel::bytes_char_array();
    for (uint8_t b = 0x21; b <= 0x7E; ++b) {
        EXPECT_EQ(b2c[b], static_cast<char32_t>(b))
            << "byte " << static_cast<int>(b) << " should map to itself";
    }
}

TEST(ByteLevelTest, BytesCharControlMapped) {
    // Control bytes like 0x00, 0x01 must map to 256+n
    auto& b2c = pre_tokenizers::ByteLevel::bytes_char_array();
    EXPECT_GE(static_cast<uint32_t>(b2c[0x00]), 256u);
    EXPECT_GE(static_cast<uint32_t>(b2c[0x01]), 256u);
    EXPECT_GE(static_cast<uint32_t>(b2c[0x7F]), 256u); // DEL
    EXPECT_GE(static_cast<uint32_t>(b2c[0xAD]), 256u); // soft hyphen
}

TEST(ByteLevelTest, RoundTripMapping) {
    // Every byte → char → byte round-trips
    auto& b2c = pre_tokenizers::ByteLevel::bytes_char_array();
    auto& c2b = pre_tokenizers::ByteLevel::char_bytes_array();
    for (uint32_t b = 0; b <= 255; ++b) {
        char32_t ch = b2c[static_cast<uint8_t>(b)];
        uint8_t back = c2b[ch];
        EXPECT_EQ(back, static_cast<uint8_t>(b)) << "round-trip failed for byte " << b;
    }
}

TEST(ByteLevelTest, BasicPreTokenize) {
    pre_tokenizers::ByteLevel pt(false, true, true);
    PreTokenizedString pts("Hello world");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    // GPT-2 regex should split "Hello world" into ["Hello", " world"]
    ASSERT_EQ(splits.size(), 2u);
    // After byte-level encoding, ASCII maps to itself for printable chars
    // 'H'=0x48 maps to 'H', 'e'=0x65 maps to 'e', etc.
    EXPECT_EQ(splits[0].first, "Hello");
    // ' ' (0x20) is a control-range byte, maps to Ġ (U+0120)
    // 'w'=0x77 maps to 'w', etc.
    // The space character 0x20 maps to char 256+... let's check
    auto& b2c = pre_tokenizers::ByteLevel::bytes_char_array();
    std::string expected_space;
    // Encode b2c[0x20] as UTF-8
    char32_t space_char = b2c[0x20];
    // space_char should be U+0120 = Ġ
    EXPECT_EQ(space_char, 0x0120u);
}

TEST(ByteLevelTest, PrefixSpace) {
    pre_tokenizers::ByteLevel pt(true, true, true);
    PreTokenizedString pts("Hello");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    // With add_prefix_space=true, input becomes " Hello" → GPT-2 regex → [" Hello"]
    // Then byte-level encode: ' '→Ġ, rest→identity
    ASSERT_EQ(splits.size(), 1u);
    // The normalized text should start with Ġ (U+0120 = 0xC4 0xA0 in UTF-8)
    EXPECT_EQ(splits[0].first[0], '\xC4');
    EXPECT_EQ(splits[0].first[1], '\xA0');
}

TEST(ByteLevelTest, NoRegex) {
    pre_tokenizers::ByteLevel pt(false, true, false);
    PreTokenizedString pts("Hello world");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    // No regex splitting → single split with byte-level encoding
    ASSERT_EQ(splits.size(), 1u);
}

TEST(ByteLevelTest, DecoderRoundTrip) {
    // Encode "Hello world" with ByteLevel pre-tokenizer, then decode back
    pre_tokenizers::ByteLevel pt(false, true, true);
    PreTokenizedString pts("Hello world");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    std::vector<std::string> tokens;
    for (auto& [text, _] : splits) tokens.push_back(text);

    decoders::ByteLevelDecoder dec;
    auto decoded = dec.decode(std::move(tokens));
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    EXPECT_EQ(*decoded, "Hello world");
}

TEST(ByteLevelTest, DecoderRoundTripUnicode) {
    // Test with non-ASCII input
    pre_tokenizers::ByteLevel pt(false, true, false);
    std::string input = "café";
    PreTokenizedString pts(input);
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    std::vector<std::string> tokens;
    for (auto& [text, _] : splits) tokens.push_back(text);

    decoders::ByteLevelDecoder dec;
    auto decoded = dec.decode(std::move(tokens));
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    EXPECT_EQ(*decoded, input);
}

// ===== Metaspace =====

TEST(MetaspaceTest, Basic) {
    // Mirrors Rust test: basic()
    pre_tokenizers::Metaspace pt(U'\u2581', pre_tokenizers::Metaspace::PrependScheme::Always, true);
    PreTokenizedString pts("Hey friend!");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 2u);
    EXPECT_EQ(splits[0].first, "\xE2\x96\x81Hey");       // ▁Hey
    EXPECT_EQ(splits[0].second, (Offsets{0, 3}));
    EXPECT_EQ(splits[1].first, "\xE2\x96\x81""friend!"); // ▁friend!
    EXPECT_EQ(splits[1].second, (Offsets{3, 11}));
}

TEST(MetaspaceTest, MultipleSpaces) {
    pre_tokenizers::Metaspace pt(U'\u2581', pre_tokenizers::Metaspace::PrependScheme::Always, true);
    PreTokenizedString pts("Hey   friend!");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 4u);
    EXPECT_EQ(splits[0].first, "\xE2\x96\x81Hey");
    EXPECT_EQ(splits[1].first, "\xE2\x96\x81");
    EXPECT_EQ(splits[2].first, "\xE2\x96\x81");
    EXPECT_EQ(splits[3].first, "\xE2\x96\x81""friend!");
}

TEST(MetaspaceTest, NoSplit) {
    pre_tokenizers::Metaspace pt(U'\u2581', pre_tokenizers::Metaspace::PrependScheme::Always, false);
    PreTokenizedString pts("Hey friend!");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 1u);
    EXPECT_EQ(splits[0].first, "\xE2\x96\x81Hey\xE2\x96\x81""friend!");
}

TEST(MetaspaceTest, PrependNever) {
    pre_tokenizers::Metaspace pt(U'\u2581', pre_tokenizers::Metaspace::PrependScheme::Never, true);
    PreTokenizedString pts("Hey friend!");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 2u);
    EXPECT_EQ(splits[0].first, "Hey");
    EXPECT_EQ(splits[1].first, "\xE2\x96\x81""friend!");
}

TEST(MetaspaceTest, CustomReplacement) {
    pre_tokenizers::Metaspace pt('_', pre_tokenizers::Metaspace::PrependScheme::Always, true);
    PreTokenizedString pts("Hey friend!");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 2u);
    EXPECT_EQ(splits[0].first, "_Hey");
    EXPECT_EQ(splits[1].first, "_friend!");
}

// ===== SplitPreTokenizer =====

TEST(SplitPreTokenizerTest, StringPattern) {
    auto pattern = std::make_unique<StringPattern>(" ");
    pre_tokenizers::SplitPreTokenizer pt(std::move(pattern),
                                          SplitDelimiterBehavior::Removed, false);
    PreTokenizedString pts("Hey, man!");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 2u);
    EXPECT_EQ(splits[0].first, "Hey,");
    EXPECT_EQ(splits[1].first, "man!");
}

TEST(SplitPreTokenizerTest, RegexPattern) {
    auto pattern = std::make_unique<RegexPattern>(R"(\s+)");
    pre_tokenizers::SplitPreTokenizer pt(std::move(pattern),
                                          SplitDelimiterBehavior::Removed, false);
    PreTokenizedString pts("How are you doing?");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 4u);
    EXPECT_EQ(splits[0], (std::pair<std::string, Offsets>{"How", {0, 3}}));
    EXPECT_EQ(splits[1], (std::pair<std::string, Offsets>{"are", {4, 7}}));
    EXPECT_EQ(splits[2], (std::pair<std::string, Offsets>{"you", {8, 11}}));
    EXPECT_EQ(splits[3], (std::pair<std::string, Offsets>{"doing?", {12, 18}}));
}

TEST(SplitPreTokenizerTest, Invert) {
    auto pattern = std::make_unique<RegexPattern>(R"(\w+|[^\w\s]+)");
    pre_tokenizers::SplitPreTokenizer pt(std::move(pattern),
                                          SplitDelimiterBehavior::Removed, true);
    PreTokenizedString pts("How are you doing?");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 5u);
    EXPECT_EQ(splits[0], (std::pair<std::string, Offsets>{"How", {0, 3}}));
    EXPECT_EQ(splits[1], (std::pair<std::string, Offsets>{"are", {4, 7}}));
    EXPECT_EQ(splits[2], (std::pair<std::string, Offsets>{"you", {8, 11}}));
    EXPECT_EQ(splits[3], (std::pair<std::string, Offsets>{"doing", {12, 17}}));
    EXPECT_EQ(splits[4], (std::pair<std::string, Offsets>{"?", {17, 18}}));
}

TEST(SplitPreTokenizerTest, Isolated) {
    auto pattern = std::make_unique<StringPattern>(" ");
    pre_tokenizers::SplitPreTokenizer pt(std::move(pattern),
                                          SplitDelimiterBehavior::Isolated, false);
    PreTokenizedString pts("How are you");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 5u);
    EXPECT_EQ(splits[0].first, "How");
    EXPECT_EQ(splits[1].first, " ");
    EXPECT_EQ(splits[2].first, "are");
    EXPECT_EQ(splits[3].first, " ");
    EXPECT_EQ(splits[4].first, "you");
}

// ===== UnicodeScripts =====

TEST(UnicodeScriptsTest, Basic) {
    // Mirrors Rust test: splits CJK from punctuation from Latin
    pre_tokenizers::UnicodeScripts pt;
    PreTokenizedString pts("どこで生れ。Yes");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 3u);
    EXPECT_EQ(splits[0].first, "どこで生れ");
    EXPECT_EQ(splits[0].second, (Offsets{0, 15}));
    EXPECT_EQ(splits[1].first, "。");
    EXPECT_EQ(splits[1].second, (Offsets{15, 18}));
    EXPECT_EQ(splits[2].first, "Yes");
    EXPECT_EQ(splits[2].second, (Offsets{18, 21}));
}

TEST(UnicodeScriptsTest, SpacesIncludedInEveryScript) {
    // Mirrors Rust test: spaces are joiners
    pre_tokenizers::UnicodeScripts pt;
    PreTokenizedString pts("Apples are りんご 林檎");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 2u);
    EXPECT_EQ(splits[0].first, "Apples are ");
    EXPECT_EQ(splits[0].second, (Offsets{0, 11}));
    EXPECT_EQ(splits[1].first, "りんご 林檎");
    EXPECT_EQ(splits[1].second, (Offsets{11, 27}));
}

TEST(UnicodeScriptsTest, PureAscii) {
    pre_tokenizers::UnicodeScripts pt;
    PreTokenizedString pts("Hello World");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    auto splits = get_splits_text(pts);
    ASSERT_EQ(splits.size(), 1u);
    EXPECT_EQ(splits[0].first, "Hello World");
}

TEST(UnicodeScriptsTest, Empty) {
    pre_tokenizers::UnicodeScripts pt;
    PreTokenizedString pts("");
    auto result = pt.pre_tokenize(pts);
    ASSERT_TRUE(result.has_value()) << result.error().message();
}

} // namespace
} // namespace tokenizers
