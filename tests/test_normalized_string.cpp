#include <gtest/gtest.h>
#include "tokenizers/normalized_string.h"
#include "tokenizers/pattern.h"

namespace tokenizers {
namespace {

// Helper to access internals for testing via public test accessors
class NormalizedStringTest : public ::testing::Test {
protected:
    static const std::vector<std::pair<size_t, size_t>>&
    alignments(const NormalizedString& ns) { return ns.alignments(); }
    static size_t original_shift(const NormalizedString& ns) { return ns.original_shift(); }
};

// ===== Construction =====

TEST_F(NormalizedStringTest, ConstructionASCII) {
    NormalizedString ns("Hello");
    EXPECT_EQ(ns.get(), "Hello");
    EXPECT_EQ(ns.get_original(), "Hello");
    EXPECT_EQ(ns.len(), 5u);
    EXPECT_EQ(ns.len_original(), 5u);
    EXPECT_FALSE(ns.is_empty());
    // 1:1 alignment for ASCII
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 5u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 1}));
    EXPECT_EQ(a[4], (std::pair<size_t,size_t>{4, 5}));
}

TEST_F(NormalizedStringTest, ConstructionUTF8) {
    NormalizedString ns("élégant");
    EXPECT_EQ(ns.get(), "élégant");
    EXPECT_EQ(ns.len(), 9u); // é=2 bytes each
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 9u);
    // é maps bytes 0,1 → (0,2)
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[2], (std::pair<size_t,size_t>{2, 3})); // l
}

TEST_F(NormalizedStringTest, ConstructionEmpty) {
    NormalizedString ns("");
    EXPECT_TRUE(ns.is_empty());
    EXPECT_EQ(ns.len(), 0u);
    EXPECT_EQ(alignments(ns).size(), 0u);
}

TEST_F(NormalizedStringTest, DefaultConstructor) {
    NormalizedString ns;
    EXPECT_TRUE(ns.is_empty());
    EXPECT_EQ(ns.get(), "");
    EXPECT_EQ(ns.get_original(), "");
}

// ===== NFD =====

TEST_F(NormalizedStringTest, NFDAddsNewChars) {
    NormalizedString ns("élégant");
    ns.nfd();
    // é (2 bytes) → e (1) + combining acute (2) = 3 bytes
    EXPECT_EQ(ns.len(), 11u);
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 11u);
    // e + combining acute for first é
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[2], (std::pair<size_t,size_t>{0, 2}));
    // l
    EXPECT_EQ(a[3], (std::pair<size_t,size_t>{2, 3}));
    // e + combining acute for second é
    EXPECT_EQ(a[4], (std::pair<size_t,size_t>{3, 5}));
    EXPECT_EQ(a[5], (std::pair<size_t,size_t>{3, 5}));
    EXPECT_EQ(a[6], (std::pair<size_t,size_t>{3, 5}));
    // g a n t
    EXPECT_EQ(a[7], (std::pair<size_t,size_t>{5, 6}));
    EXPECT_EQ(a[8], (std::pair<size_t,size_t>{6, 7}));
    EXPECT_EQ(a[9], (std::pair<size_t,size_t>{7, 8}));
    EXPECT_EQ(a[10], (std::pair<size_t,size_t>{8, 9}));
}

TEST_F(NormalizedStringTest, NFDThenFilterCombiningMarks) {
    NormalizedString ns("élégant");
    ns.nfd();
    // Filter out combining marks (U+0300-U+036F)
    ns.filter([](char32_t c) {
        return c < 0x0300 || c > 0x036F;
    });
    EXPECT_EQ(ns.get(), "elegant");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 7u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{2, 3}));
    EXPECT_EQ(a[2], (std::pair<size_t,size_t>{3, 5}));
    EXPECT_EQ(a[3], (std::pair<size_t,size_t>{5, 6}));
    EXPECT_EQ(a[4], (std::pair<size_t,size_t>{6, 7}));
    EXPECT_EQ(a[5], (std::pair<size_t,size_t>{7, 8}));
    EXPECT_EQ(a[6], (std::pair<size_t,size_t>{8, 9}));
}

// ===== Filter =====

TEST_F(NormalizedStringTest, RemoveChars) {
    NormalizedString ns("élégant");
    ns.filter([](char32_t c) { return c != 'n'; });
    EXPECT_EQ(ns.get(), "\xC3\xA9l\xC3\xA9gat"); // élégat
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 8u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[7], (std::pair<size_t,size_t>{8, 9})); // t skipped n
}

TEST_F(NormalizedStringTest, RemoveAtBeginning) {
    NormalizedString ns("     Hello");
    ns.filter([](char32_t c) { return !std::isspace(static_cast<int>(c)); });
    EXPECT_EQ(ns.get(), "Hello");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, ns.len())), "Hello");
}

TEST_F(NormalizedStringTest, RemoveAtEnd) {
    NormalizedString ns("Hello    ");
    ns.filter([](char32_t c) { return !std::isspace(static_cast<int>(c)); });
    EXPECT_EQ(ns.get(), "Hello");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, ns.len())), "Hello");
}

TEST_F(NormalizedStringTest, RemoveAroundBothEdges) {
    NormalizedString ns("  Hello  ");
    ns.filter([](char32_t c) { return !std::isspace(static_cast<int>(c)); });
    EXPECT_EQ(ns.get(), "Hello");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, 5)), "Hello");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(1, 4)), "ell");
}

// ===== Offset conversion =====

TEST_F(NormalizedStringTest, RangeConversion) {
    NormalizedString ns("    __Hello__   ");
    ns.filter([](char32_t c) { return !std::isspace(static_cast<int>(c)); });
    ns.lowercase();

    auto hello_n = ns.convert_offsets(Range::original(6, 11));
    ASSERT_TRUE(hello_n.has_value());
    EXPECT_EQ(hello_n->first, 2u);
    EXPECT_EQ(hello_n->second, 7u);
    EXPECT_EQ(*ns.get_range(Range::normalized(hello_n->first, hello_n->second)), "hello");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(hello_n->first, hello_n->second)), "Hello");
    EXPECT_EQ(*ns.get_range(Range::original(6, 11)), "hello");
    EXPECT_EQ(*ns.get_range_original(Range::original(6, 11)), "Hello");

    // Edge cases
    EXPECT_EQ(ns.convert_offsets(Range::original(0, 0)),
              (std::optional{std::pair<size_t,size_t>{0, 0}}));
    EXPECT_EQ(ns.convert_offsets(Range::original(3, 3)),
              (std::optional{std::pair<size_t,size_t>{3, 3}}));
    EXPECT_EQ(ns.convert_offsets(Range::normalized(0, 0)),
              (std::optional{std::pair<size_t,size_t>{0, 0}}));
}

TEST_F(NormalizedStringTest, OriginalRange) {
    NormalizedString ns("Hello_______ World!");
    ns.filter([](char32_t c) { return c != '_'; });
    ns.lowercase();
    auto world_n = ns.get_range(Range::normalized(6, 11));
    EXPECT_EQ(*world_n, "world");
    auto world_o = ns.get_range_original(Range::normalized(6, 11));
    EXPECT_EQ(*world_o, "World");
}

// ===== Transform =====

TEST_F(NormalizedStringTest, AddedAroundEdges) {
    NormalizedString ns("Hello");
    ns.transform(
        {{' ', 1}, {'H', 0}, {'e', 0}, {'l', 0}, {'l', 0}, {'o', 0}, {' ', 1}},
        0);
    EXPECT_EQ(ns.get(), " Hello ");
    EXPECT_EQ(*ns.get_range_original(
        Range::normalized(1, ns.len() - 1)), "Hello");
}

TEST_F(NormalizedStringTest, AddedCharactersAlignment) {
    NormalizedString ns("\xe9\x87\x8e\xe5\x8f\xa3 No"); // 野口 No
    ns.transform(
        {
            {' ', 0}, {0x91CE, 1}, {' ', 1},  // 野 → " 野 "
            {' ', 0}, {0x53E3, 1}, {' ', 1},  // 口 → " 口 "
            {' ', 0},                          // space
            {'N', 0},                          // N
            {'o', 0},                          // o
        },
        0);
    EXPECT_EQ(ns.get(), " \xe9\x87\x8e  \xe5\x8f\xa3  No"); // " 野  口  No"
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 13u);
    // First group: 野 occupies 3 bytes in original
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 3}));
    EXPECT_EQ(a[4], (std::pair<size_t,size_t>{0, 3}));
    // Second group: 口
    EXPECT_EQ(a[5], (std::pair<size_t,size_t>{3, 6}));
    EXPECT_EQ(a[9], (std::pair<size_t,size_t>{3, 6}));
    // Space, N, o
    EXPECT_EQ(a[10], (std::pair<size_t,size_t>{6, 7}));
    EXPECT_EQ(a[11], (std::pair<size_t,size_t>{7, 8}));
    EXPECT_EQ(a[12], (std::pair<size_t,size_t>{8, 9}));
}

TEST_F(NormalizedStringTest, TransformRangeRemovingAtBeginning) {
    NormalizedString ns("Hello friend");
    ns.transform_range(Range::original(0, 4), {{'Y', 0}}, 3);
    EXPECT_EQ(ns.get(), "Yo friend");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 9u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{3, 4}));
}

TEST_F(NormalizedStringTest, TransformRangeRemovingInMiddle) {
    NormalizedString ns("Hello friend");
    ns.transform_range(
        Range::original(3, 10),
        {{'_', 0}, {'F', 0}, {'R', -2}},
        2);
    EXPECT_EQ(ns.get(), "Hel_FRnd");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 8u);
    EXPECT_EQ(a[3], (std::pair<size_t,size_t>{5, 6}));
    EXPECT_EQ(a[4], (std::pair<size_t,size_t>{6, 7}));
    EXPECT_EQ(a[5], (std::pair<size_t,size_t>{7, 8}));
    EXPECT_EQ(a[6], (std::pair<size_t,size_t>{10, 11}));
}

TEST_F(NormalizedStringTest, TransformRangeRemovingAtEnd) {
    NormalizedString ns("Hello friend");
    ns.transform_range(Range::original(5, 12), {{'_', 0}, {'F', -5}}, 0);
    EXPECT_EQ(ns.get(), "Hello_F");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 7u);
    for (size_t i = 0; i < 7; i++)
        EXPECT_EQ(a[i], (std::pair<size_t,size_t>{i, i + 1}));
}

TEST_F(NormalizedStringTest, TransformRangeAddingAtBeginning) {
    NormalizedString ns("Hello friend");
    ns.transform_range(Range::original(0, 1), {{'H', 1}, {'H', 0}}, 0);
    EXPECT_EQ(ns.get(), "HHello friend");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 13u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 0}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{0, 1}));
}

TEST_F(NormalizedStringTest, TransformRangeAddingViaEmptyRange) {
    NormalizedString ns("Hello friend");
    ns.transform_range(Range::original(0, 0), {{'H', 1}}, 0);
    EXPECT_EQ(ns.get(), "HHello friend");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 13u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 0}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{0, 1}));
}

TEST_F(NormalizedStringTest, TransformRangeAddingAsPartOfFirst) {
    NormalizedString ns("Hello friend");
    ns.transform_range(Range::original(0, 1), {{'H', 0}, {'H', 1}}, 0);
    EXPECT_EQ(ns.get(), "HHello friend");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 13u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 1}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{0, 1}));
}

TEST_F(NormalizedStringTest, TransformRangeAddingInMiddle) {
    NormalizedString ns("Hello friend");
    ns.transform_range(
        Range::original(5, 6),
        {{'_', 0}, {'m', 1}, {'y', 1}, {'_', 1}},
        0);
    EXPECT_EQ(ns.get(), "Hello_my_friend");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 15u);
    EXPECT_EQ(a[5], (std::pair<size_t,size_t>{5, 6}));
    EXPECT_EQ(a[6], (std::pair<size_t,size_t>{5, 6}));
    EXPECT_EQ(a[7], (std::pair<size_t,size_t>{5, 6}));
    EXPECT_EQ(a[8], (std::pair<size_t,size_t>{5, 6}));
}

TEST_F(NormalizedStringTest, TransformRangeAddingAtEnd) {
    NormalizedString ns("Hello friend");
    ns.transform_range(Range::original(11, 12), {{'d', 0}, {'_', 1}, {'!', 1}}, 0);
    EXPECT_EQ(ns.get(), "Hello friend_!");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 14u);
    EXPECT_EQ(a[11], (std::pair<size_t,size_t>{11, 12}));
    EXPECT_EQ(a[12], (std::pair<size_t,size_t>{11, 12}));
    EXPECT_EQ(a[13], (std::pair<size_t,size_t>{11, 12}));
}

// ===== Multi-byte transform tests =====

TEST_F(NormalizedStringTest, TransformRangeMultiByteRemoveBeginning) {
    // 𝔾𝕠𝕠𝕕 - each char is 4 bytes
    NormalizedString ns("\xF0\x9D\x94\xBE\xF0\x9D\x95\xA0\xF0\x9D\x95\xA0\xF0\x9D\x95\x95");
    EXPECT_EQ(ns.len(), 16u);
    ns.transform_range(Range::original(0, 8), {{'G', -1}}, 0);
    EXPECT_EQ(ns.get(), "G\xF0\x9D\x95\xA0\xF0\x9D\x95\x95"); // G𝕠𝕕
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 9u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 4}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{8, 12}));
}

TEST_F(NormalizedStringTest, TransformRangeMultiByteAddBeginning) {
    NormalizedString ns("\xF0\x9D\x94\xBE\xF0\x9D\x95\xA0\xF0\x9D\x95\xA0\xF0\x9D\x95\x95");
    ns.transform_range(Range::original(0, 4), {{'_', 1}, {0x1D53E, 0}}, 0);
    EXPECT_EQ(ns.get(), "_\xF0\x9D\x94\xBE\xF0\x9D\x95\xA0\xF0\x9D\x95\xA0\xF0\x9D\x95\x95"); // _𝔾𝕠𝕠𝕕
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 17u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 0}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{0, 4}));
}

// ===== NFKC =====

TEST_F(NormalizedStringTest, NfkcMathDoubleStruck) {
    // 𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘
    NormalizedString ns(
        "\xF0\x9D\x94\xBE\xF0\x9D\x95\xA0\xF0\x9D\x95\xA0\xF0\x9D\x95\x95"
        " "
        "\xF0\x9D\x95\x9E\xF0\x9D\x95\xA0\xF0\x9D\x95\xA3\xF0\x9D\x95\x9F"
        "\xF0\x9D\x95\x9A\xF0\x9D\x95\x9F\xF0\x9D\x95\x98");
    ns.nfkc();
    EXPECT_EQ(ns.get(), "Good morning");
    EXPECT_EQ(ns.len(), 12u);

    // Slice tests matching Rust
    auto orig_slice = ns.slice(Range::original(0, 4));
    ASSERT_TRUE(orig_slice.has_value());
    EXPECT_EQ(orig_slice->get(), "G");
    EXPECT_EQ(orig_slice->get_original(),
        "\xF0\x9D\x94\xBE"); // 𝔾

    auto norm_slice = ns.slice(Range::normalized(0, 4));
    ASSERT_TRUE(norm_slice.has_value());
    EXPECT_EQ(norm_slice->get(), "Good");
    EXPECT_EQ(norm_slice->get_original(),
        "\xF0\x9D\x94\xBE\xF0\x9D\x95\xA0\xF0\x9D\x95\xA0\xF0\x9D\x95\x95"); // 𝔾𝕠𝕠𝕕
}

// ===== NFKD =====

TEST_F(NormalizedStringTest, NfkdEllipsis) {
    NormalizedString ns("abc\xe2\x80\xa6"); // "abc…"
    ns.nfkd();
    EXPECT_EQ(ns.get(), "abc...");
    EXPECT_EQ(ns.len(), 6u);
}

TEST_F(NormalizedStringTest, TransformCheck) {
    // Matches Rust transform_check test
    NormalizedString ns("abc\xe2\x80\xa6"); // "abc…"
    ns.nfkd();
    std::vector<std::pair<char32_t, int>> transforms =
        {{'a', -2}, {'.', 0}, {'.', 0}, {'.', 0}};
    ns.transform(transforms, 0);
    ns.lowercase();
    EXPECT_EQ(ns.get(), "a...");
}

// ===== Lowercase / Uppercase =====

TEST_F(NormalizedStringTest, LowercaseASCII) {
    NormalizedString ns("HELLO");
    ns.lowercase();
    EXPECT_EQ(ns.get(), "hello");
    EXPECT_EQ(ns.get_original(), "HELLO");
}

TEST_F(NormalizedStringTest, UppercaseASCII) {
    NormalizedString ns("hello");
    ns.uppercase();
    EXPECT_EQ(ns.get(), "HELLO");
}

TEST_F(NormalizedStringTest, LowercaseWithAlignment) {
    NormalizedString ns("    __Hello__   ");
    ns.filter([](char32_t c) { return !std::isspace(static_cast<int>(c)); });
    ns.lowercase();
    EXPECT_EQ(ns.get(), "__hello__");
    // Check alignment still works
    EXPECT_EQ(*ns.get_range(Range::original(6, 11)), "hello");
    EXPECT_EQ(*ns.get_range_original(Range::original(6, 11)), "Hello");
}

// ===== Prepend =====

TEST_F(NormalizedStringTest, Prepend) {
    NormalizedString ns("there");
    ns.prepend("Hey ");
    EXPECT_EQ(ns.get(), "Hey there");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 9u);
    // "Hey " all share alignment with first original char
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 1}));
    EXPECT_EQ(a[3], (std::pair<size_t,size_t>{0, 1}));
    EXPECT_EQ(a[4], (std::pair<size_t,size_t>{0, 1}));
    // "here" maps to original chars
    EXPECT_EQ(a[5], (std::pair<size_t,size_t>{1, 2}));

    auto conv = ns.convert_offsets(Range::normalized(0, 4));
    ASSERT_TRUE(conv.has_value());
    EXPECT_EQ(conv->first, 0u);
    EXPECT_EQ(conv->second, 1u);
}

// ===== Append =====

TEST_F(NormalizedStringTest, Append) {
    NormalizedString ns("Hey");
    ns.append(" there");
    EXPECT_EQ(ns.get(), "Hey there");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 9u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 1}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{1, 2}));
    EXPECT_EQ(a[2], (std::pair<size_t,size_t>{2, 3}));
    // " there" all share alignment with last original char 'y'
    for (size_t i = 3; i < 9; i++)
        EXPECT_EQ(a[i], (std::pair<size_t,size_t>{2, 3}));
}

TEST_F(NormalizedStringTest, AppendAfterClear) {
    NormalizedString ns("Hello");
    EXPECT_EQ(ns.get(), "Hello");
    ns.clear();
    EXPECT_EQ(ns.get(), "");
    ns.append(" World");
    EXPECT_EQ(ns.get(), " World");
    EXPECT_EQ(ns.len_original(), 5u);
    EXPECT_EQ(ns.len(), 6u);
    EXPECT_EQ(*ns.get_range_original(Range::original(0, 5)), "Hello");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, 6)), "");
    EXPECT_EQ(*ns.get_range(Range::normalized(0, 6)), " World");
}

// ===== Strip =====

TEST_F(NormalizedStringTest, Lstrip) {
    NormalizedString ns("  This is an example  ");
    ns.lstrip();
    EXPECT_EQ(ns.get(), "This is an example  ");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, ns.len())),
              "This is an example  ");
}

TEST_F(NormalizedStringTest, Rstrip) {
    NormalizedString ns("  This is an example  ");
    ns.rstrip();
    EXPECT_EQ(ns.get(), "  This is an example");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, ns.len())),
              "  This is an example");
}

TEST_F(NormalizedStringTest, Strip) {
    NormalizedString ns("  This is an example  ");
    ns.strip();
    EXPECT_EQ(ns.get(), "This is an example");
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, ns.len())),
              "This is an example");
}

TEST_F(NormalizedStringTest, StripUnicode) {
    NormalizedString ns("  \xe4\xbd\xa0\xe5\xa5\xbd" "asa \n"); // "  你好asa \n"
    ns.strip();
    EXPECT_EQ(ns.get(), "\xe4\xbd\xa0\xe5\xa5\xbd" "asa"); // "你好asa"
    EXPECT_EQ(*ns.get_range_original(Range::normalized(0, ns.len())),
              "\xe4\xbd\xa0\xe5\xa5\xbd" "asa");
}

// ===== Split =====

TEST_F(NormalizedStringTest, SplitRemoved) {
    NormalizedString ns("The-final--countdown");
    CharPattern pat('-');
    auto result = ns.split(pat, SplitDelimiterBehavior::Removed);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    EXPECT_EQ((*result)[0].get(), "The");
    EXPECT_EQ((*result)[1].get(), "final");
    EXPECT_EQ((*result)[2].get(), "countdown");
}

TEST_F(NormalizedStringTest, SplitIsolated) {
    NormalizedString ns("The-final--countdown");
    CharPattern pat('-');
    auto result = ns.split(pat, SplitDelimiterBehavior::Isolated);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 6u);
    EXPECT_EQ((*result)[0].get(), "The");
    EXPECT_EQ((*result)[1].get(), "-");
    EXPECT_EQ((*result)[2].get(), "final");
    EXPECT_EQ((*result)[3].get(), "-");
    EXPECT_EQ((*result)[4].get(), "-");
    EXPECT_EQ((*result)[5].get(), "countdown");
}

TEST_F(NormalizedStringTest, SplitMergedWithPrevious) {
    NormalizedString ns("The-final--countdown");
    CharPattern pat('-');
    auto result = ns.split(pat, SplitDelimiterBehavior::MergedWithPrevious);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 4u);
    EXPECT_EQ((*result)[0].get(), "The-");
    EXPECT_EQ((*result)[1].get(), "final-");
    EXPECT_EQ((*result)[2].get(), "-");
    EXPECT_EQ((*result)[3].get(), "countdown");
}

TEST_F(NormalizedStringTest, SplitMergedWithNext) {
    NormalizedString ns("The-final--countdown");
    CharPattern pat('-');
    auto result = ns.split(pat, SplitDelimiterBehavior::MergedWithNext);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 4u);
    EXPECT_EQ((*result)[0].get(), "The");
    EXPECT_EQ((*result)[1].get(), "-final");
    EXPECT_EQ((*result)[2].get(), "-");
    EXPECT_EQ((*result)[3].get(), "-countdown");
}

TEST_F(NormalizedStringTest, SplitContiguous) {
    NormalizedString ns("The-final--countdown");
    CharPattern pat('-');
    auto result = ns.split(pat, SplitDelimiterBehavior::Contiguous);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 5u);
    EXPECT_EQ((*result)[0].get(), "The");
    EXPECT_EQ((*result)[1].get(), "-");
    EXPECT_EQ((*result)[2].get(), "final");
    EXPECT_EQ((*result)[3].get(), "--");
    EXPECT_EQ((*result)[4].get(), "countdown");
}

// ===== Slice =====

TEST_F(NormalizedStringTest, SliceAfterStrip) {
    NormalizedString ns("   Good Morning!   ");
    ns.strip();
    EXPECT_EQ(ns.get(), "Good Morning!");

    auto slice = ns.slice(Range::original(0, 19));
    ASSERT_TRUE(slice.has_value());
    EXPECT_EQ(*slice->get_range_original(Range::normalized(0, 4)), "Good");

    auto slice2 = ns.slice(Range::normalized(0, ns.len()));
    ASSERT_TRUE(slice2.has_value());
    EXPECT_EQ(*slice2->get_range_original(Range::normalized(0, 4)), "Good");
}

// ===== Replace =====

TEST_F(NormalizedStringTest, ReplaceChar) {
    NormalizedString ns(" Hello   friend ");
    CharPattern pat(' ');
    EXPECT_TRUE(ns.replace(pat, "_").has_value());
    EXPECT_EQ(ns.get(), "_Hello___friend_");
}

TEST_F(NormalizedStringTest, ReplaceCharSameLen) {
    NormalizedString ns("aaaab");
    CharPattern pat('a');
    EXPECT_TRUE(ns.replace(pat, "b").has_value());
    EXPECT_EQ(ns.get(), "bbbbb");
}

TEST_F(NormalizedStringTest, ReplaceStringOverlapping) {
    NormalizedString ns("aaaab");
    StringPattern pat("aaa");
    EXPECT_TRUE(ns.replace(pat, "b").has_value());
    EXPECT_EQ(ns.get(), "bab");
}

// ===== Map =====

TEST_F(NormalizedStringTest, MapToUppercase) {
    NormalizedString ns("hello");
    ns.map([](char32_t c) -> char32_t {
        return (c >= 'a' && c <= 'z') ? c - 32 : c;
    });
    EXPECT_EQ(ns.get(), "HELLO");
    EXPECT_EQ(ns.get_original(), "hello");
}

// ===== Clear =====

TEST_F(NormalizedStringTest, Clear) {
    NormalizedString ns("Hello");
    size_t old_len = ns.clear();
    EXPECT_EQ(old_len, 5u);
    EXPECT_TRUE(ns.is_empty());
    EXPECT_EQ(ns.get(), "");
    EXPECT_EQ(ns.get_original(), "Hello");
}

// ===== Offsets original =====

TEST_F(NormalizedStringTest, OffsetsOriginal) {
    NormalizedString ns("Hello");
    auto offsets = ns.offsets_original();
    EXPECT_EQ(offsets.first, 0u);
    EXPECT_EQ(offsets.second, 5u);
}

TEST_F(NormalizedStringTest, SliceOffsetsOriginal) {
    NormalizedString ns("Hello World");
    auto slice = ns.slice(Range::normalized(6, 11));
    ASSERT_TRUE(slice.has_value());
    auto offsets = slice->offsets_original();
    EXPECT_EQ(offsets.first, 6u);
    EXPECT_EQ(offsets.second, 11u);
}

// ===== Mixed operations =====

TEST_F(NormalizedStringTest, MixedAdditionAndRemoval) {
    NormalizedString ns("élégant");
    ns.nfd();
    ns.filter([](char32_t c) {
        return (c < 0x0300 || c > 0x036F) && c != 'n';
    });
    EXPECT_EQ(ns.get(), "elegat");
    auto& a = alignments(ns);
    ASSERT_EQ(a.size(), 6u);
    EXPECT_EQ(a[0], (std::pair<size_t,size_t>{0, 2}));
    EXPECT_EQ(a[1], (std::pair<size_t,size_t>{2, 3}));
    EXPECT_EQ(a[2], (std::pair<size_t,size_t>{3, 5}));
    EXPECT_EQ(a[3], (std::pair<size_t,size_t>{5, 6}));
    EXPECT_EQ(a[4], (std::pair<size_t,size_t>{6, 7}));
    EXPECT_EQ(a[5], (std::pair<size_t,size_t>{8, 9}));
}

} // namespace
} // namespace tokenizers
