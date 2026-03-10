#include <gtest/gtest.h>
#include "tokenizers/pattern.h"

namespace tokenizers {
namespace {

// Mirrors tokenizers/src/tokenizer/pattern.rs tests

using Matches = std::vector<PatternMatch>;

// ===== CharPattern =====

TEST(PatternTest, CharBasic) {
    CharPattern p('a');
    auto r = p.find_matches("aba");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, true}, {{1, 2}, false}, {{2, 3}, true}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, CharTrailing) {
    CharPattern p('a');
    auto r = p.find_matches("bbbba");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 4}, false}, {{4, 5}, true}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, CharLeading) {
    CharPattern p('a');
    auto r = p.find_matches("aabbb");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, true}, {{1, 2}, true}, {{2, 5}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, CharEmpty) {
    CharPattern p('a');
    auto r = p.find_matches("");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 0}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, CharNoMatch) {
    CharPattern p('b');
    auto r = p.find_matches("aaa");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 3}, false}};
    EXPECT_EQ(*r, expected);
}

// ===== StringPattern =====

TEST(PatternTest, StrBasic) {
    StringPattern p("a");
    auto r = p.find_matches("aba");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, true}, {{1, 2}, false}, {{2, 3}, true}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, StrMultiChar) {
    StringPattern p("ab");
    auto r = p.find_matches("aabbb");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, false}, {{1, 3}, true}, {{3, 5}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, StrMultiOccurrence) {
    StringPattern p("ab");
    auto r = p.find_matches("aabbab");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, false}, {{1, 3}, true}, {{3, 4}, false}, {{4, 6}, true}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, StrEmptyPattern) {
    StringPattern p("");
    auto r = p.find_matches("aaa");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 3}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, StrEmptyInput) {
    StringPattern p("");
    auto r = p.find_matches("");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 0}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, StrNoMatch) {
    StringPattern p("b");
    auto r = p.find_matches("aaa");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 3}, false}};
    EXPECT_EQ(*r, expected);
}

// ===== FuncPattern =====

TEST(PatternTest, FuncBasic) {
    FuncPattern p([](char32_t c) { return c == 'b'; });
    auto r = p.find_matches("aba");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, false}, {{1, 2}, true}, {{2, 3}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, FuncTrailing) {
    FuncPattern p([](char32_t c) { return c == 'b'; });
    auto r = p.find_matches("aaaab");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 4}, false}, {{4, 5}, true}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, FuncLeading) {
    FuncPattern p([](char32_t c) { return c == 'b'; });
    auto r = p.find_matches("bbaaa");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, true}, {{1, 2}, true}, {{2, 5}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, FuncEmpty) {
    FuncPattern p([](char32_t c) { return c == 'b'; });
    auto r = p.find_matches("");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 0}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, FuncNoMatch) {
    FuncPattern p([](char32_t c) { return c == 'b'; });
    auto r = p.find_matches("aaa");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 3}, false}};
    EXPECT_EQ(*r, expected);
}

// ===== RegexPattern =====

TEST(PatternTest, RegexWhitespace) {
    RegexPattern p("\\s+");
    auto r = p.find_matches("a   b");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, false}, {{1, 4}, true}, {{4, 5}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, RegexWhitespaceMultiple) {
    RegexPattern p("\\s+");
    auto r = p.find_matches("   a   b   ");
    ASSERT_TRUE(r.has_value());
    Matches expected = {
        {{0, 3}, true}, {{3, 4}, false}, {{4, 7}, true},
        {{7, 8}, false}, {{8, 11}, true}
    };
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, RegexEmpty) {
    RegexPattern p("\\s+");
    auto r = p.find_matches("");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 0}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, RegexUnicode) {
    RegexPattern p("\\s+");
    // 𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘  — each math symbol is 4 bytes, space is 1 byte
    auto r = p.find_matches("\xF0\x9D\x94\xBE\xF0\x9D\x95\xA0\xF0\x9D\x95\xA0\xF0\x9D\x95\x95 \xF0\x9D\x95\x9E\xF0\x9D\x95\xA0\xF0\x9D\x95\xA3\xF0\x9D\x95\x9F\xF0\x9D\x95\x9A\xF0\x9D\x95\x9F\xF0\x9D\x95\x98");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 16}, false}, {{16, 17}, true}, {{17, 45}, false}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, RegexNoMatch) {
    RegexPattern p("\\s+");
    auto r = p.find_matches("aaa");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 3}, false}};
    EXPECT_EQ(*r, expected);
}

// ===== InvertPattern =====

TEST(PatternTest, InvertChar) {
    auto inner = std::make_unique<CharPattern>('a');
    InvertPattern p(std::move(inner));
    auto r = p.find_matches("aba");
    ASSERT_TRUE(r.has_value());
    // Invert flips the match flags
    Matches expected = {{{0, 1}, false}, {{1, 2}, true}, {{2, 3}, false}};
    EXPECT_EQ(*r, expected);
}

} // namespace
} // namespace tokenizers
