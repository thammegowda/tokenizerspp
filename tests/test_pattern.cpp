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

TEST(PatternTest, RegexEndAnchorMatchesEmptyInput) {
    RegexPattern p("$");
    auto r = p.find_matches("");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 0}, true}};
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

TEST(PatternTest, RegexLookaheadTrailingWhitespace) {
    // GPT-2 / Llama 3 / Qwen whitespace tail: \s+(?!\S)|\s+
    // For "a   b" (spaces at 1..4), \s+(?!\S) matches [1, 3] ("  ") so [3, 4] (" ")
    // remains attached to "b" when combined with " ?\p{L}+".
    RegexPattern gpt2(R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)");
    auto r = gpt2.find_matches("Hello   world");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 5}, true}, {{5, 7}, true}, {{7, 13}, true}};
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, RegexLlama3CaseInsensitiveAndBoundedDigits) {
    RegexPattern llama3(
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)");
    auto r = llama3.find_matches("IT'S 12345 中文!");
    ASSERT_TRUE(r.has_value());
    // "IT" [0,2], "'S" [2,4], " " [4,5], "123" [5,8], "45" [8,10], " 中文" [10,17], "!" [17,18]
    Matches expected = {
        {{0, 2}, true}, {{2, 4}, true}, {{4, 5}, true}, {{5, 8}, true},
        {{8, 10}, true}, {{10, 17}, true}, {{17, 18}, true}
    };
    EXPECT_EQ(*r, expected);
}

TEST(PatternTest, RegexBracedHexAndScripts) {
    RegexPattern ws(R"([\x{0009}-\x{000D}\x{0020}\x{3000}]+)");
    auto r = ws.find_matches("a\u3000 b");
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 1}, false}, {{1, 5}, true}, {{5, 6}, false}};
    EXPECT_EQ(*r, expected);

    RegexPattern han(R"(\p{Han}+)");
    auto r2 = han.find_matches("ab漢字cd");
    ASSERT_TRUE(r2.has_value());
    Matches expected2 = {{{0, 2}, false}, {{2, 8}, true}, {{8, 10}, false}};
    EXPECT_EQ(*r2, expected2);
}

TEST(PatternTest, RegexLinearTimeAndNullableLoops) {
    RegexPattern tricky("(a+)+b");
    std::string input(64, 'a');
    input.push_back('c');
    auto r = tricky.find_matches(input);
    ASSERT_TRUE(r.has_value());
    Matches expected = {{{0, 65}, false}};
    EXPECT_EQ(*r, expected);

    RegexPattern nullable("(a*)*b");
    auto r2 = nullable.find_matches("aaab");
    ASSERT_TRUE(r2.has_value());
    Matches expected2 = {{{0, 4}, true}};
    EXPECT_EQ(*r2, expected2);

    RegexPattern alt_nullable("(|a)+b");
    auto r3 = alt_nullable.find_matches("aab");
    ASSERT_TRUE(r3.has_value());
    Matches expected3 = {{{0, 3}, true}};
    EXPECT_EQ(*r3, expected3);
}

TEST(PatternTest, RegexBracketClassesAndCaseInsensitivity) {
    // Leading ']' and leading/trailing '-' inside [...]
    RegexPattern edge_chars(R"([]a-]+)");
    auto r1 = edge_chars.find_matches("x]a-y");
    ASSERT_TRUE(r1.has_value());
    Matches exp1 = {{{0, 1}, false}, {{1, 4}, true}, {{4, 5}, false}};
    EXPECT_EQ(*r1, exp1);

    // Negated bracket with leading ']'
    RegexPattern neg_edge(R"([^]0-9]+)");
    auto r2 = neg_edge.find_matches("ab]12cd");
    ASSERT_TRUE(r2.has_value());
    Matches exp2 = {{{0, 2}, true}, {{2, 5}, false}, {{5, 7}, true}};
    EXPECT_EQ(*r2, exp2);

    // Case-insensitive ASCII + Unicode (Cyrillic & Latin-1)
    RegexPattern icase(R"((?i:café|[а-я]+))");
    std::string cyr_input = "CAFÉ Привет!";
    auto r3 = icase.find_matches(cyr_input);
    ASSERT_TRUE(r3.has_value());
    // "CAFÉ" = 5 bytes [0,5], " " = [5,6], "Привет" = 12 bytes [6,18], "!" = [18,19]
    Matches exp3 = {{{0, 5}, true}, {{5, 6}, false}, {{6, 18}, true}, {{18, 19}, false}};
    EXPECT_EQ(*r3, exp3);

    // Flag toggling (?i)...(?-i:...)
    RegexPattern toggle(R"((?i)ab(?-i:cd))");
    auto r4 = toggle.find_matches("ABcd ABCD");
    ASSERT_TRUE(r4.has_value());
    Matches exp4 = {{{0, 4}, true}, {{4, 9}, false}};
    EXPECT_EQ(*r4, exp4);
}

TEST(PatternTest, RegexQuantifiersGreedyAndLazy) {
    RegexPattern lazy_tag(R"(<.+?>)");
    auto r1 = lazy_tag.find_matches("<a><b>");
    ASSERT_TRUE(r1.has_value());
    Matches exp1 = {{{0, 3}, true}, {{3, 6}, true}};
    EXPECT_EQ(*r1, exp1);

    RegexPattern greedy_tag(R"(<.+>)");
    auto r2 = greedy_tag.find_matches("<a><b>");
    ASSERT_TRUE(r2.has_value());
    Matches exp2 = {{{0, 6}, true}};
    EXPECT_EQ(*r2, exp2);

    RegexPattern exact_and_min(R"(\d{2}-\d{3,})");
    auto r3 = exact_and_min.find_matches("1-22 12-3456 99-12");
    ASSERT_TRUE(r3.has_value());
    Matches exp3 = {{{0, 5}, false}, {{5, 12}, true}, {{12, 18}, false}};
    EXPECT_EQ(*r3, exp3);

    RegexPattern lazy_bounded(R"(a{1,3}?)");
    auto r4 = lazy_bounded.find_matches("aaa");
    ASSERT_TRUE(r4.has_value());
    Matches exp4 = {{{0, 1}, true}, {{1, 2}, true}, {{2, 3}, true}};
    EXPECT_EQ(*r4, exp4);
}

TEST(PatternTest, RegexAnchorsDotallAndWordBoundaries) {
    RegexPattern start_anchor(R"(^hello)");
    EXPECT_EQ(*start_anchor.find_matches("hello hello"), (Matches{{{0, 5}, true}, {{5, 11}, false}}));
    EXPECT_EQ(*start_anchor.find_matches(" hello"), (Matches{{{0, 6}, false}}));

    RegexPattern multiline(R"((?m:^[a-z]+$))");
    auto rm = multiline.find_matches("123\nabc\ndef");
    ASSERT_TRUE(rm.has_value());
    Matches expm = {{{0, 4}, false}, {{4, 7}, true}, {{7, 8}, false}, {{8, 11}, true}};
    EXPECT_EQ(*rm, expm);

    RegexPattern dot_default(R"(a.b)");
    EXPECT_EQ(*dot_default.find_matches("a\nb"), (Matches{{{0, 3}, false}}));
    RegexPattern dot_all(R"((?s:a.b))");
    EXPECT_EQ(*dot_all.find_matches("a\nb"), (Matches{{{0, 3}, true}}));

    RegexPattern word_bound(R"(\b\w+\b)");
    auto rw = word_bound.find_matches("hi, мир!");
    ASSERT_TRUE(rw.has_value());
    // "hi" [0,2], ", " [2,4], "мир" [4,10], "!" [10,11]
    Matches expw = {{{0, 2}, true}, {{2, 4}, false}, {{4, 10}, true}, {{10, 11}, false}};
    EXPECT_EQ(*rw, expw);
}

TEST(PatternTest, RegexLookaheadAndLookbehind) {
    RegexPattern pos_la(R"(\w+(?=:))");
    auto r1 = pos_la.find_matches("host:8080 path");
    ASSERT_TRUE(r1.has_value());
    Matches exp1 = {{{0, 4}, true}, {{4, 14}, false}};
    EXPECT_EQ(*r1, exp1);

    // Multi-byte UTF-8 fixed-length lookbehind: € is 3 bytes (0xE2 0x82 0xAC)
    RegexPattern pos_lb(R"((?<=€)\d+)");
    std::string prices = "$100 €250";
    auto r2 = pos_lb.find_matches(prices);
    ASSERT_TRUE(r2.has_value());
    // "$100 €" = 8 bytes [0,8], "250" = [8,11]
    Matches exp2 = {{{0, 8}, false}, {{8, 11}, true}};
    EXPECT_EQ(*r2, exp2);

    RegexPattern neg_lb(R"((?<!un)able)");
    auto r3 = neg_lb.find_matches("unable cap_able");
    ASSERT_TRUE(r3.has_value());
    Matches exp3 = {{{0, 11}, false}, {{11, 15}, true}};
    EXPECT_EQ(*r3, exp3);
}

TEST(PatternTest, RegexUnicodeCategoriesScriptsAndInvalidUtf8) {
    RegexPattern cats(R"(\p{Sc}\p{Nd}+)");
    auto r1 = cats.find_matches("Price: ₹42!");
    ASSERT_TRUE(r1.has_value());
    // "Price: " = 7 bytes, "₹" = 3 bytes, "42" = 2 bytes -> [7, 12], "!" = [12, 13]
    Matches exp1 = {{{0, 7}, false}, {{7, 12}, true}, {{12, 13}, false}};
    EXPECT_EQ(*r1, exp1);

    // Qwen / OmniVoice single-digit \p{N} pre-tokenizer pattern
    RegexPattern qwen(
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)");
    auto rq = qwen.find_matches("v2026\r\n  ok");
    ASSERT_TRUE(rq.has_value());
    // "v" [0,1], "2" [1,2], "0" [2,3], "2" [3,4], "6" [4,5], "\r\n" [5,7], " " [7,8], " ok" [8,11]
    Matches expq = {
        {{0, 1}, true}, {{1, 2}, true}, {{2, 3}, true}, {{3, 4}, true},
        {{4, 5}, true}, {{5, 7}, true}, {{7, 8}, true}, {{8, 11}, true}
    };
    EXPECT_EQ(*rq, expq);

    // Malformed UTF-8 bytes must not crash or hang
    std::string bad_utf8 = "ok\xFF\xC0\xAF" "end";
    auto rb = qwen.find_matches(bad_utf8);
    ASSERT_TRUE(rb.has_value());
    size_t covered = 0;
    for (const auto& [off, _] : *rb) {
        EXPECT_EQ(off.first, covered);
        covered = off.second;
    }
    EXPECT_EQ(covered, bad_utf8.size());
}

TEST(PatternTest, RegexRejectsInvalidSyntax) {
    EXPECT_THROW(RegexPattern("(abc"), std::runtime_error);
    EXPECT_THROW(RegexPattern("abc)"), std::runtime_error);
    EXPECT_THROW(RegexPattern("[abc"), std::runtime_error);
    EXPECT_THROW(RegexPattern("[z-a]"), std::runtime_error);
    EXPECT_THROW(RegexPattern("*abc"), std::runtime_error);
    EXPECT_THROW(RegexPattern("a{5,2}"), std::runtime_error);
    EXPECT_THROW(RegexPattern("a{2000}"), std::runtime_error);
    EXPECT_THROW(RegexPattern("abc\\"), std::runtime_error);
    EXPECT_THROW(RegexPattern(R"(\p{NotARealProperty})"), std::runtime_error);
    EXPECT_THROW(RegexPattern(R"((?<=a+)b)"), std::runtime_error);
    EXPECT_THROW(RegexPattern(R"(^+)"), std::runtime_error);
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
