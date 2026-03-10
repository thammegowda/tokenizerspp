#pragma once
/// @file tokenizers/pattern.h
/// Pattern interface for splitting NormalizedStrings.
/// Mirrors tokenizers/src/tokenizer/pattern.rs

#include "tokenizers/common.h"
#include "tokenizers/error.h"
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tokenizers {

/// A match result: (byte_offsets, is_match).
using PatternMatch = std::pair<Offsets, bool>;

/// Abstract pattern used to split strings.
class Pattern {
public:
    virtual ~Pattern() = default;

    /// Find all matches in `inside`, returning contiguous ordered slices
    /// that cover the whole string.
    [[nodiscard]] virtual Result<std::vector<PatternMatch>>
    find_matches(std::string_view inside) const = 0;
};

/// Inverts the is_match flag of a wrapped Pattern.
class InvertPattern : public Pattern {
public:
    explicit InvertPattern(std::unique_ptr<Pattern> inner)
        : inner_(std::move(inner)) {}

    [[nodiscard]] Result<std::vector<PatternMatch>>
    find_matches(std::string_view inside) const override;

private:
    std::unique_ptr<Pattern> inner_;
};

/// Pattern that matches a single Unicode code point.
class CharPattern : public Pattern {
public:
    explicit CharPattern(char32_t ch) : ch_(ch) {}

    [[nodiscard]] Result<std::vector<PatternMatch>>
    find_matches(std::string_view inside) const override;

private:
    char32_t ch_;
};

/// Pattern that matches a literal string.
class StringPattern : public Pattern {
public:
    explicit StringPattern(std::string pattern) : pattern_(std::move(pattern)) {}

    [[nodiscard]] Result<std::vector<PatternMatch>>
    find_matches(std::string_view inside) const override;

    [[nodiscard]] const std::string& get_pattern() const { return pattern_; }

private:
    std::string pattern_;
};

/// Pattern that matches using a predicate on each Unicode code point.
class FuncPattern : public Pattern {
public:
    explicit FuncPattern(std::function<bool(char32_t)> pred)
        : pred_(std::move(pred)) {}

    [[nodiscard]] Result<std::vector<PatternMatch>>
    find_matches(std::string_view inside) const override;

private:
    std::function<bool(char32_t)> pred_;
};

/// Pattern that matches using a regex (PCRE2, supports lookahead/lookbehind).
class RegexPattern : public Pattern {
public:
    explicit RegexPattern(const std::string& pattern);
    ~RegexPattern() override;

    RegexPattern(const RegexPattern&) = delete;
    RegexPattern& operator=(const RegexPattern&) = delete;
    RegexPattern(RegexPattern&&) noexcept;
    RegexPattern& operator=(RegexPattern&&) noexcept;

    [[nodiscard]] Result<std::vector<PatternMatch>>
    find_matches(std::string_view inside) const override;

    [[nodiscard]] const std::string& get_pattern() const { return pattern_str_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string pattern_str_;
};

} // namespace tokenizers
