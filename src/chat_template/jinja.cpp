/// @file chat_template/jinja.cpp
/// Jinja2 template engine: token-based lexer, recursive-descent parser,
/// variant-based AST, table-driven evaluator.

#include "chat_template/jinja.h"
#include "tokenizers/chat_template.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tokenizers::jinja {

// ════════════════════════════════════════════════════════
// String → enum resolution (called at parse time)
// ════════════════════════════════════════════════════════

TestOp string_to_test_op(const std::string& s) {
    static const std::unordered_map<std::string, TestOp> map = {
        {"defined",   TestOp::Defined},
        {"undefined", TestOp::Undefined},
        {"none",      TestOp::None},
        {"true",      TestOp::True},
        {"false",     TestOp::False},
        {"boolean",   TestOp::Boolean},
        {"string",    TestOp::String},
        {"number",    TestOp::Number},
        {"integer",   TestOp::Integer},
        {"float",     TestOp::Float},
        {"mapping",   TestOp::Mapping},
        {"iterable",  TestOp::Iterable},
        {"sequence",  TestOp::Sequence},
        {"callable",  TestOp::Callable},
        {"even",      TestOp::Even},
        {"odd",       TestOp::Odd},
        {"eq",        TestOp::Eq},
        {"ne",        TestOp::Ne},
        {"equalto",   TestOp::Eq},
    };
    auto it = map.find(s);
    if (it != map.end()) return it->second;
    throw std::runtime_error("Unknown test: " + s);
}

FilterId string_to_filter_id(const std::string& s) {
    static const std::unordered_map<std::string, FilterId> map = {
        {"trim",       FilterId::Trim},
        {"length",     FilterId::Length},
        {"count",      FilterId::Length},
        {"default",    FilterId::Default},
        {"d",          FilterId::Default},
        {"first",      FilterId::First},
        {"last",       FilterId::Last},
        {"upper",      FilterId::Upper},
        {"lower",      FilterId::Lower},
        {"title",      FilterId::Title},
        {"join",       FilterId::Join},
        {"list",       FilterId::List},
        {"int",        FilterId::Int},
        {"float",      FilterId::Float},
        {"string",     FilterId::String},
        {"tojson",     FilterId::ToJson},
        {"replace",    FilterId::Replace},
        {"map",        FilterId::Map},
        {"selectattr", FilterId::SelectAttr},
        {"batch",      FilterId::Batch},
        {"reverse",    FilterId::Reverse},
        {"sort",       FilterId::Sort},
        {"reject",     FilterId::Reject},
        {"rejectattr", FilterId::RejectAttr},
        {"select",     FilterId::Select},
        {"abs",        FilterId::Abs},
        {"round",      FilterId::Round},
        {"truncate",   FilterId::Truncate},
        {"indent",     FilterId::Indent},
        {"capitalize", FilterId::Capitalize},
        {"unique",     FilterId::Unique},
        {"dictsort",   FilterId::DictSort},
        {"items",      FilterId::Items},
    };
    auto it = map.find(s);
    if (it != map.end()) return it->second;
    return FilterId::Unknown;
}

// ════════════════════════════════════════════════════════
// Section 1: Template Lexer
// ════════════════════════════════════════════════════════

std::vector<TemplateToken> tokenize_template(std::string_view src) {
    std::vector<TemplateToken> tokens;
    size_t pos = 0;

    while (pos < src.size()) {
        size_t var_pos = src.find("{{", pos);
        size_t block_pos = src.find("{%", pos);
        size_t comment_pos = src.find("{#", pos);

        size_t next = std::min({var_pos, block_pos, comment_pos});
        if (next == std::string_view::npos) {
            if (pos < src.size()) {
                tokens.push_back({TemplateTokenType::Text, std::string(src.substr(pos)), false, false});
            }
            break;
        }

        if (next > pos) {
            tokens.push_back({TemplateTokenType::Text, std::string(src.substr(pos, next - pos)), false, false});
        }

        if (next == comment_pos && comment_pos != std::string_view::npos) {
            size_t end = src.find("#}", next + 2);
            if (end == std::string_view::npos) {
                tokens.push_back({TemplateTokenType::Text, std::string(src.substr(next)), false, false});
                break;
            }
            pos = end + 2;
            continue;
        }

        if (next == var_pos && var_pos != std::string_view::npos) {
            bool trim_left = (next + 2 < src.size() && src[next + 2] == '-');
            size_t expr_start = next + 2 + (trim_left ? 1 : 0);
            size_t end = src.find("}}", expr_start);
            if (end == std::string_view::npos) {
                tokens.push_back({TemplateTokenType::Text, std::string(src.substr(next)), false, false});
                break;
            }
            bool trim_right = (end > 0 && src[end - 1] == '-');
            size_t expr_end = trim_right ? end - 1 : end;
            std::string expr(src.substr(expr_start, expr_end - expr_start));
            tokens.push_back({TemplateTokenType::VarExpr, expr, trim_left, trim_right});
            pos = end + 2;
        } else {
            bool trim_left = (next + 2 < src.size() && src[next + 2] == '-');
            size_t stmt_start = next + 2 + (trim_left ? 1 : 0);
            size_t end = src.find("%}", stmt_start);
            if (end == std::string_view::npos) {
                tokens.push_back({TemplateTokenType::Text, std::string(src.substr(next)), false, false});
                break;
            }
            bool trim_right = (end > 0 && src[end - 1] == '-');
            size_t stmt_end = trim_right ? end - 1 : end;
            std::string stmt(src.substr(stmt_start, stmt_end - stmt_start));
            tokens.push_back({TemplateTokenType::BlockExpr, stmt, trim_left, trim_right});
            pos = end + 2;
        }
    }

    // Apply whitespace trimming
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i].type != TemplateTokenType::Text) {
            if (tokens[i].trim_left && i > 0 && tokens[i - 1].type == TemplateTokenType::Text) {
                auto& t = tokens[i - 1].value;
                auto end = t.find_last_not_of(" \t\n\r");
                if (end == std::string::npos) {
                    t.clear();
                } else {
                    t.erase(end + 1);
                }
            }
            if (tokens[i].trim_right && i + 1 < tokens.size() && tokens[i + 1].type == TemplateTokenType::Text) {
                auto& t = tokens[i + 1].value;
                auto start = t.find_first_not_of(" \t\n\r");
                if (start == std::string::npos) {
                    t.clear();
                } else {
                    t.erase(0, start);
                }
            }
        }
    }

    return tokens;
}

// ════════════════════════════════════════════════════════
// Section 2: Expression Lexer
// ════════════════════════════════════════════════════════

namespace {

std::vector<Token> lex_expression(std::string_view src) {
    std::vector<Token> tokens;
    size_t pos = 0;

    auto skip_ws = [&]() {
        while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos])))
            pos++;
    };

    static const std::unordered_map<std::string, TokenKind> keywords = {
        {"true", TokenKind::KW_True},   {"True", TokenKind::KW_True},
        {"false", TokenKind::KW_False}, {"False", TokenKind::KW_False},
        {"none", TokenKind::KW_None},   {"None", TokenKind::KW_None},
        {"null", TokenKind::KW_None},
        {"and", TokenKind::KW_And},     {"or", TokenKind::KW_Or},
        {"not", TokenKind::KW_Not},     {"in", TokenKind::KW_In},
        {"is", TokenKind::KW_Is},
        {"if", TokenKind::KW_If},       {"else", TokenKind::KW_Else},
        {"for", TokenKind::KW_For},     {"set", TokenKind::KW_Set},
        {"endfor", TokenKind::KW_Endfor}, {"endif", TokenKind::KW_Endif},
        {"elif", TokenKind::KW_Elif},
    };

    while (true) {
        skip_ws();
        if (pos >= src.size()) break;
        char c = src[pos];

        // String literals
        if (c == '\'' || c == '"') {
            char quote = c;
            pos++;
            std::string s;
            while (pos < src.size() && src[pos] != quote) {
                if (src[pos] == '\\' && pos + 1 < src.size()) {
                    pos++;
                    switch (src[pos]) {
                    case 'n':  s += '\n'; break;
                    case 't':  s += '\t'; break;
                    case '\\': s += '\\'; break;
                    case '\'': s += '\''; break;
                    case '"':  s += '"';  break;
                    default:   s += '\\'; s += src[pos]; break;
                    }
                } else {
                    s += src[pos];
                }
                pos++;
            }
            if (pos < src.size()) pos++; // closing quote
            tokens.push_back({TokenKind::String, s, json(s)});
            continue;
        }

        // Number literals
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t start = pos;
            while (pos < src.size() && std::isdigit(static_cast<unsigned char>(src[pos]))) pos++;
            bool is_float = false;
            if (pos < src.size() && src[pos] == '.') {
                is_float = true;
                pos++;
                while (pos < src.size() && std::isdigit(static_cast<unsigned char>(src[pos]))) pos++;
            }
            std::string text(src.substr(start, pos - start));
            if (is_float)
                tokens.push_back({TokenKind::Float, text, json(std::stod(text))});
            else
                tokens.push_back({TokenKind::Integer, text, json(static_cast<int64_t>(std::stoll(text)))});
            continue;
        }

        // Identifiers and keywords
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = pos;
            while (pos < src.size() &&
                   (std::isalnum(static_cast<unsigned char>(src[pos])) || src[pos] == '_'))
                pos++;
            std::string text(src.substr(start, pos - start));
            auto it = keywords.find(text);
            if (it != keywords.end())
                tokens.push_back({it->second, text, {}});
            else
                tokens.push_back({TokenKind::Ident, text, {}});
            continue;
        }

        // Two-character operators (check before single-char)
        if (pos + 1 < src.size()) {
            char c2 = src[pos + 1];
            if (c == '*' && c2 == '*') { tokens.push_back({TokenKind::DoubleStar, "**", {}}); pos += 2; continue; }
            if (c == '/' && c2 == '/') { tokens.push_back({TokenKind::DoubleSlash, "//", {}}); pos += 2; continue; }
            if (c == '=' && c2 == '=') { tokens.push_back({TokenKind::Eq, "==", {}}); pos += 2; continue; }
            if (c == '!' && c2 == '=') { tokens.push_back({TokenKind::NotEq, "!=", {}}); pos += 2; continue; }
            if (c == '<' && c2 == '=') { tokens.push_back({TokenKind::LtEq, "<=", {}}); pos += 2; continue; }
            if (c == '>' && c2 == '=') { tokens.push_back({TokenKind::GtEq, ">=", {}}); pos += 2; continue; }
        }

        // Single-character operators and punctuation
        switch (c) {
        case '+': tokens.push_back({TokenKind::Plus,     "+", {}}); pos++; continue;
        case '-': tokens.push_back({TokenKind::Minus,    "-", {}}); pos++; continue;
        case '*': tokens.push_back({TokenKind::Star,     "*", {}}); pos++; continue;
        case '/': tokens.push_back({TokenKind::Slash,    "/", {}}); pos++; continue;
        case '%': tokens.push_back({TokenKind::Percent,  "%", {}}); pos++; continue;
        case '~': tokens.push_back({TokenKind::Tilde,    "~", {}}); pos++; continue;
        case '<': tokens.push_back({TokenKind::Lt,       "<", {}}); pos++; continue;
        case '>': tokens.push_back({TokenKind::Gt,       ">", {}}); pos++; continue;
        case '=': tokens.push_back({TokenKind::Assign,   "=", {}}); pos++; continue;
        case '(': tokens.push_back({TokenKind::LParen,   "(", {}}); pos++; continue;
        case ')': tokens.push_back({TokenKind::RParen,   ")", {}}); pos++; continue;
        case '[': tokens.push_back({TokenKind::LBracket, "[", {}}); pos++; continue;
        case ']': tokens.push_back({TokenKind::RBracket, "]", {}}); pos++; continue;
        case '{': tokens.push_back({TokenKind::LBrace,   "{", {}}); pos++; continue;
        case '}': tokens.push_back({TokenKind::RBrace,   "}", {}}); pos++; continue;
        case '.': tokens.push_back({TokenKind::Dot,      ".", {}}); pos++; continue;
        case ',': tokens.push_back({TokenKind::Comma,    ",", {}}); pos++; continue;
        case ':': tokens.push_back({TokenKind::Colon,    ":", {}}); pos++; continue;
        case '|': tokens.push_back({TokenKind::Pipe,     "|", {}}); pos++; continue;
        default:
            throw std::runtime_error(
                std::string("Unexpected character '") + c + "' at position " +
                std::to_string(pos) + " in expression: " + std::string(src));
        }
    }

    tokens.push_back({TokenKind::Eof, "", {}});
    return tokens;
}

// ════════════════════════════════════════════════════════
// Section 3: Expression Parser
// ════════════════════════════════════════════════════════

template<typename T>
static ExprPtr make_expr(T&& t) {
    return std::make_unique<ExprNode>(std::forward<T>(t));
}

struct ExprParser {
    const std::vector<Token>& tokens;
    size_t pos = 0;

    const Token& peek() const { return tokens[pos]; }

    Token advance() { return std::move(const_cast<Token&>(tokens[pos++])); }

    bool at(TokenKind k) const { return tokens[pos].kind == k; }

    bool match(TokenKind k) {
        if (at(k)) { pos++; return true; }
        return false;
    }

    Token expect(TokenKind k) {
        if (!at(k)) {
            throw std::runtime_error(
                "Expected token kind " + std::to_string(static_cast<int>(k)) +
                " but got '" + tokens[pos].text + "'");
        }
        return advance();
    }

    // ---- Precedence chain (low to high) ----
    // parse_expression → parse_ternary → parse_or → parse_and →
    // parse_not → parse_compare → parse_concat → parse_add →
    // parse_mul → parse_unary → parse_pow → parse_filter_chain →
    // parse_postfix → parse_primary

    ExprPtr parse_expression() { return parse_ternary(); }

    ExprPtr parse_ternary() {
        auto expr = parse_or();
        if (at(TokenKind::KW_If)) {
            advance();
            auto cond = parse_or();
            ExprPtr alt;
            if (match(TokenKind::KW_Else)) {
                alt = parse_ternary();
            }
            return make_expr(CondExpr{std::move(expr), std::move(cond), std::move(alt)});
        }
        return expr;
    }

    ExprPtr parse_or() {
        auto left = parse_and();
        while (match(TokenKind::KW_Or)) {
            auto right = parse_and();
            left = make_expr(BinaryExpr{BinOp::Or, std::move(left), std::move(right)});
        }
        return left;
    }

    ExprPtr parse_and() {
        auto left = parse_not();
        while (match(TokenKind::KW_And)) {
            auto right = parse_not();
            left = make_expr(BinaryExpr{BinOp::And, std::move(left), std::move(right)});
        }
        return left;
    }

    ExprPtr parse_not() {
        if (match(TokenKind::KW_Not)) {
            auto operand = parse_not();
            return make_expr(UnaryExpr{UnOp::Not, std::move(operand)});
        }
        return parse_compare();
    }

    // Parse test name after 'is' (can be a keyword or identifier)
    std::string parse_test_name() {
        if (at(TokenKind::KW_None))  { advance(); return "none"; }
        if (at(TokenKind::KW_True))  { advance(); return "true"; }
        if (at(TokenKind::KW_False)) { advance(); return "false"; }
        if (at(TokenKind::Ident))    { return advance().text; }
        throw std::runtime_error("Expected test name after 'is'");
    }

    ExprPtr parse_compare() {
        auto left = parse_concat();
        while (true) {
            if (at(TokenKind::Eq))    { advance(); auto r = parse_concat(); left = make_expr(BinaryExpr{BinOp::Eq, std::move(left), std::move(r)}); }
            else if (at(TokenKind::NotEq)) { advance(); auto r = parse_concat(); left = make_expr(BinaryExpr{BinOp::NotEq, std::move(left), std::move(r)}); }
            else if (at(TokenKind::Lt))    { advance(); auto r = parse_concat(); left = make_expr(BinaryExpr{BinOp::Lt,  std::move(left), std::move(r)}); }
            else if (at(TokenKind::Gt))    { advance(); auto r = parse_concat(); left = make_expr(BinaryExpr{BinOp::Gt,  std::move(left), std::move(r)}); }
            else if (at(TokenKind::LtEq))  { advance(); auto r = parse_concat(); left = make_expr(BinaryExpr{BinOp::LtEq, std::move(left), std::move(r)}); }
            else if (at(TokenKind::GtEq))  { advance(); auto r = parse_concat(); left = make_expr(BinaryExpr{BinOp::GtEq, std::move(left), std::move(r)}); }
            else if (at(TokenKind::KW_Not) && pos + 1 < tokens.size() &&
                     tokens[pos + 1].kind == TokenKind::KW_In) {
                advance(); advance(); // consume 'not' 'in'
                auto r = parse_concat();
                left = make_expr(BinaryExpr{BinOp::NotIn, std::move(left), std::move(r)});
            }
            else if (at(TokenKind::KW_In)) {
                advance();
                auto r = parse_concat();
                left = make_expr(BinaryExpr{BinOp::In, std::move(left), std::move(r)});
            }
            else if (at(TokenKind::KW_Is)) {
                advance();
                bool negated = false;
                if (match(TokenKind::KW_Not)) negated = true;
                auto test_name = parse_test_name();
                left = make_expr(TestExpr{std::move(left), string_to_test_op(test_name), negated});
            }
            else break;
        }
        return left;
    }

    ExprPtr parse_concat() {
        auto left = parse_add();
        while (match(TokenKind::Tilde)) {
            auto right = parse_add();
            left = make_expr(BinaryExpr{BinOp::Concat, std::move(left), std::move(right)});
        }
        return left;
    }

    ExprPtr parse_add() {
        auto left = parse_mul();
        while (true) {
            if (at(TokenKind::Plus))  { advance(); auto r = parse_mul(); left = make_expr(BinaryExpr{BinOp::Add, std::move(left), std::move(r)}); }
            else if (at(TokenKind::Minus)) { advance(); auto r = parse_mul(); left = make_expr(BinaryExpr{BinOp::Sub, std::move(left), std::move(r)}); }
            else break;
        }
        return left;
    }

    ExprPtr parse_mul() {
        auto left = parse_unary();
        while (true) {
            if (at(TokenKind::Star))        { advance(); auto r = parse_unary(); left = make_expr(BinaryExpr{BinOp::Mul,  std::move(left), std::move(r)}); }
            else if (at(TokenKind::Slash))       { advance(); auto r = parse_unary(); left = make_expr(BinaryExpr{BinOp::Div,  std::move(left), std::move(r)}); }
            else if (at(TokenKind::DoubleSlash)) { advance(); auto r = parse_unary(); left = make_expr(BinaryExpr{BinOp::FloorDiv, std::move(left), std::move(r)}); }
            else if (at(TokenKind::Percent))     { advance(); auto r = parse_unary(); left = make_expr(BinaryExpr{BinOp::Mod,  std::move(left), std::move(r)}); }
            else break;
        }
        return left;
    }

    ExprPtr parse_unary() {
        if (match(TokenKind::Minus)) {
            auto operand = parse_unary();
            return make_expr(UnaryExpr{UnOp::Neg, std::move(operand)});
        }
        return parse_pow();
    }

    ExprPtr parse_pow() {
        auto base = parse_filter_chain();
        if (match(TokenKind::DoubleStar)) {
            auto exp = parse_unary(); // right-associative
            return make_expr(BinaryExpr{BinOp::Pow, std::move(base), std::move(exp)});
        }
        return base;
    }

    ExprPtr parse_filter_chain() {
        auto expr = parse_postfix();
        while (match(TokenKind::Pipe)) {
            auto name = expect(TokenKind::Ident).text;
            auto filter = string_to_filter_id(name);
            std::vector<ExprPtr> args;
            std::vector<std::pair<std::string, ExprPtr>> kwargs;
            if (at(TokenKind::LParen)) {
                parse_args_kwargs(args, kwargs);
            }
            expr = make_expr(FilterExpr{std::move(expr), filter, std::move(name),
                                        std::move(args), std::move(kwargs)});
        }
        return expr;
    }

    ExprPtr parse_postfix() {
        auto expr = parse_primary();
        while (true) {
            if (at(TokenKind::Dot)) {
                advance();
                auto attr = expect(TokenKind::Ident).text;
                expr = make_expr(GetAttrExpr{std::move(expr), std::move(attr)});
            } else if (at(TokenKind::LBracket)) {
                advance();
                // Optional start index
                ExprPtr start_expr;
                if (!at(TokenKind::Colon))
                    start_expr = parse_expression();
                if (at(TokenKind::Colon)) {
                    // Slice: [start:end], [start:end:step], [::step], etc.
                    advance();
                    ExprPtr end_expr;
                    if (!at(TokenKind::Colon) && !at(TokenKind::RBracket))
                        end_expr = parse_expression();
                    ExprPtr step_expr;
                    if (at(TokenKind::Colon)) {
                        advance();
                        if (!at(TokenKind::RBracket))
                            step_expr = parse_expression();
                    }
                    expect(TokenKind::RBracket);
                    expr = make_expr(SliceExpr{std::move(expr), std::move(start_expr),
                                               std::move(end_expr), std::move(step_expr)});
                } else {
                    // Subscript: [key]
                    expect(TokenKind::RBracket);
                    expr = make_expr(GetItemExpr{std::move(expr), std::move(start_expr)});
                }
            } else if (at(TokenKind::LParen)) {
                std::vector<ExprPtr> args;
                std::vector<std::pair<std::string, ExprPtr>> kwargs;
                parse_args_kwargs(args, kwargs);
                expr = make_expr(CallExpr{std::move(expr), std::move(args), std::move(kwargs)});
            } else {
                break;
            }
        }
        return expr;
    }

    ExprPtr parse_primary() {
        // String literals (with implicit concatenation of adjacent strings)
        if (at(TokenKind::String)) {
            auto t = advance();
            std::string s = t.text;
            while (at(TokenKind::String))
                s += advance().text;
            return make_expr(LiteralExpr{json(s)});
        }

        // Number literals
        if (at(TokenKind::Integer)) { auto t = advance(); return make_expr(LiteralExpr{t.value}); }
        if (at(TokenKind::Float))   { auto t = advance(); return make_expr(LiteralExpr{t.value}); }

        // Boolean / none literals
        if (match(TokenKind::KW_True))  { return make_expr(LiteralExpr{json(true)}); }
        if (match(TokenKind::KW_False)) { return make_expr(LiteralExpr{json(false)}); }
        if (match(TokenKind::KW_None))  { return make_expr(LiteralExpr{json(nullptr)}); }

        // Identifiers
        if (at(TokenKind::Ident)) {
            auto t = advance();
            return make_expr(IdentExpr{std::move(t.text)});
        }

        // Parenthesized expression
        if (match(TokenKind::LParen)) {
            auto expr = parse_ternary();
            expect(TokenKind::RParen);
            return expr;
        }

        // List literal
        if (match(TokenKind::LBracket)) {
            std::vector<ExprPtr> items;
            if (!at(TokenKind::RBracket)) {
                items.push_back(parse_expression());
                while (match(TokenKind::Comma)) {
                    if (at(TokenKind::RBracket)) break;
                    items.push_back(parse_expression());
                }
            }
            expect(TokenKind::RBracket);
            return make_expr(ListExpr{std::move(items)});
        }

        // Dict literal
        if (match(TokenKind::LBrace)) {
            std::vector<std::pair<ExprPtr, ExprPtr>> items;
            if (!at(TokenKind::RBrace)) {
                auto k = parse_expression();
                expect(TokenKind::Colon);
                auto v = parse_expression();
                items.emplace_back(std::move(k), std::move(v));
                while (match(TokenKind::Comma)) {
                    if (at(TokenKind::RBrace)) break;
                    k = parse_expression();
                    expect(TokenKind::Colon);
                    v = parse_expression();
                    items.emplace_back(std::move(k), std::move(v));
                }
            }
            expect(TokenKind::RBrace);
            return make_expr(DictExpr{std::move(items)});
        }

        throw std::runtime_error(
            "Unexpected token '" + tokens[pos].text + "' in expression");
    }

    // Parse argument list: (arg1, arg2, name=value, ...)
    void parse_args_kwargs(std::vector<ExprPtr>& args,
                           std::vector<std::pair<std::string, ExprPtr>>& kwargs) {
        expect(TokenKind::LParen);
        while (!at(TokenKind::RParen)) {
            // Try kwarg: Ident '=' expr  (but not '==')
            if (at(TokenKind::Ident) &&
                pos + 1 < tokens.size() &&
                tokens[pos + 1].kind == TokenKind::Assign) {
                auto name = advance().text;
                advance(); // skip '='
                kwargs.emplace_back(std::move(name), parse_expression());
            } else {
                args.push_back(parse_expression());
            }
            if (!at(TokenKind::RParen))
                expect(TokenKind::Comma);
        }
        expect(TokenKind::RParen);
    }
};

// ════════════════════════════════════════════════════════
// Section 4: Template Parser
// ════════════════════════════════════════════════════════

// -- Helpers --

std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

bool starts_with_keyword(const std::string& s, const std::string& keyword) {
    auto trimmed = trim(s);
    if (trimmed.size() < keyword.size()) return false;
    if (trimmed.substr(0, keyword.size()) != keyword) return false;
    if (trimmed.size() == keyword.size()) return true;
    char next = trimmed[keyword.size()];
    return !std::isalnum(static_cast<unsigned char>(next)) && next != '_';
}

// -- Parser --

struct TemplateParser {
    const std::vector<TemplateToken>& tokens;
    size_t pos = 0;

    bool at_end() const { return pos >= tokens.size(); }
    const TemplateToken& current() const { return tokens[pos]; }

    std::vector<NodePtr> parse_body(const std::vector<std::string>& end_keywords) {
        std::vector<NodePtr> nodes;
        while (!at_end()) {
            const auto& tok = current();
            if (tok.type == TemplateTokenType::BlockExpr) {
                auto trimmed = trim(tok.value);
                for (const auto& kw : end_keywords) {
                    if (starts_with_keyword(trimmed, kw)) {
                        return nodes;
                    }
                }
            }
            nodes.push_back(parse_node());
        }
        return nodes;
    }

    NodePtr parse_node() {
        const auto& tok = current();

        if (tok.type == TemplateTokenType::Text) {
            pos++;
            auto node = std::make_unique<Node>();
            node->type = NodeType::Text;
            node->text = tok.value;
            return node;
        }

        if (tok.type == TemplateTokenType::VarExpr) {
            pos++;
            auto node = std::make_unique<Node>();
            node->type = NodeType::Output;
            node->expr = parse_expression(tok.value);
            return node;
        }

        if (tok.type == TemplateTokenType::BlockExpr) {
            auto trimmed = trim(tok.value);
            if (starts_with_keyword(trimmed, "for")) {
                return parse_for();
            } else if (starts_with_keyword(trimmed, "if")) {
                return parse_if();
            } else if (starts_with_keyword(trimmed, "macro")) {
                return parse_macro();
            } else if (starts_with_keyword(trimmed, "set")) {
                return parse_set();
            } else {
                throw std::runtime_error("Unknown block tag: " + trimmed);
            }
        }

        throw std::runtime_error("Unexpected token type");
    }

    NodePtr parse_for() {
        auto toks = lex_expression(current().value);
        pos++; // consume 'for' block token

        ExprParser ep{toks};
        ep.expect(TokenKind::KW_For);

        // Read variable name(s) before 'in'
        std::string var_name = ep.expect(TokenKind::Ident).text;
        while (ep.at(TokenKind::Comma)) {
            ep.advance();
            var_name += ", " + ep.expect(TokenKind::Ident).text;
        }
        ep.expect(TokenKind::KW_In);

        auto iterable = ep.parse_expression();
        auto body = parse_body({"endfor"});
        if (!at_end() && starts_with_keyword(trim(current().value), "endfor"))
            pos++;

        auto node = std::make_unique<Node>();
        node->type = NodeType::For;
        node->var_name = var_name;
        node->iterable = std::move(iterable);
        node->body = std::move(body);
        return node;
    }

    NodePtr parse_if() {
        auto node = std::make_unique<Node>();
        node->type = NodeType::If;

        // Parse initial "if EXPR"
        {
            auto toks = lex_expression(current().value);
            pos++;
            ExprParser ep{toks};
            ep.expect(TokenKind::KW_If);
            IfBranch branch;
            branch.condition = ep.parse_expression();
            branch.body = parse_body({"elif", "else", "endif"});
            node->branches.push_back(std::move(branch));
        }

        // Parse elif/else/endif
        while (!at_end()) {
            auto trimmed = trim(current().value);
            if (starts_with_keyword(trimmed, "elif")) {
                auto toks = lex_expression(current().value);
                pos++;
                ExprParser ep{toks};
                ep.expect(TokenKind::KW_Elif);
                IfBranch branch;
                branch.condition = ep.parse_expression();
                branch.body = parse_body({"elif", "else", "endif"});
                node->branches.push_back(std::move(branch));
            } else if (starts_with_keyword(trimmed, "else")) {
                pos++;
                IfBranch branch;
                branch.condition = nullptr;
                branch.body = parse_body({"endif"});
                node->branches.push_back(std::move(branch));
            } else if (starts_with_keyword(trimmed, "endif")) {
                pos++;
                break;
            } else {
                break;
            }
        }

        return node;
    }

    NodePtr parse_set() {
        auto toks = lex_expression(current().value);
        pos++;

        ExprParser ep{toks};
        ep.expect(TokenKind::KW_Set);

        std::string var_name = ep.expect(TokenKind::Ident).text;
        if (ep.at(TokenKind::Dot)) {
            ep.advance();
            var_name += "." + ep.expect(TokenKind::Ident).text;
        }

        auto node = std::make_unique<Node>();
        node->type = NodeType::Set;
        node->var_name = var_name;
        if (ep.at(TokenKind::Assign)) {
            // Inline set: {% set var = expr %}
            ep.advance();
            node->expr = ep.parse_expression();
        } else {
            // Block set: {% set var %} ... {% endset %}
            // node->expr stays null as the sentinel; body holds the content.
            node->expr = nullptr;
            node->body = parse_body({"endset"});
            if (!at_end() && starts_with_keyword(trim(current().value), "endset"))
                pos++;
        }
        return node;
    }

    NodePtr parse_macro() {
        auto toks = lex_expression(current().value);
        pos++;

        ExprParser ep{toks};
        ep.expect(TokenKind::Ident);  // 'macro' keyword (lexes as identifier)

        auto node = std::make_unique<Node>();
        node->type = NodeType::Macro;
        node->var_name = ep.expect(TokenKind::Ident).text;  // macro name
        ep.expect(TokenKind::LParen);
        while (!ep.at(TokenKind::RParen)) {
            std::string pname = ep.expect(TokenKind::Ident).text;
            ExprPtr default_expr;
            if (ep.at(TokenKind::Assign)) {
                ep.advance();
                default_expr = ep.parse_expression();
            }
            node->params.emplace_back(std::move(pname), std::move(default_expr));
            if (ep.at(TokenKind::Comma)) ep.advance();
            else break;
        }
        ep.expect(TokenKind::RParen);

        node->body = parse_body({"endmacro"});
        if (!at_end() && starts_with_keyword(trim(current().value), "endmacro"))
            pos++;
        return node;
    }
};

// ════════════════════════════════════════════════════════
// Section 5: Builtin Functions, Filters, Methods, Tests
// ════════════════════════════════════════════════════════

// -- Shared helpers --

bool is_truthy(const json& val) {
    if (val.is_null()) return false;
    // The undefined sentinel is falsy (a missing variable is false in Jinja).
    if (val.is_string() && val.get<std::string>() == "__jinja_undefined__") return false;
    if (val.is_boolean()) return val.get<bool>();
    if (val.is_number_integer()) return val.get<int64_t>() != 0;
    if (val.is_number_float()) return val.get<double>() != 0.0;
    if (val.is_string()) return !val.get<std::string>().empty();
    if (val.is_array()) return !val.empty();
    if (val.is_object()) return !val.empty();
    return true;
}

const json UNDEFINED = json("__jinja_undefined__");

bool is_undefined(const json& val) {
    return val.is_string() && val.get<std::string>() == "__jinja_undefined__";
}

std::string json_to_string(const json& val) {
    if (val.is_string()) return val.get<std::string>();
    if (val.is_null()) return "";
    if (val.is_boolean()) return val.get<bool>() ? "True" : "False";
    if (val.is_number_integer()) return std::to_string(val.get<int64_t>());
    if (val.is_number_float()) {
        std::ostringstream oss;
        oss << val.get<double>();
        return oss.str();
    }
    return val.dump();
}

using KwargsVec = std::vector<std::pair<std::string, json>>;

// ──────────────────────────────────────────────────────
// Builtin global functions
// ──────────────────────────────────────────────────────

namespace functions {

/// Exception thrown by raise_exception() — caught by the evaluator.
struct TemplateError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// raise_exception("msg") — aborts template rendering with an error.
json raise_exception(const std::vector<json>& args) {
    std::string msg = "Template error";
    if (!args.empty()) msg = json_to_string(args[0]);
    throw TemplateError(msg);
}

/// namespace(key=val, ...) — creates a mutable namespace object.
json make_namespace(const std::vector<json>& /*args*/, const KwargsVec& kwargs) {
    json ns = json::object();
    for (const auto& [key, val] : kwargs)
        ns[key] = val;
    return ns;
}

/// range(stop) or range(start, stop[, step]).
json range(const std::vector<json>& args) {
    json arr = json::array();
    int64_t start = 0, stop = 0, step = 1;
    if (args.size() == 1) {
        stop = args[0].get<int64_t>();
    } else if (args.size() >= 2) {
        start = args[0].get<int64_t>();
        stop = args[1].get<int64_t>();
        if (args.size() >= 3) step = args[2].get<int64_t>();
    }
    for (int64_t i = start; i < stop; i += step) arr.push_back(i);
    return arr;
}

/// dict(**kwargs) — creates a dict from keyword arguments.
json dict(const KwargsVec& kwargs) {
    json obj = json::object();
    for (const auto& [key, val] : kwargs)
        obj[key] = val;
    return obj;
}

} // namespace functions

// ──────────────────────────────────────────────────────
// Builtin filters
// ──────────────────────────────────────────────────────

namespace filters {

json trim(const json& value, const std::vector<json>& /*args*/) {
    auto s = json_to_string(value);
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

json length(const json& value, const std::vector<json>& /*args*/) {
    if (value.is_string()) return static_cast<int64_t>(value.get<std::string>().size());
    if (value.is_array())  return static_cast<int64_t>(value.size());
    if (value.is_object()) return static_cast<int64_t>(value.size());
    return 0;
}

json default_filter(const json& value, const std::vector<json>& args) {
    if (is_undefined(value) || value.is_null()) {
        if (!args.empty()) return args[0];
        return "";
    }
    if (args.size() >= 2) {
        if (is_truthy(args[1]) && !is_truthy(value))
            return args[0];
    }
    return value;
}

json first(const json& value, const std::vector<json>& /*args*/) {
    if (value.is_array() && !value.empty()) return value[0];
    return UNDEFINED;
}

json last(const json& value, const std::vector<json>& /*args*/) {
    if (value.is_array() && !value.empty()) return value[value.size() - 1];
    return UNDEFINED;
}

json upper(const json& value, const std::vector<json>& /*args*/) {
    auto s = json_to_string(value);
    std::transform(s.begin(), s.end(), s.begin(),
                  [](unsigned char c) { return std::toupper(c); });
    return s;
}

json lower(const json& value, const std::vector<json>& /*args*/) {
    auto s = json_to_string(value);
    std::transform(s.begin(), s.end(), s.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    return s;
}

json title(const json& value, const std::vector<json>& /*args*/) {
    auto s = json_to_string(value);
    bool next_upper = true;
    for (auto& c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            next_upper = true;
        } else if (next_upper) {
            c = std::toupper(static_cast<unsigned char>(c));
            next_upper = false;
        } else {
            c = std::tolower(static_cast<unsigned char>(c));
        }
    }
    return s;
}

json join(const json& value, const std::vector<json>& args) {
    std::string sep;
    if (!args.empty()) sep = json_to_string(args[0]);
    if (value.is_array()) {
        std::string result;
        for (size_t i = 0; i < value.size(); i++) {
            if (i > 0) result += sep;
            result += json_to_string(value[i]);
        }
        return result;
    }
    return json_to_string(value);
}

json to_list(const json& value, const std::vector<json>& /*args*/) {
    if (value.is_array()) return value;
    if (value.is_string()) {
        json arr = json::array();
        for (char c : value.get<std::string>())
            arr.push_back(std::string(1, c));
        return arr;
    }
    return json::array();
}

json to_int(const json& value, const std::vector<json>& /*args*/) {
    if (value.is_number()) return static_cast<int64_t>(value.get<double>());
    if (value.is_string()) {
        try { return static_cast<int64_t>(std::stoll(value.get<std::string>())); }
        catch (...) { return 0; }
    }
    return 0;
}

json to_float(const json& value, const std::vector<json>& /*args*/) {
    if (value.is_number()) return value.get<double>();
    if (value.is_string()) {
        try { return std::stod(value.get<std::string>()); }
        catch (...) { return 0.0; }
    }
    return 0.0;
}

json to_string(const json& value, const std::vector<json>& /*args*/) {
    return json_to_string(value);
}

json tojson(const json& value, const std::vector<json>& /*args*/) {
    return value.dump();
}

json replace(const json& value, const std::vector<json>& args) {
    if (args.size() >= 2) {
        auto s = json_to_string(value);
        auto from = json_to_string(args[0]);
        auto to = json_to_string(args[1]);
        size_t p = 0;
        while ((p = s.find(from, p)) != std::string::npos) {
            s.replace(p, from.length(), to);
            p += to.length();
        }
        return json(s);
    }
    return value;
}

json batch(const json& value, const std::vector<json>& args) {
    if (value.is_array() && !args.empty()) {
        auto size = args[0].get<int64_t>();
        json arr = json::array();
        for (size_t i = 0; i < value.size(); i += size) {
            json b = json::array();
            for (size_t j = i; j < std::min(i + static_cast<size_t>(size), value.size()); j++)
                b.push_back(value[j]);
            arr.push_back(b);
        }
        return arr;
    }
    return value;
}

json reverse(const json& value, const std::vector<json>& /*args*/) {
    if (value.is_array()) {
        json arr = value;
        std::reverse(arr.begin(), arr.end());
        return arr;
    }
    return value;
}

json dictsort(const json& value, const std::vector<json>& args) {
    // Sort a mapping by key, returning a list of [key, value] pairs.
    // Like Jinja, keys compare case-insensitively unless the first argument
    // (case_sensitive) is true.
    bool case_sensitive = !args.empty() && is_truthy(args[0]);
    auto fold = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) out.push_back(static_cast<char>(std::tolower(c)));
        return out;
    };
    json arr = json::array();
    if (value.is_object()) {
        std::vector<std::string> keys;
        keys.reserve(value.size());
        for (auto& [k, v] : value.items()) keys.push_back(k);
        std::sort(keys.begin(), keys.end(),
                  [&](const std::string& a, const std::string& b) {
                      if (case_sensitive) return a < b;
                      const auto fa = fold(a), fb = fold(b);
                      // Raw keys break ties so the order stays deterministic.
                      return fa != fb ? fa < fb : a < b;
                  });
        for (const auto& k : keys) arr.push_back(json::array({k, value[k]}));
    }
    return arr;
}

json items(const json& value, const std::vector<json>& /*args*/) {
    // Mapping -> list of [key, value] pairs (insertion order).
    json arr = json::array();
    if (value.is_object())
        for (auto& [k, v] : value.items()) arr.push_back(json::array({k, v}));
    return arr;
}

} // namespace filters

// ──────────────────────────────────────────────────────
// String & dict methods
// ──────────────────────────────────────────────────────

namespace methods {

// -- String methods --

json str_upper(const std::string& s, const std::vector<json>& /*args*/) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                  [](unsigned char c) { return std::toupper(c); });
    return json(result);
}

json str_lower(const std::string& s, const std::vector<json>& /*args*/) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    return json(result);
}

json str_strip(const std::string& s, const std::vector<json>& /*args*/) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

json str_split(const std::string& s, const std::vector<json>& args) {
    std::string delim = " ";
    if (!args.empty()) delim = json_to_string(args[0]);
    json arr = json::array();
    size_t start_pos = 0, found;
    while ((found = s.find(delim, start_pos)) != std::string::npos) {
        arr.push_back(s.substr(start_pos, found - start_pos));
        start_pos = found + delim.size();
    }
    arr.push_back(s.substr(start_pos));
    return arr;
}

json str_replace(const std::string& s, const std::vector<json>& args) {
    if (args.size() >= 2) {
        std::string result = s;
        auto from = json_to_string(args[0]);
        auto to = json_to_string(args[1]);
        size_t p = 0;
        while ((p = result.find(from, p)) != std::string::npos) {
            result.replace(p, from.length(), to);
            p += to.length();
        }
        return json(result);
    }
    return json(s);
}

json str_startswith(const std::string& s, const std::vector<json>& args) {
    if (!args.empty()) {
        auto prefix = json_to_string(args[0]);
        return json(s.substr(0, prefix.size()) == prefix);
    }
    return json(false);
}

json str_endswith(const std::string& s, const std::vector<json>& args) {
    if (!args.empty()) {
        auto suffix = json_to_string(args[0]);
        if (suffix.size() > s.size()) return json(false);
        return json(s.substr(s.size() - suffix.size()) == suffix);
    }
    return json(false);
}

json str_lstrip(const std::string& s, const std::vector<json>& args) {
    std::string chars = " \t\n\r";
    if (!args.empty() && args[0].is_string()) chars = args[0].get<std::string>();
    auto start = s.find_first_not_of(chars);
    if (start == std::string::npos) return std::string();
    return json(s.substr(start));
}

json str_rstrip(const std::string& s, const std::vector<json>& args) {
    std::string chars = " \t\n\r";
    if (!args.empty() && args[0].is_string()) chars = args[0].get<std::string>();
    auto end = s.find_last_not_of(chars);
    if (end == std::string::npos) return std::string();
    return json(s.substr(0, end + 1));
}

// -- Dict methods --

json dict_items(const json& obj, const std::vector<json>& /*args*/) {
    json arr = json::array();
    for (auto& [key, val] : obj.items())
        arr.push_back(json::array({key, val}));
    return arr;
}

json dict_keys(const json& obj, const std::vector<json>& /*args*/) {
    json arr = json::array();
    for (auto& [key, val] : obj.items()) arr.push_back(key);
    return arr;
}

json dict_values(const json& obj, const std::vector<json>& /*args*/) {
    json arr = json::array();
    for (auto& [key, val] : obj.items()) arr.push_back(val);
    return arr;
}

json dict_get(const json& obj, const std::vector<json>& args) {
    if (!args.empty()) {
        auto key = json_to_string(args[0]);
        if (obj.contains(key)) return obj[key];
        if (args.size() >= 2) return args[1];
    }
    return UNDEFINED;
}

} // namespace methods

// ──────────────────────────────────────────────────────
// Jinja tests (is defined, is string, is odd, etc.)
// ──────────────────────────────────────────────────────

namespace tests {

bool defined(const json& val)   { return !is_undefined(val); }
bool undefined(const json& val) { return is_undefined(val); }
bool none(const json& val)      { return val.is_null() || is_undefined(val); }
bool is_true(const json& val)   { return val.is_boolean() && val.get<bool>(); }
bool is_false(const json& val)  { return val.is_boolean() && !val.get<bool>(); }
bool boolean(const json& val)   { return val.is_boolean(); }
bool string(const json& val)    { return val.is_string() && !is_undefined(val); }
bool number(const json& val)    { return val.is_number(); }
bool integer(const json& val)   { return val.is_number_integer(); }
bool is_float(const json& val)  { return val.is_number_float(); }
bool mapping(const json& val)   { return val.is_object(); }
bool iterable(const json& val)  { return val.is_array() || val.is_string() || val.is_object(); }
bool sequence(const json& val)  { return val.is_array(); }
bool callable(const json&)      { return false; }
bool even(const json& val)      { return val.is_number_integer() && val.get<int64_t>() % 2 == 0; }
bool odd(const json& val)       { return val.is_number_integer() && val.get<int64_t>() % 2 != 0; }

} // namespace tests

// ──────────────────────────────────────────────────────
// Registration maps — static lookup tables for dispatch
// ──────────────────────────────────────────────────────

using FuncFn = std::function<json(const std::vector<json>&, const KwargsVec&)>;

static const std::unordered_map<std::string, FuncFn> BUILTIN_FUNCTIONS = {
    {"raise_exception", [](const auto& args, const auto&)    { return functions::raise_exception(args); }},
    {"namespace",       [](const auto& args, const auto& kw) { return functions::make_namespace(args, kw); }},
    {"range",           [](const auto& args, const auto&)    { return functions::range(args); }},
    {"dict",            [](const auto&, const auto& kw)      { return functions::dict(kw); }},
};

using FilterFn = std::function<json(const json&, const std::vector<json>&)>;

static const std::unordered_map<FilterId, FilterFn> SIMPLE_FILTERS = {
    {FilterId::Trim,    filters::trim},
    {FilterId::Length,  filters::length},
    {FilterId::Default, filters::default_filter},
    {FilterId::First,   filters::first},
    {FilterId::Last,    filters::last},
    {FilterId::Upper,   filters::upper},
    {FilterId::Lower,   filters::lower},
    {FilterId::Title,   filters::title},
    {FilterId::Join,    filters::join},
    {FilterId::List,    filters::to_list},
    {FilterId::Int,     filters::to_int},
    {FilterId::Float,   filters::to_float},
    {FilterId::String,  filters::to_string},
    {FilterId::ToJson,  filters::tojson},
    {FilterId::Replace, filters::replace},
    {FilterId::Batch,   filters::batch},
    {FilterId::Reverse, filters::reverse},
    {FilterId::DictSort, filters::dictsort},
    {FilterId::Items,   filters::items},
};

using StrMethodFn = std::function<json(const std::string&, const std::vector<json>&)>;

static const std::unordered_map<std::string, StrMethodFn> STRING_METHODS = {
    {"upper",      methods::str_upper},
    {"lower",      methods::str_lower},
    {"strip",      methods::str_strip},
    {"split",      methods::str_split},
    {"replace",    methods::str_replace},
    {"startswith", methods::str_startswith},
    {"endswith",   methods::str_endswith},
    {"lstrip",     methods::str_lstrip},
    {"rstrip",     methods::str_rstrip},
};

using DictMethodFn = std::function<json(const json&, const std::vector<json>&)>;

static const std::unordered_map<std::string, DictMethodFn> DICT_METHODS = {
    {"items",  methods::dict_items},
    {"keys",   methods::dict_keys},
    {"values", methods::dict_values},
    {"get",    methods::dict_get},
};

using TestFn = std::function<bool(const json&)>;

static const std::unordered_map<TestOp, TestFn> BUILTIN_TESTS = {
    {TestOp::Defined,   tests::defined},
    {TestOp::Undefined, tests::undefined},
    {TestOp::None,      tests::none},
    {TestOp::True,      tests::is_true},
    {TestOp::False,     tests::is_false},
    {TestOp::Boolean,   tests::boolean},
    {TestOp::String,    tests::string},
    {TestOp::Number,    tests::number},
    {TestOp::Integer,   tests::integer},
    {TestOp::Float,     tests::is_float},
    {TestOp::Mapping,   tests::mapping},
    {TestOp::Iterable,  tests::iterable},
    {TestOp::Sequence,  tests::sequence},
    {TestOp::Callable,  tests::callable},
    {TestOp::Even,      tests::even},
    {TestOp::Odd,       tests::odd},
    {TestOp::Eq,        [](const json&) { return false; }},
    {TestOp::Ne,        [](const json&) { return false; }},
};

// ════════════════════════════════════════════════════════
// Section 6: Evaluator
// ════════════════════════════════════════════════════════

struct Evaluator {
    json vars;
    std::string output;
    std::string error_msg;
    bool has_error = false;
    std::vector<json*> scope_stack;
    std::unordered_map<std::string, const Node*> macros;
    int macro_depth = 0;

    Evaluator(const json& context) : vars(context) {
        scope_stack.push_back(&vars);
    }

    json resolve(const std::string& name) {
        for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
            auto* scope = *it;
            if (scope->is_object() && scope->contains(name))
                return (*scope)[name];
        }
        return UNDEFINED;
    }

    void set_var(const std::string& name, const json& value) {
        auto dot = name.find('.');
        if (dot != std::string::npos) {
            std::string obj_name = name.substr(0, dot);
            std::string attr_name = name.substr(dot + 1);
            json obj = resolve(obj_name);
            if (obj.is_object()) {
                obj[attr_name] = value;
                for (auto it = scope_stack.rbegin(); it != scope_stack.rend(); ++it) {
                    if ((*it)->is_object() && (*it)->contains(obj_name)) {
                        (**it)[obj_name] = obj;
                        return;
                    }
                }
                (*scope_stack.back())[obj_name] = obj;
            }
            return;
        }
        (*scope_stack.back())[name] = value;
    }

    // -- Argument evaluation helpers --

    std::vector<json> eval_args(const std::vector<ExprPtr>& exprs) {
        std::vector<json> result;
        result.reserve(exprs.size());
        for (const auto& e : exprs) result.push_back(eval(*e));
        return result;
    }

    KwargsVec eval_kwargs(const std::vector<std::pair<std::string, ExprPtr>>& kw_exprs) {
        KwargsVec result;
        result.reserve(kw_exprs.size());
        for (const auto& [k, v] : kw_exprs) result.emplace_back(k, eval(*v));
        return result;
    }

    // ---- Expression evaluation (dispatches on variant) ----

    json eval(const ExprNode& node) {
        if (auto* e = std::get_if<LiteralExpr>(&node))  return e->value;
        if (auto* e = std::get_if<IdentExpr>(&node))    return resolve(e->name);
        if (auto* e = std::get_if<BinaryExpr>(&node))   return eval_binary(*e);
        if (auto* e = std::get_if<UnaryExpr>(&node))    return eval_unary(*e);
        if (auto* e = std::get_if<GetAttrExpr>(&node))  return eval_getattr(*e);
        if (auto* e = std::get_if<GetItemExpr>(&node))  return eval_getitem(*e);
        if (auto* e = std::get_if<SliceExpr>(&node))    return eval_slice(*e);
        if (auto* e = std::get_if<CallExpr>(&node))     return eval_call(*e);
        if (auto* e = std::get_if<FilterExpr>(&node))   return eval_filter(*e);
        if (auto* e = std::get_if<TestExpr>(&node))     return eval_test(*e);
        if (auto* e = std::get_if<CondExpr>(&node))     return eval_cond(*e);
        if (auto* e = std::get_if<ListExpr>(&node))     return eval_list(*e);
        if (auto* e = std::get_if<DictExpr>(&node))     return eval_dict(*e);
        return UNDEFINED;
    }

    json eval_getattr(const GetAttrExpr& e) {
        auto obj = eval(*e.object);
        if (obj.is_object() && obj.contains(e.attr))
            return obj[e.attr];
        return UNDEFINED;
    }

    json eval_getitem(const GetItemExpr& e) {
        auto obj = eval(*e.object);
        auto key = eval(*e.key);
        if (obj.is_array() && key.is_number_integer()) {
            auto idx = key.get<int64_t>();
            if (idx < 0) idx += static_cast<int64_t>(obj.size());
            if (idx >= 0 && idx < static_cast<int64_t>(obj.size()))
                return obj[static_cast<size_t>(idx)];
            return UNDEFINED;
        }
        if (obj.is_object() && key.is_string()) {
            auto k = key.get<std::string>();
            if (obj.contains(k)) return obj[k];
            return UNDEFINED;
        }
        return UNDEFINED;
    }

    /// Fold a negative slice index against @p len and clamp it into [lo, hi],
    /// saturating instead of overflowing on extreme values (e.g. -1e12).
    static int64_t clamp_index(int64_t idx, int64_t len, int64_t lo, int64_t hi) {
        if (idx < 0) idx = (idx < -len) ? lo : idx + len;
        return std::min(std::max(idx, lo), hi);
    }

    json eval_slice(const SliceExpr& e) {
        auto obj = eval(*e.object);
        if (!obj.is_array()) return UNDEFINED;
        int64_t len = static_cast<int64_t>(obj.size());
        int64_t step = 1;
        if (e.step) {
            auto sv = eval(*e.step);
            if (sv.is_number_integer()) step = sv.get<int64_t>();
            if (step == 0) step = 1;
        }
        // |step| >= len yields at most one element, so clamping the magnitude
        // keeps the result identical while stopping `i += step` from overflowing.
        if (len > 0) step = std::min(std::max(step, -len), len);

        json result = json::array();
        if (step > 0) {
            // Both bounds live in [0, len], so the loop is bounded by len.
            int64_t start = 0, end = len;
            if (e.start) start = clamp_index(eval(*e.start).get<int64_t>(), len, 0, len);
            if (e.end) end = clamp_index(eval(*e.end).get<int64_t>(), len, 0, len);
            for (int64_t i = start; i < end; i += step)
                result.push_back(obj[static_cast<size_t>(i)]);
        } else {
            // Negative step: iterate high -> low (e.g. [::-1] reverses).
            // Both bounds live in [-1, len-1], so the loop is bounded by len.
            int64_t start = len - 1, end = -1;
            if (e.start) start = clamp_index(eval(*e.start).get<int64_t>(), len, -1, len - 1);
            if (e.end) end = clamp_index(eval(*e.end).get<int64_t>(), len, -1, len - 1);
            for (int64_t i = start; i > end; i += step)
                result.push_back(obj[static_cast<size_t>(i)]);
        }
        return result;
    }

    json eval_cond(const CondExpr& e) {
        auto cond = eval(*e.condition);
        if (is_truthy(cond)) return eval(*e.true_val);
        if (e.false_val) return eval(*e.false_val);
        return "";
    }

    json eval_list(const ListExpr& e) {
        json arr = json::array();
        for (const auto& item : e.items)
            arr.push_back(eval(*item));
        return arr;
    }

    json eval_dict(const DictExpr& e) {
        json obj = json::object();
        for (const auto& [k, v] : e.items)
            obj[json_to_string(eval(*k))] = eval(*v);
        return obj;
    }

    // ---- Unary operators ----

    json eval_unary(const UnaryExpr& e) {
        auto val = eval(*e.operand);
        switch (e.op) {
            case UnOp::Not: return json(!is_truthy(val));
            case UnOp::Neg:
                if (val.is_number_integer()) return json(-val.get<int64_t>());
                if (val.is_number_float()) return json(-val.get<double>());
                return UNDEFINED;
        }
        return UNDEFINED;
    }

    // ---- Binary operators ----

    static bool contains(const json& haystack, const json& needle) {
        if (haystack.is_array()) {
            for (const auto& item : haystack)
                if (item == needle) return true;
            return false;
        }
        if (haystack.is_string() && needle.is_string())
            return haystack.get<std::string>().find(needle.get<std::string>()) != std::string::npos;
        if (haystack.is_object() && needle.is_string())
            return haystack.contains(needle.get<std::string>());
        return false;
    }

    json eval_binary(const BinaryExpr& e) {
        // Short-circuit operators — must not evaluate both sides
        switch (e.op) {
            case BinOp::And: {
                auto left = eval(*e.left);
                return is_truthy(left) ? eval(*e.right) : left;
            }
            case BinOp::Or: {
                auto left = eval(*e.left);
                return is_truthy(left) ? left : eval(*e.right);
            }
            default: break;
        }

        auto left = eval(*e.left);
        auto right = eval(*e.right);

        switch (e.op) {
            // Comparison
            case BinOp::Eq:    return json(left == right);
            case BinOp::NotEq: return json(left != right);
            case BinOp::Lt:    return json(left < right);
            case BinOp::Gt:    return json(left > right);
            case BinOp::LtEq:  return json(left <= right);
            case BinOp::GtEq:  return json(left >= right);

            // Membership
            case BinOp::In:    return json(contains(right, left));
            case BinOp::NotIn: return json(!contains(right, left));

            // String concatenation
            case BinOp::Concat:
                return json_to_string(left) + json_to_string(right);

            // Arithmetic
            case BinOp::Add:
                if (left.is_string() && right.is_string())
                    return left.get<std::string>() + right.get<std::string>();
                if (left.is_number() && right.is_number()) {
                    if (left.is_number_float() || right.is_number_float())
                        return left.get<double>() + right.get<double>();
                    return left.get<int64_t>() + right.get<int64_t>();
                }
                return json_to_string(left) + json_to_string(right);

            case BinOp::Sub:
                if (left.is_number() && right.is_number()) {
                    if (left.is_number_float() || right.is_number_float())
                        return left.get<double>() - right.get<double>();
                    return left.get<int64_t>() - right.get<int64_t>();
                }
                return UNDEFINED;

            case BinOp::Mul:
                if (left.is_number() && right.is_number()) {
                    if (left.is_number_float() || right.is_number_float())
                        return left.get<double>() * right.get<double>();
                    return left.get<int64_t>() * right.get<int64_t>();
                }
                return UNDEFINED;

            case BinOp::Div:
                if (left.is_number() && right.is_number())
                    return left.get<double>() / right.get<double>();
                return UNDEFINED;

            case BinOp::FloorDiv:
                if (left.is_number() && right.is_number())
                    return static_cast<int64_t>(left.get<double>() / right.get<double>());
                return UNDEFINED;

            case BinOp::Mod:
                if (left.is_number_integer() && right.is_number_integer())
                    return left.get<int64_t>() % right.get<int64_t>();
                return UNDEFINED;

            case BinOp::Pow:
                if (left.is_number() && right.is_number())
                    return std::pow(left.get<double>(), right.get<double>());
                return UNDEFINED;

            default: return UNDEFINED;
        }
    }

    // ---- Tests (is / is not) — dispatched from BUILTIN_TESTS map ----

    json eval_test(const TestExpr& e) {
        auto val = eval(*e.value);
        auto it = BUILTIN_TESTS.find(e.test);
        bool result = (it != BUILTIN_TESTS.end()) ? it->second(val) : false;
        return json(e.negated ? !result : result);
    }

    // ---- Function & method calls — dispatched from registration maps ----

    json eval_call(const CallExpr& e) {
        auto args = eval_args(e.args);
        auto kwargs = eval_kwargs(e.kwargs);

        // Global function call: name(args, kwargs)
        if (auto* ident = std::get_if<IdentExpr>(e.callee.get())) {
            // User-defined macro call takes precedence over builtins.
            auto mit = macros.find(ident->name);
            if (mit != macros.end())
                return invoke_macro(*mit->second, args, kwargs);

            auto it = BUILTIN_FUNCTIONS.find(ident->name);
            if (it != BUILTIN_FUNCTIONS.end()) {
                try {
                    return it->second(args, kwargs);
                } catch (const functions::TemplateError& err) {
                    error_msg = err.what();
                    has_error = true;
                    return UNDEFINED;
                }
            }
        }

        // Method call: obj.method(args)
        if (auto* ga = std::get_if<GetAttrExpr>(e.callee.get())) {
            auto obj = eval(*ga->object);

            if (obj.is_string()) {
                auto it = STRING_METHODS.find(ga->attr);
                if (it != STRING_METHODS.end())
                    return it->second(obj.get<std::string>(), args);
            }

            if (obj.is_object()) {
                auto it = DICT_METHODS.find(ga->attr);
                if (it != DICT_METHODS.end())
                    return it->second(obj, args);
            }
        }

        return UNDEFINED;
    }

    // ---- Filters — simple filters from map, complex filters inline ----

    json eval_filter(const FilterExpr& e) {
        auto val = eval(*e.value);
        auto args = eval_args(e.args);

        // Simple filters — dispatched from the SIMPLE_FILTERS map
        auto it = SIMPLE_FILTERS.find(e.filter);
        if (it != SIMPLE_FILTERS.end())
            return it->second(val, args);

        // Complex filters that need kwargs or special handling
        switch (e.filter) {
        case FilterId::Map: {
            if (val.is_array()) {
                auto kwargs = eval_kwargs(e.kwargs);
                std::string attr;
                for (const auto& [k, v] : kwargs) {
                    if (k == "attribute") attr = json_to_string(v);
                }
                if (!attr.empty()) {
                    // map(attribute='name') — extract an attribute from each item.
                    json arr = json::array();
                    for (const auto& item : val) {
                        if (item.is_object() && item.contains(attr))
                            arr.push_back(item[attr]);
                        else
                            arr.push_back(UNDEFINED);
                    }
                    return arr;
                }
                if (!args.empty()) {
                    // map('filter_name', extra_args...) — apply a filter to each item.
                    FilterId fid = string_to_filter_id(json_to_string(args[0]));
                    std::vector<json> fargs(args.begin() + 1, args.end());
                    auto fit = SIMPLE_FILTERS.find(fid);
                    json arr = json::array();
                    for (const auto& item : val) {
                        if (fit != SIMPLE_FILTERS.end())
                            arr.push_back(fit->second(item, fargs));
                        else
                            arr.push_back(item);
                    }
                    return arr;
                }
            }
            return val;
        }
        case FilterId::SelectAttr: {
            if (val.is_array() && !args.empty()) {
                auto attr = json_to_string(args[0]);
                json arr = json::array();
                if (args.size() >= 3) {
                    auto test = json_to_string(args[1]);
                    auto test_val = args[2];
                    for (const auto& item : val) {
                        if (!item.is_object() || !item.contains(attr)) continue;
                        if ((test == "equalto" || test == "eq" || test == "==") && item[attr] == test_val)
                            arr.push_back(item);
                        else if ((test == "ne" || test == "!=") && item[attr] != test_val)
                            arr.push_back(item);
                    }
                } else {
                    for (const auto& item : val) {
                        if (item.is_object() && item.contains(attr) && is_truthy(item[attr]))
                            arr.push_back(item);
                    }
                }
                return arr;
            }
            return val;
        }
        default:
            return val;  // Unimplemented filters pass through
        }
    }

    // ---- Macro invocation — renders the macro body into a string ----

    /// Guards against a self-recursive macro overflowing the stack. Templates
    /// come from model repos, so a runaway definition must fail, not crash.
    static constexpr int MAX_MACRO_DEPTH = 64;

    json invoke_macro(const Node& macro, const std::vector<json>& args,
                      const KwargsVec& kwargs) {
        if (macro_depth >= MAX_MACRO_DEPTH) {
            error_msg = "macro recursion too deep in '" + macro.var_name + "' (max " +
                        std::to_string(MAX_MACRO_DEPTH) + ")";
            has_error = true;
            return UNDEFINED;
        }
        json scope = json::object();
        for (size_t i = 0; i < macro.params.size(); i++) {
            const auto& pname = macro.params[i].first;
            const auto& pdefault = macro.params[i].second;
            json val = UNDEFINED;
            if (i < args.size()) {
                val = args[i];
            } else {
                bool found = false;
                for (const auto& [k, v] : kwargs) {
                    if (k == pname) { val = v; found = true; break; }
                }
                if (!found && pdefault) val = eval(*pdefault);
            }
            scope[pname] = val;
        }
        // Push the macro scope onto the current stack so the body can read
        // globals and mutate enclosing namespace objects (e.g. counters).
        scope_stack.push_back(&scope);
        macro_depth++;
        std::string saved = std::move(output);
        output.clear();
        for (const auto& child : macro.body) {
            exec(*child);
            if (has_error) break;
        }
        std::string result = std::move(output);
        output = std::move(saved);
        macro_depth--;
        scope_stack.pop_back();
        return json(result);
    }

    // ---- Statement execution ----

    void exec(const Node& node) {
        if (has_error) return;

        switch (node.type) {
        case NodeType::Text:
            output += node.text;
            break;

        case NodeType::Output: {
            auto val = eval(*node.expr);
            if (has_error) return;
            if (!is_undefined(val))
                output += json_to_string(val);
            break;
        }

        case NodeType::Block:
            for (const auto& child : node.body) {
                exec(*child);
                if (has_error) return;
            }
            break;

        case NodeType::For: {
            auto iterable = eval(*node.iterable);
            if (!iterable.is_array()) {
                if (iterable.is_object()) {
                    json keys = json::array();
                    for (auto& [k, v] : iterable.items()) keys.push_back(k);
                    iterable = keys;
                } else {
                    break;
                }
            }

            size_t length = iterable.size();
            json loop_scope = json::object();
            scope_stack.push_back(&loop_scope);

            bool tuple_unpack = node.var_name.find(',') != std::string::npos;
            std::vector<std::string> var_names;
            if (tuple_unpack) {
                std::string names = node.var_name;
                size_t cpos = 0;
                while (true) {
                    auto comma = names.find(',', cpos);
                    std::string n = (comma == std::string::npos)
                        ? names.substr(cpos)
                        : names.substr(cpos, comma - cpos);
                    auto s = n.find_first_not_of(" \t");
                    auto e2 = n.find_last_not_of(" \t");
                    if (s != std::string::npos)
                        var_names.push_back(n.substr(s, e2 - s + 1));
                    if (comma == std::string::npos) break;
                    cpos = comma + 1;
                }
            }

            for (size_t i = 0; i < length; i++) {
                if (tuple_unpack && iterable[i].is_array()) {
                    for (size_t j = 0; j < var_names.size() && j < iterable[i].size(); j++)
                        loop_scope[var_names[j]] = iterable[i][j];
                } else {
                    loop_scope[node.var_name] = iterable[i];
                }

                json loop_var = json::object();
                loop_var["index"]     = static_cast<int64_t>(i + 1);
                loop_var["index0"]    = static_cast<int64_t>(i);
                loop_var["first"]     = (i == 0);
                loop_var["last"]      = (i == length - 1);
                loop_var["length"]    = static_cast<int64_t>(length);
                loop_var["revindex"]  = static_cast<int64_t>(length - i);
                loop_var["revindex0"] = static_cast<int64_t>(length - i - 1);
                // previtem/nextitem are null (falsy) at the sequence boundaries.
                loop_var["previtem"]  = (i > 0) ? iterable[i - 1] : json(nullptr);
                loop_var["nextitem"]  = (i + 1 < length) ? iterable[i + 1] : json(nullptr);
                loop_scope["loop"]    = loop_var;

                for (const auto& child : node.body) {
                    exec(*child);
                    if (has_error) return;
                }
            }

            scope_stack.pop_back();
            break;
        }

        case NodeType::If:
            for (const auto& branch : node.branches) {
                if (!branch.condition || is_truthy(eval(*branch.condition))) {
                    for (const auto& child : branch.body) {
                        exec(*child);
                        if (has_error) return;
                    }
                    break;
                }
            }
            break;

        case NodeType::Set: {
            if (node.expr) {
                auto val = eval(*node.expr);
                set_var(node.var_name, val);
            } else {
                // Block set: render the body into a string and assign it.
                std::string saved = std::move(output);
                output.clear();
                for (const auto& child : node.body) {
                    exec(*child);
                    if (has_error) break;
                }
                std::string captured = std::move(output);
                output = std::move(saved);
                if (!has_error) set_var(node.var_name, json(captured));
            }
            break;
        }

        case NodeType::Macro:
            // Register the macro definition; produces no output itself.
            macros[node.var_name] = &node;
            break;
        }
    }
};

} // anonymous namespace

// ════════════════════════════════════════════════════════
// Section 7: Public API
// ════════════════════════════════════════════════════════

ExprPtr parse_expression(std::string_view src) {
    auto tokens = lex_expression(src);
    ExprParser parser{tokens, 0};
    return parser.parse_expression();
}

NodePtr parse(const std::vector<TemplateToken>& tokens) {
    TemplateParser parser{tokens, 0};
    auto root = std::make_unique<Node>();
    root->type = NodeType::Block;
    root->body = parser.parse_body({});
    return root;
}

Result<std::string> render(const Node& root, const json& context) {
    Evaluator evaluator(context);
    evaluator.exec(root);
    if (evaluator.has_error)
        return make_error(evaluator.error_msg);
    return evaluator.output;
}

} // namespace tokenizers::jinja

// ════════════════════════════════════════════════════════
// ChatTemplate implementation
// ════════════════════════════════════════════════════════

namespace tokenizers {

ChatTemplate::ChatTemplate(const std::string& template_str,
                           std::optional<std::string> bos_token,
                           std::optional<std::string> eos_token)
    : template_str_(template_str),
      bos_token_(std::move(bos_token)),
      eos_token_(std::move(eos_token)) {}

namespace {

/// Context keys the engine supplies itself. A caller setting one of these would
/// be changing what the template sees about the *conversation* rather than
/// about the model, so they are rejected instead of silently overridden.
bool is_reserved_context_key(std::string_view key) {
    return key == "messages" || key == "add_generation_prompt" ||
           key == "bos_token" || key == "eos_token";
}

Result<std::string> render_template(const std::string& template_str,
                                    const std::optional<std::string>& bos_token,
                                    const std::optional<std::string>& eos_token,
                                    nlohmann::json messages,
                                    bool add_generation_prompt,
                                    const nlohmann::json& template_args) {
    using json = nlohmann::json;

    json context = json::object();

    // Caller-supplied first, so the reserved keys below always win even if the
    // rejection above is ever relaxed.
    if (!template_args.is_null()) {
        if (!template_args.is_object()) {
            return make_error("Chat template arguments must be a JSON object");
        }
        for (const auto& [key, value] : template_args.items()) {
            if (is_reserved_context_key(key)) {
                return make_error("Chat template argument '" + key +
                                  "' is reserved by the template engine");
            }
            context[key] = value;
        }
    }

    context["messages"] = std::move(messages);
    context["add_generation_prompt"] = add_generation_prompt;
    context["bos_token"] = bos_token ? *bos_token : std::string{};
    context["eos_token"] = eos_token ? *eos_token : std::string{};

    try {
        auto tokens = jinja::tokenize_template(template_str);
        auto ast = jinja::parse(tokens);
        return jinja::render(*ast, context);
    } catch (const std::exception& e) {
        return make_error(std::string("Template error: ") + e.what());
    }
}

}  // namespace

Result<std::string> ChatTemplate::apply(const std::vector<ChatMessage>& messages,
                                        bool add_generation_prompt,
                                        const nlohmann::json& template_args) const {
    nlohmann::json msgs = nlohmann::json::array();
    for (const auto& msg : messages) {
        nlohmann::json m = nlohmann::json::object();
        m["role"] = msg.role;
        m["content"] = msg.content;
        msgs.push_back(m);
    }
    return render_template(template_str_, bos_token_, eos_token_, std::move(msgs),
                           add_generation_prompt, template_args);
}

Result<std::string> ChatTemplate::apply_json(const nlohmann::json& messages_json,
                                             bool add_generation_prompt,
                                             const nlohmann::json& template_args) const {
    return render_template(template_str_, bos_token_, eos_token_, messages_json,
                           add_generation_prompt, template_args);
}

} // namespace tokenizers
