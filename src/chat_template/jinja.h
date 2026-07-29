#pragma once
/// @file chat_template/jinja.h
/// Minimal Jinja2 template engine for HuggingFace chat templates.
/// Token-based lexer, recursive-descent parser, variant-based AST.

#include "tokenizers/error.h"
#include "tokenizers/common.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace tokenizers::jinja {

using json = nlohmann::json;

// ============================================================================
// Template tokens (template-level lexer output)
// ============================================================================

enum class TemplateTokenType {
    Text,       // literal text
    VarExpr,    // {{ expr }}
    BlockExpr,  // {% statement %}
};

struct TemplateToken {
    TemplateTokenType type;
    std::string value;
    bool trim_left = false;   // {%- or {{-
    bool trim_right = false;  // -%} or -}}
};

/// Tokenize a Jinja2 template string into template tokens.
std::vector<TemplateToken> tokenize_template(std::string_view src);

// ============================================================================
// Expression tokens (expression-level lexer output)
// ============================================================================

enum class TokenKind {
    Ident, Integer, Float, String,
    KW_True, KW_False, KW_None,
    KW_And, KW_Or, KW_Not, KW_In, KW_Is,
    KW_If, KW_Else, KW_For, KW_Set, KW_Endfor, KW_Endif, KW_Elif,
    Plus, Minus, Star, Slash, DoubleSlash, Percent, DoubleStar, Tilde,
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    LParen, RParen, LBracket, RBracket, LBrace, RBrace,
    Dot, Comma, Colon, Pipe, Assign,
    Eof,
};

struct Token {
    TokenKind kind;
    std::string text;
    json value;  // for Integer, Float, String literals
};

// ============================================================================
// Operator enums — used in AST nodes for switch-based dispatch
// ============================================================================

// Test operations — used in 'is' expressions (e.g., "is defined")
enum class TestOp {
    Defined, Undefined, None, True, False, Boolean,
    String, Number, Integer, Float,
    Mapping, Iterable, Sequence, Callable,
    Even, Odd,
    Eq, Ne,
};

// Filter identifiers — used in pipe expressions (e.g., "| trim")
enum class FilterId {
    Trim, Length, Default, First, Last,
    Upper, Lower, Title, Join, List,
    Int, Float, String, ToJson,
    Replace, Map, SelectAttr, Batch, Reverse,
    Sort, Reject, RejectAttr, Select, Abs,
    Round, Truncate, Indent, Capitalize,
    Unique, DictSort, Items,
    Unknown,
};

enum class BinOp {
    // Arithmetic
    Add, Sub, Mul, Div, FloorDiv, Mod, Pow,
    // Comparison
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    // Logical (short-circuit)
    And, Or,
    // Membership
    In, NotIn,
    // String
    Concat,  // ~
};

enum class UnOp {
    Not,
    Neg,  // unary minus
};

// ============================================================================
// AST nodes — variant-based expression tree
// ============================================================================

struct ExprNode;
using ExprPtr = std::unique_ptr<ExprNode>;

struct LiteralExpr  { json value; };
struct IdentExpr    { std::string name; };
struct BinaryExpr   { BinOp op; ExprPtr left, right; };
struct UnaryExpr    { UnOp op; ExprPtr operand; };
struct GetAttrExpr  { ExprPtr object; std::string attr; };
struct GetItemExpr  { ExprPtr object; ExprPtr key; };
struct SliceExpr    { ExprPtr object; ExprPtr start, end, step; };
struct CallExpr     {
    ExprPtr callee;
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, ExprPtr>> kwargs;
};
struct FilterExpr   {
    ExprPtr value;
    FilterId filter;
    std::string name;  // original string for error messages
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, ExprPtr>> kwargs;
};
struct TestExpr     { ExprPtr value; TestOp test; bool negated; };
struct CondExpr     { ExprPtr true_val, condition, false_val; };
struct ListExpr     { std::vector<ExprPtr> items; };
struct DictExpr     { std::vector<std::pair<ExprPtr, ExprPtr>> items; };

using ExprVariant = std::variant<
    LiteralExpr, IdentExpr, BinaryExpr, UnaryExpr,
    GetAttrExpr, GetItemExpr, SliceExpr,
    CallExpr, FilterExpr, TestExpr, CondExpr,
    ListExpr, DictExpr
>;

struct ExprNode : ExprVariant {
    using ExprVariant::ExprVariant;
    using ExprVariant::operator=;
};

// ============================================================================
// Statement nodes
// ============================================================================

struct Node;
using NodePtr = std::unique_ptr<Node>;

enum class NodeType {
    Text,
    Output,   // {{ expr }}
    For,
    If,
    Set,
    Macro,    // {% macro name(params) %} ... {% endmacro %}
    Block,    // sequence of nodes
};

struct IfBranch {
    ExprPtr condition;  // nullptr for else
    std::vector<NodePtr> body;
};

struct Node {
    NodeType type;
    std::string text;                 // for Text
    ExprPtr expr;                     // for Output, Set value (nullptr => block set)
    std::string var_name;             // for For (loop var), Set (var name), Macro (name)
    ExprPtr iterable;                 // for For
    std::vector<NodePtr> body;        // for For, Block, Macro, block-Set
    std::vector<IfBranch> branches;   // for If (if/elif/else)
    // for Macro: parameter names with optional default-value expressions
    std::vector<std::pair<std::string, ExprPtr>> params;
};

/// Parse template tokens into an AST.
NodePtr parse(const std::vector<TemplateToken>& tokens);

/// Parse an expression string into an expression AST.
ExprPtr parse_expression(std::string_view src);

// ============================================================================
// Evaluation
// ============================================================================

/// Render an AST against a JSON context.
tokenizers::Result<std::string> render(const Node& root, const json& context);

} // namespace tokenizers::jinja
