#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "tokenizers/chat_template.h"

using namespace tokenizers;
using json = nlohmann::json;

// ============================================================================
// Basic functionality
// ============================================================================

TEST(ChatTemplateTest, SimpleTemplate) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}Assistant: {% endif %}");

    std::vector<ChatMessage> msgs = {{"user", "Hello"}, {"assistant", "Hi!"}};
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("User: Hello") != std::string::npos);
    EXPECT_TRUE(result->find("Assistant: Hi!") != std::string::npos);
    // Should end with generation prompt
    auto last_assistant = result->rfind("Assistant: ");
    EXPECT_NE(last_assistant, std::string::npos);
}

TEST(ChatTemplateTest, SpecialTokens) {
    ChatTemplate ct(
        "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}{{ eos_token }}",
        "<bos>", "<eos>");
    auto result = ct.apply({{"user", "Hello"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "<bos>Hello<eos>");
}

TEST(ChatTemplateTest, NoSpecialTokens) {
    ChatTemplate ct("{% for m in messages %}{{ m['content'] }}{% endfor %}");
    auto result = ct.apply({{"user", "Hi"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "Hi");
}

// ============================================================================
// Filters
// ============================================================================

TEST(ChatTemplateTest, FilterLength) {
    ChatTemplate ct("{{ messages | length }}");
    auto result = ct.apply({{"user", "a"}, {"user", "b"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "2");
}

TEST(ChatTemplateTest, FilterTrim) {
    ChatTemplate ct("{{ '  hello  ' | trim }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello");
}

TEST(ChatTemplateTest, FilterUpper) {
    ChatTemplate ct("{{ 'hello' | upper }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "HELLO");
}

TEST(ChatTemplateTest, FilterLower) {
    ChatTemplate ct("{{ 'HELLO' | lower }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello");
}

TEST(ChatTemplateTest, FilterTitle) {
    ChatTemplate ct("{{ 'hello world' | title }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "Hello World");
}

TEST(ChatTemplateTest, FilterDefault) {
    ChatTemplate ct("{{ undefined_var | default('fallback') }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "fallback");
}

TEST(ChatTemplateTest, FilterFirst) {
    ChatTemplate ct("{{ messages | first }}");
    auto result = ct.apply({{"user", "first_msg"}, {"user", "second"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    // First element is a JSON object, dumped as string
    EXPECT_TRUE(result->find("first_msg") != std::string::npos);
}

TEST(ChatTemplateTest, FilterJoin) {
    ChatTemplate ct("{% set items = ['a', 'b', 'c'] %}{{ items | join(', ') }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "a, b, c");
}

TEST(ChatTemplateTest, FilterReplace) {
    ChatTemplate ct("{{ 'hello world' | replace('world', 'there') }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello there");
}

// ============================================================================
// Loop variables
// ============================================================================

TEST(ChatTemplateTest, LoopIndex) {
    ChatTemplate ct("{% for m in messages %}{{ loop.index }}{% endfor %}");
    auto result = ct.apply({{"user", "a"}, {"user", "b"}, {"user", "c"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "123");
}

TEST(ChatTemplateTest, LoopIndex0) {
    ChatTemplate ct("{% for m in messages %}{{ loop.index0 }}{% endfor %}");
    auto result = ct.apply({{"user", "a"}, {"user", "b"}, {"user", "c"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "012");
}

TEST(ChatTemplateTest, LoopFirst) {
    ChatTemplate ct(
        "{% for m in messages %}"
        "{% if loop.first %}FIRST{% endif %}"
        "{{ m['content'] }}"
        "{% endfor %}");
    auto result = ct.apply({{"user", "a"}, {"user", "b"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("FIRSTa") != std::string::npos);
    // "b" should not be preceded by FIRST
    auto pos_first = result->find("FIRST");
    auto pos_b = result->find("b");
    // There should only be one FIRST
    EXPECT_EQ(result->find("FIRST", pos_first + 1), std::string::npos);
}

TEST(ChatTemplateTest, LoopLast) {
    ChatTemplate ct(
        "{% for m in messages %}"
        "{{ m['content'] }}"
        "{% if not loop.last %},{% endif %}"
        "{% endfor %}");
    auto result = ct.apply({{"user", "a"}, {"user", "b"}, {"user", "c"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "a,b,c");
}

TEST(ChatTemplateTest, LoopLength) {
    ChatTemplate ct("{% for m in messages %}{{ loop.length }}{% endfor %}");
    auto result = ct.apply({{"user", "a"}, {"user", "b"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "22");
}

// ============================================================================
// Whitespace control
// ============================================================================

TEST(ChatTemplateTest, WhitespaceControlBothSides) {
    ChatTemplate ct("  {{- 'hello' -}}  ");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello");
}

TEST(ChatTemplateTest, WhitespaceControlLeft) {
    ChatTemplate ct("  {{- 'hello' }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello");
}

TEST(ChatTemplateTest, WhitespaceControlRight) {
    ChatTemplate ct("{{ 'hello' -}}  ");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello");
}

TEST(ChatTemplateTest, WhitespaceControlBlocks) {
    ChatTemplate ct("  {%- if true -%}  hello  {%- endif -%}  ");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello");
}

// ============================================================================
// Set statement
// ============================================================================

TEST(ChatTemplateTest, SetVariable) {
    ChatTemplate ct("{% set x = 'hello' %}{{ x }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello");
}

TEST(ChatTemplateTest, SetWithExpression) {
    ChatTemplate ct("{% set x = 1 + 2 %}{{ x }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "3");
}

// ============================================================================
// Conditionals
// ============================================================================

TEST(ChatTemplateTest, IfTrue) {
    ChatTemplate ct("{% if true %}yes{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

TEST(ChatTemplateTest, IfFalse) {
    ChatTemplate ct("{% if false %}yes{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "");
}

TEST(ChatTemplateTest, IfElse) {
    ChatTemplate ct("{% if false %}yes{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "no");
}

TEST(ChatTemplateTest, IfElif) {
    ChatTemplate ct("{% if false %}a{% elif true %}b{% else %}c{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "b");
}

TEST(ChatTemplateTest, IfWithComparison) {
    ChatTemplate ct("{% if 1 == 1 %}equal{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "equal");
}

TEST(ChatTemplateTest, IfNotEqual) {
    ChatTemplate ct("{% if 1 != 2 %}different{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "different");
}

// ============================================================================
// Boolean operators
// ============================================================================

TEST(ChatTemplateTest, BooleanAnd) {
    ChatTemplate ct("{% if true and true %}yes{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

TEST(ChatTemplateTest, BooleanOr) {
    ChatTemplate ct("{% if false or true %}yes{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

TEST(ChatTemplateTest, BooleanNot) {
    ChatTemplate ct("{% if not false %}yes{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

// ============================================================================
// String operations
// ============================================================================

TEST(ChatTemplateTest, StringConcatenation) {
    ChatTemplate ct("{{ 'hello' + ' ' + 'world' }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello world");
}

TEST(ChatTemplateTest, StringComparison) {
    ChatTemplate ct("{% if 'a' == 'a' %}same{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "same");
}

// ============================================================================
// Dict/list access
// ============================================================================

TEST(ChatTemplateTest, DictBracketAccess) {
    ChatTemplate ct("{% for m in messages %}{{ m['role'] }}{% endfor %}");
    auto result = ct.apply({{"user", "Hi"}, {"assistant", "Hello"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "userassistant");
}

TEST(ChatTemplateTest, DictDotAccess) {
    ChatTemplate ct("{% for m in messages %}{{ m.role }}{% endfor %}");
    auto result = ct.apply({{"user", "Hi"}, {"assistant", "Hello"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "userassistant");
}

TEST(ChatTemplateTest, ListIndexAccess) {
    ChatTemplate ct("{{ messages[0]['content'] }}");
    auto result = ct.apply({{"user", "first"}, {"assistant", "second"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "first");
}

TEST(ChatTemplateTest, NegativeIndex) {
    ChatTemplate ct("{{ messages[-1]['content'] }}");
    auto result = ct.apply({{"user", "first"}, {"assistant", "last"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "last");
}

// ============================================================================
// Truthiness
// ============================================================================

TEST(ChatTemplateTest, EmptyStringIsFalsy) {
    ChatTemplate ct("{% if '' %}yes{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "no");
}

TEST(ChatTemplateTest, ZeroIsFalsy) {
    ChatTemplate ct("{% if 0 %}yes{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "no");
}

TEST(ChatTemplateTest, NonEmptyStringIsTruthy) {
    ChatTemplate ct("{% if 'hello' %}yes{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

// ============================================================================
// Raise exception
// ============================================================================

TEST(ChatTemplateTest, RaiseException) {
    ChatTemplate ct("{{ raise_exception('test error') }}");
    auto result = ct.apply({}, false);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().message().find("test error") != std::string::npos);
}

// ============================================================================
// Real-world template patterns
// ============================================================================

TEST(ChatTemplateTest, Llama2Style) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<<SYS>>{{ message['content'] }}<</SYS>>"
        "{% elif message['role'] == 'user' %}"
        "[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}");

    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful."},
        {"user", "Hello"},
        {"assistant", "Hi there!"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("<<SYS>>You are helpful.<</SYS>>") != std::string::npos);
    EXPECT_TRUE(result->find("[INST] Hello [/INST]") != std::string::npos);
    EXPECT_TRUE(result->find("Hi there!") != std::string::npos);
}

TEST(ChatTemplateTest, ChatMLStyle) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}");

    std::vector<ChatMessage> msgs = {
        {"user", "Hello"},
        {"assistant", "Hi!"},
        {"user", "How are you?"}};
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("<|im_start|>user\nHello<|im_end|>") != std::string::npos);
    EXPECT_TRUE(result->find("<|im_start|>assistant\nHi!<|im_end|>") != std::string::npos);
    EXPECT_TRUE(result->find("<|im_start|>assistant\n") != std::string::npos);
}

TEST(ChatTemplateTest, WithWhitespaceTrimming) {
    ChatTemplate ct(
        "{%- for message in messages -%}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{%- endfor -%}");

    std::vector<ChatMessage> msgs = {
        {"user", "Hi"},
        {"assistant", "Hello"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    // The \n before {%- endfor -%} gets trimmed by the left trim on endfor
    EXPECT_EQ(*result, "user: Hiassistant: Hello");
}

TEST(ChatTemplateTest, SpecialTokensInTemplate) {
    ChatTemplate ct(
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% if not loop.last %} {% endif %}"
        "{% endfor %}"
        "{{ eos_token }}",
        "<s>", "</s>");

    auto result = ct.apply({{"user", "Hello"}, {"assistant", "Hi"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "<s>Hello Hi</s>");
}

// ============================================================================
// Ternary (inline if)
// ============================================================================

TEST(ChatTemplateTest, TernaryExpression) {
    ChatTemplate ct("{{ 'yes' if true else 'no' }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

TEST(ChatTemplateTest, TernaryExpressionFalse) {
    ChatTemplate ct("{{ 'yes' if false else 'no' }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "no");
}

// ============================================================================
// List literal
// ============================================================================

TEST(ChatTemplateTest, ListLiteral) {
    ChatTemplate ct("{% set items = [1, 2, 3] %}{{ items | length }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "3");
}

// ============================================================================
// In operator
// ============================================================================

TEST(ChatTemplateTest, InOperator) {
    ChatTemplate ct("{% if 'user' in ['user', 'assistant'] %}found{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "found");
}

TEST(ChatTemplateTest, NotInOperator) {
    ChatTemplate ct("{% if 'system' not in ['user', 'assistant'] %}not found{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "not found");
}

// ============================================================================
// Is defined / is none
// ============================================================================

TEST(ChatTemplateTest, IsDefined) {
    ChatTemplate ct("{% if bos_token is defined %}yes{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

TEST(ChatTemplateTest, IsNotDefined) {
    ChatTemplate ct("{% if unknown_var is defined %}yes{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "no");
}

TEST(ChatTemplateTest, StringMethods) {
    ChatTemplate ct("{{ 'hello world'.upper() }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "HELLO WORLD");
}

// ============================================================================
// Tilde operator (string concat)
// ============================================================================

TEST(ChatTemplateTest, TildeOperator) {
    ChatTemplate ct("{{ 'hello' ~ ' ' ~ 'world' }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "hello world");
}

// ============================================================================
// Empty messages
// ============================================================================

TEST(ChatTemplateTest, EmptyMessages) {
    ChatTemplate ct("{% for m in messages %}{{ m['content'] }}{% endfor %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "");
}

// ============================================================================
// Nested conditionals
// ============================================================================

TEST(ChatTemplateTest, NestedConditionals) {
    ChatTemplate ct(
        "{% for m in messages %}"
        "{% if m['role'] == 'user' %}"
        "{% if loop.first %}FIRST {% endif %}"
        "User: {{ m['content'] }}\n"
        "{% endif %}"
        "{% endfor %}");
    auto result = ct.apply({{"user", "Hello"}, {"user", "World"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("FIRST User: Hello") != std::string::npos);
    // Second message should not have FIRST
    auto second_user = result->find("User: World");
    EXPECT_NE(second_user, std::string::npos);
}

// ============================================================================
// Comment handling
// ============================================================================

TEST(ChatTemplateTest, Comments) {
    ChatTemplate ct("before{# this is a comment #}after");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "beforeafter");
}

// ============================================================================
// Real HuggingFace model templates
// ============================================================================

TEST(RealTemplateTest, Phi3_UserOnly) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{'<|system|>\n' + message['content'] + '<|end|>\n'}}"
        "{% elif message['role'] == 'user' %}"
        "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}"
        "{% elif message['role'] == 'assistant' %}"
        "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
        std::nullopt, "<|endoftext|>");

    auto result = ct.apply({{"user", "Hello"}}, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "<|user|>\nHello<|end|>\n<|assistant|>\n");
}

TEST(RealTemplateTest, Phi3_MultiTurn) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{'<|system|>\n' + message['content'] + '<|end|>\n'}}"
        "{% elif message['role'] == 'user' %}"
        "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}"
        "{% elif message['role'] == 'assistant' %}"
        "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
        std::nullopt, "<|endoftext|>");

    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful."},
        {"user", "Hello"},
        {"assistant", "Hi!"},
        {"user", "How are you?"}};
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result,
        "<|system|>\nYou are helpful.<|end|>\n"
        "<|user|>\nHello<|end|>\n"
        "<|assistant|>\nHi!<|end|>\n"
        "<|user|>\nHow are you?<|end|>\n"
        "<|assistant|>\n");
}

TEST(RealTemplateTest, Phi3_NoGenPrompt) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{'<|system|>\n' + message['content'] + '<|end|>\n'}}"
        "{% elif message['role'] == 'user' %}"
        "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}"
        "{% elif message['role'] == 'assistant' %}"
        "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
        std::nullopt, "<|endoftext|>");

    auto result = ct.apply({{"user", "Hello"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "<|user|>\nHello<|end|>\n<|endoftext|>");
}

TEST(RealTemplateTest, Qwen2_UserOnly) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
        "{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}");

    auto result = ct.apply({{"user", "Hello"}}, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n");
}

TEST(RealTemplateTest, Qwen2_WithSystemMessage) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
        "{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}");

    std::vector<ChatMessage> msgs = {
        {"system", "Be concise."},
        {"user", "Hello"}};
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    // Should NOT inject default system message since one is provided
    EXPECT_EQ(*result,
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n");
}

TEST(RealTemplateTest, Qwen2_NoGenPrompt) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
        "{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}");

    auto result = ct.apply({{"user", "Hello"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n");
}

TEST(RealTemplateTest, ChatML_Standard) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");

    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful."},
        {"user", "Hello"},
        {"assistant", "Hi there!"}};
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result,
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi there!<|im_end|>\n"
        "<|im_start|>assistant\n");
}

TEST(RealTemplateTest, ChatML_NoGenPrompt) {
    ChatTemplate ct(
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");

    auto result = ct.apply({{"user", "What is 2+2?"}}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result,
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n");
}

// ============================================================================
// List slicing (needed for Mistral and many other templates)
// ============================================================================

TEST(RealTemplateTest, ListSlice_StartToEnd) {
    // messages[1:] — skip first element
    ChatTemplate ct("{% for m in messages[1:] %}{{ m['content'] }},{% endfor %}");
    std::vector<ChatMessage> msgs = {{"user", "first"}, {"user", "second"}, {"user", "third"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "second,third,");
}

TEST(RealTemplateTest, ListSlice_BeginToPos) {
    // messages[:2] — first two elements
    ChatTemplate ct("{% for m in messages[:2] %}{{ m['content'] }},{% endfor %}");
    std::vector<ChatMessage> msgs = {{"user", "first"}, {"user", "second"}, {"user", "third"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "first,second,");
}

TEST(RealTemplateTest, ListSlice_Range) {
    // messages[1:3]
    ChatTemplate ct("{% for m in messages[1:3] %}{{ m['content'] }},{% endfor %}");
    std::vector<ChatMessage> msgs = {{"user", "a"}, {"user", "b"}, {"user", "c"}, {"user", "d"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "b,c,");
}

TEST(RealTemplateTest, ListSlice_NegativeStart) {
    // messages[-2:]
    ChatTemplate ct("{% for m in messages[-2:] %}{{ m['content'] }},{% endfor %}");
    std::vector<ChatMessage> msgs = {{"user", "a"}, {"user", "b"}, {"user", "c"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "b,c,");
}

// ============================================================================
// selectattr with test (needed for Mistral)
// ============================================================================

TEST(RealTemplateTest, SelectAttr_Equalto) {
    ChatTemplate ct(
        "{% set users = messages | selectattr('role', 'equalto', 'user') | list %}"
        "{{ users | length }}");
    std::vector<ChatMessage> msgs = {
        {"system", "sys"}, {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "2");
}

// ============================================================================
// namespace() mutations (needed for Mistral and other complex templates)
// ============================================================================

TEST(RealTemplateTest, Namespace_BasicMutation) {
    ChatTemplate ct(
        "{% set ns = namespace(found=false) %}"
        "{% for m in messages %}"
        "{% if m['role'] == 'user' %}"
        "{% set ns.found = true %}"
        "{% endif %}"
        "{% endfor %}"
        "{{ ns.found }}");
    std::vector<ChatMessage> msgs = {{"system", "sys"}, {"user", "hello"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "True");
}

TEST(RealTemplateTest, Namespace_CountInLoop) {
    ChatTemplate ct(
        "{% set ns = namespace(count=0) %}"
        "{% for m in messages %}"
        "{% set ns.count = ns.count + 1 %}"
        "{% endfor %}"
        "{{ ns.count }}");
    std::vector<ChatMessage> msgs = {{"user", "a"}, {"user", "b"}, {"user", "c"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "3");
}

// ============================================================================
// Mistral-style basic flow (simplified)
// ============================================================================

TEST(RealTemplateTest, MistralBasicFlow) {
    // Simplified Mistral template: handles system message + user/assistant alternation
    ChatTemplate ct(
        "{% if messages[0]['role'] == 'system' %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{% set system_message = '' %}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "[INST] {% if system_message != '' %}{{ system_message }}\n\n{% endif %}"
        "{{ message['content'] }} [/INST]"
        "{% set system_message = '' %}"
        "{% elif message['role'] == 'assistant' %}"
        " {{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}");

    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful."},
        {"user", "Hello"},
        {"assistant", "Hi!"},
        {"user", "Bye"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    // System message should appear before first user message
    EXPECT_TRUE(result->find("You are helpful.") != std::string::npos);
    EXPECT_TRUE(result->find("[INST]") != std::string::npos);
    EXPECT_TRUE(result->find("Hello") != std::string::npos);
    EXPECT_TRUE(result->find("Hi!") != std::string::npos);
}

// ============================================================================
// Jinja "is string" / "is iterable" tests
// ============================================================================

TEST(ChatTemplateTest, IsStringTest) {
    ChatTemplate ct("{% if 'hello' is string %}yes{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "yes");
}

TEST(ChatTemplateTest, IsNotStringOnList) {
    // A list value should satisfy "is not string"
    ChatTemplate ct(
        "{% set x = [1, 2] %}"
        "{% if x is not string %}not_str{% else %}str{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "not_str");
}

TEST(ChatTemplateTest, IsIterableOnList) {
    ChatTemplate ct(
        "{% set x = [1, 2] %}"
        "{% if x is iterable %}iter{% else %}no{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "iter");
}

TEST(ChatTemplateTest, IsNotIterableOnNull) {
    ChatTemplate ct(
        "{% if none is not iterable %}not_iter{% else %}iter{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "not_iter");
}

// ============================================================================
// JSON apply() overload
// ============================================================================

TEST(ChatTemplateTest, JsonApplyOverload) {
    ChatTemplate ct(
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'][0]['text'] }}\n{% endfor %}",
        std::nullopt, std::nullopt);
    json messages = json::array({
        {{"role", "user"}, {"content", json::array({{{"text", "Hello"}}})}},
    });
    auto result = ct.apply_json(messages, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("user: Hello") != std::string::npos);
}

// ============================================================================
// Gemma translation chat template (loaded from file)
// ============================================================================

namespace {

std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string test_dir() {
    // __FILE__ points to tests/test_chat_template.cpp
    std::filesystem::path p(__FILE__);
    return p.parent_path().string();
}

const std::string GEMMA3_TEMPLATE = read_file(test_dir() + "/chat-templates/gemma3.jinja");
const std::string TRANSLATE_GEMMA_TEMPLATE = read_file(test_dir() + "/chat-templates/translate-gemma.jinja");
const std::string GEMMA4_TEMPLATE = read_file(test_dir() + "/chat-templates/gemma4.jinja");
const std::string QWEN35_TEMPLATE = read_file(test_dir() + "/chat-templates/qwen3_5.jinja");


} // anonymous namespace

TEST(GemmaTranslateTest, SingleUserTurn) {
    ASSERT_FALSE(TRANSLATE_GEMMA_TEMPLATE.empty()) << "Failed to read template file";

    ChatTemplate ct(TRANSLATE_GEMMA_TEMPLATE, "<bos>", "<eos>");

    json messages = json::array({
        {{"role", "user"},
         {"content", json::array({
             {{"type", "text"},
              {"source_lang_code", "en"},
              {"target_lang_code", "fr"},
              {"text", "Hello world"}}
         })}}
    });

    auto result = ct.apply_json(messages, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    EXPECT_TRUE(result->find("<bos>") != std::string::npos);
    EXPECT_TRUE(result->find("<start_of_turn>user") != std::string::npos);
    EXPECT_TRUE(result->find("English") != std::string::npos);
    EXPECT_TRUE(result->find("French") != std::string::npos);
    EXPECT_TRUE(result->find("Hello world") != std::string::npos);
    EXPECT_TRUE(result->find("<end_of_turn>") != std::string::npos);
    EXPECT_TRUE(result->find("<start_of_turn>model") != std::string::npos);
}

TEST(GemmaTranslateTest, MultiTurnWithAssistant) {
    ASSERT_FALSE(TRANSLATE_GEMMA_TEMPLATE.empty()) << "Failed to read template file";

    ChatTemplate ct(TRANSLATE_GEMMA_TEMPLATE, "<bos>", "<eos>");

    json messages = json::array({
        {{"role", "user"},
         {"content", json::array({
             {{"type", "text"},
              {"source_lang_code", "en"},
              {"target_lang_code", "fr"},
              {"text", "Hello world"}}
         })}},
        {{"role", "assistant"},
         {"content", "Bonjour le monde"}}
    });

    auto result = ct.apply_json(messages, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    EXPECT_TRUE(result->find("<start_of_turn>model") != std::string::npos);
    EXPECT_TRUE(result->find("Bonjour le monde") != std::string::npos);
}

TEST(GemmaTranslateTest, UnderscoreLangCode) {
    ASSERT_FALSE(TRANSLATE_GEMMA_TEMPLATE.empty()) << "Failed to read template file";

    ChatTemplate ct(TRANSLATE_GEMMA_TEMPLATE, "<bos>", "<eos>");

    // Use underscore lang code — template uses replace("_", "-")
    json messages = json::array({
        {{"role", "user"},
         {"content", json::array({
             {{"type", "text"},
              {"source_lang_code", "pt_BR"},
              {"target_lang_code", "es"},
              {"text", "Olá mundo"}}
         })}}
    });

    auto result = ct.apply_json(messages, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // pt_BR should become pt-BR and map to Portuguese
    EXPECT_TRUE(result->find("Portuguese") != std::string::npos);
    EXPECT_TRUE(result->find("Spanish") != std::string::npos);
    EXPECT_TRUE(result->find("pt-BR") != std::string::npos);
}

// ============================================================================
// Gemma3 standard chat template (the one used by google/gemma-3-1b-it)
// Reference outputs verified against HuggingFace transformers Python library.
// ============================================================================


TEST(Gemma3ChatTest, SimpleUserMessage) {
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {{"user", "Hello, world!"}};
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // Reference: '<bos><start_of_turn>user\nHello, world!<end_of_turn>\n<start_of_turn>model\n'
    EXPECT_EQ(*result, "<bos><start_of_turn>user\nHello, world!<end_of_turn>\n<start_of_turn>model\n");
}

TEST(Gemma3ChatTest, SimpleUserMessageNoGen) {
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {{"user", "Hello, world!"}};
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // Reference: '<bos><start_of_turn>user\nHello, world!<end_of_turn>\n'
    EXPECT_EQ(*result, "<bos><start_of_turn>user\nHello, world!<end_of_turn>\n");
}

TEST(Gemma3ChatTest, SystemPlusUser) {
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {
        {"system", "You are a helpful assistant."},
        {"user", "What is 2+2?"}
    };
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // Reference: '<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n'
    EXPECT_EQ(*result, "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n");
}

TEST(Gemma3ChatTest, SystemPlusUserNoGen) {
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {
        {"system", "You are a helpful assistant."},
        {"user", "What is 2+2?"}
    };
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    EXPECT_EQ(*result, "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWhat is 2+2?<end_of_turn>\n");
}

TEST(Gemma3ChatTest, MultiTurn) {
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {
        {"user", "Hi"},
        {"assistant", "Hello!"},
        {"user", "How are you?"}
    };
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // Reference: '<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nHow are you?<end_of_turn>\n<start_of_turn>model\n'
    EXPECT_EQ(*result, "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nHow are you?<end_of_turn>\n<start_of_turn>model\n");
}

TEST(Gemma3ChatTest, MultiTurnNoGen) {
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {
        {"user", "Hi"},
        {"assistant", "Hello!"},
        {"user", "How are you?"}
    };
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    EXPECT_EQ(*result, "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nHow are you?<end_of_turn>\n");
}

TEST(Gemma3ChatTest, AssistantRoleMappedToModel) {
    // Gemma3 maps "assistant" -> "model" in the output
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {
        {"user", "Test"},
        {"assistant", "Response"}
    };
    auto result = ct.apply(msgs, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // "assistant" role should appear as "model" in output
    EXPECT_TRUE(result->find("<start_of_turn>model") != std::string::npos);
    EXPECT_TRUE(result->find("<start_of_turn>assistant") == std::string::npos);
}

TEST(Gemma3ChatTest, SystemMessagePrefixedToFirstUser) {
    // System message content is prepended to first user message, not as separate turn
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {
        {"system", "Be concise."},
        {"user", "Explain ML."}
    };
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // System content + "\n\n" + user content in the same turn
    EXPECT_TRUE(result->find("Be concise.\n\nExplain ML.") != std::string::npos);
    // No separate "system" turn
    EXPECT_TRUE(result->find("<start_of_turn>system") == std::string::npos);
}

TEST(Gemma3ChatTest, WhitespaceTrimming) {
    // The template uses {{ message['content'] | trim }}
    ChatTemplate ct(GEMMA3_TEMPLATE, "<bos>", "<eos>");

    std::vector<ChatMessage> msgs = {{"user", "  Hello  "}};
    auto result = ct.apply(msgs, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    // Content should be trimmed
    EXPECT_TRUE(result->find("Hello<end_of_turn>") != std::string::npos);
    // No leading/trailing spaces around "Hello"
    EXPECT_TRUE(result->find("  Hello") == std::string::npos);
}

// ============================================================================
// Macros ({% macro %} ... {% endmacro %})
// ============================================================================

TEST(JinjaMacroTest, BasicDefineAndCall) {
    ChatTemplate ct("{%- macro greet(x) -%}Hello {{ x }}!{%- endmacro -%}{{ greet('world') }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "Hello world!");
}

TEST(JinjaMacroTest, DefaultParametersAndKwargs) {
    ChatTemplate ct(
        "{%- macro g(x, y=true) -%}{{ x }}/{{ y }}{%- endmacro -%}"
        "{{ g('a') }}|{{ g('a', false) }}|{{ g('a', y=false) }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "a/True|a/False|a/False");
}

TEST(JinjaMacroTest, Recursion) {
    ChatTemplate ct(
        "{%- macro fmt(v) -%}"
        "{%- if v is sequence and v is not string -%}"
        "[{% for i in v %}{{ fmt(i) }}{% if not loop.last %},{% endif %}{% endfor %}]"
        "{%- else -%}{{ v }}{%- endif -%}"
        "{%- endmacro -%}{{ fmt([1, [2, 3], 4]) }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "[1,[2,3],4]");
}

TEST(JinjaMacroTest, MutatesEnclosingNamespace) {
    // A macro can mutate a namespace declared in the enclosing (global) scope.
    ChatTemplate ct(
        "{%- set ns = namespace(count=0) -%}"
        "{%- macro bump() -%}{%- set ns.count = ns.count + 1 -%}{%- endmacro -%}"
        "{{ bump() }}{{ bump() }}{{ bump() }}{{ ns.count }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "3");
}

// Unbounded recursion must surface as an error, not blow the stack.
TEST(JinjaMacroTest, RunawayRecursionIsAnError) {
    ChatTemplate ct("{%- macro loop_forever() -%}{{ loop_forever() }}{%- endmacro -%}{{ loop_forever() }}");
    auto result = ct.apply({}, false);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().message().find("recursion too deep"), std::string::npos)
        << result.error().message();
}

// ============================================================================
// Block set ({% set x %} ... {% endset %})
// ============================================================================

TEST(JinjaBlockSetTest, CapturesRenderedBody) {
    ChatTemplate ct("{%- set cap -%}A{{ 'B' }}C{%- endset -%}[{{ cap }}]");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "[ABC]");
}

// ============================================================================
// Slice with step (e.g. messages[::-1])
// ============================================================================

TEST(JinjaSliceTest, StepAndReverse) {
    ChatTemplate ct(
        "{% set xs = [1,2,3,4,5] %}{{ xs[::-1] }}|{{ xs[1:] }}|{{ xs[::2] }}|{{ xs[1:4] }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "[5,4,3,2,1]|[2,3,4,5]|[1,3,5]|[2,3,4]");
}

// Out-of-range bounds and huge steps must stay bounded by the sequence length.
TEST(JinjaSliceTest, OutOfRangeBoundsAreClamped) {
    ChatTemplate ct(
        "{% set xs = [1,2,3] %}"
        "{{ xs[2:-1000000000000:-1] }}|{{ xs[-1000000000000:] }}|"
        "{{ xs[:1000000000000] }}|{{ xs[::1000000000000] }}|{{ xs[::-1000000000000] }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "[3,2,1]|[1,2,3]|[1,2,3]|[1]|[3]");
}

// ============================================================================
// loop.previtem / loop.nextitem
// ============================================================================

TEST(JinjaLoopTest, PrevAndNextItem) {
    ChatTemplate ct(
        "{% for m in messages %}"
        "{{ loop.previtem.role | default('_') }}<{{ m.role }}>{{ loop.nextitem.role | default('_') }} "
        "{% endfor %}");
    json ms = json::array({
        {{"role", "a"}, {"content", "1"}},
        {{"role", "b"}, {"content", "2"}},
        {{"role", "c"}, {"content", "3"}},
    });
    auto result = ct.apply_json(ms, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "_<a>b a<b>c b<c>_ ");
}

// ============================================================================
// New filters / tests / methods
// ============================================================================

TEST(JinjaFilterTest, BooleanTest) {
    ChatTemplate ct("{{ true is boolean }}/{{ 'x' is boolean }}/{{ 1 is boolean }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "True/False/False");
}

TEST(JinjaFilterTest, DictSort) {
    ChatTemplate ct(
        "{% set d = {'b': 2, 'a': 1, 'c': 3} %}"
        "{% for k, v in d | dictsort %}{{ k }}={{ v }};{% endfor %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "a=1;b=2;c=3;");
}

// Jinja's dictsort is case-insensitive by default; case_sensitive=true opts out.
TEST(JinjaFilterTest, DictSortCaseSensitivity) {
    ChatTemplate ct(
        "{% set d = {'B': 2, 'a': 1, 'C': 3} %}"
        "{% for k, v in d | dictsort %}{{ k }};{% endfor %}|"
        "{% for k, v in d | dictsort(true) %}{{ k }};{% endfor %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "a;B;C;|B;C;a;");
}

TEST(JinjaFilterTest, ItemsFilter) {
    ChatTemplate ct(
        "{% set d = {'x': 1, 'y': 2} %}"
        "{% for k, v in d | items %}{{ k }}:{{ v }},{% endfor %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "x:1,y:2,");
}

TEST(JinjaFilterTest, MapWithFilterName) {
    ChatTemplate ct("{{ ['a','b','c'] | map('upper') | list | join('-') }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "A-B-C");
}

TEST(JinjaMethodTest, RstripLstrip) {
    ChatTemplate ct(
        "{{ '  hi\\n'.rstrip() }}|{{ '\\n\\nhi'.lstrip() }}|"
        "{{ 'zzhizz'.rstrip('z') }}|{{ 'zzhizz'.lstrip('z') }}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "  hi|hi|zzhi|hizz");
}

TEST(JinjaTruthinessTest, UndefinedIsFalsy) {
    ChatTemplate ct(
        "{% if undefined_thing %}Y{% else %}N{% endif %}"
        "{% if tools %}T{% else %}F{% endif %}");
    auto result = ct.apply({}, false);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(*result, "NF");
}

// ============================================================================
// Gemma 4 canonical chat template (macros + block set + multimodal markers)
// ============================================================================

TEST(Gemma4ChatTest, TextImageConversationRenders) {
    ASSERT_FALSE(GEMMA4_TEMPLATE.empty()) << "Failed to read gemma4.jinja";
    ChatTemplate ct(GEMMA4_TEMPLATE, "<bos>", "<eos>");

    json messages = json::array({
        {{"role", "user"},
         {"content", json::array({
             {{"type", "text"}, {"text", "What is in this image?"}},
             {{"type", "image"}},
         })}},
        {{"role", "assistant"}, {"content", "A cat."}},
        {{"role", "user"}, {"content", "And this one?"}},
    });

    auto result = ct.apply_json(messages, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("<bos>") != std::string::npos);
    EXPECT_TRUE(result->find("What is in this image?") != std::string::npos);
    EXPECT_TRUE(result->find("<|image|>") != std::string::npos);  // image placeholder
    EXPECT_TRUE(result->find("A cat.") != std::string::npos);
    EXPECT_TRUE(result->find("<|turn>model") != std::string::npos);  // generation prompt
}

TEST(Gemma4ChatTest, AudioAndVideoMarkersRender) {
    ASSERT_FALSE(GEMMA4_TEMPLATE.empty()) << "Failed to read gemma4.jinja";
    ChatTemplate ct(GEMMA4_TEMPLATE, "<bos>", "<eos>");

    json messages = json::array({
        {{"role", "user"},
         {"content", json::array({
             {{"type", "text"}, {"text", "Transcribe this."}},
             {{"type", "audio"}},
             {{"type", "video"}},
         })}},
    });

    auto result = ct.apply_json(messages, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("Transcribe this.") != std::string::npos);
    EXPECT_TRUE(result->find("<|audio|>") != std::string::npos);
    EXPECT_TRUE(result->find("<|video|>") != std::string::npos);
}

// ============================================================================
// Qwen 3.5 chat template (macro + reverse slice + previtem/nextitem)
// ============================================================================

TEST(Qwen35ChatTest, TextImageConversationRenders) {
    ASSERT_FALSE(QWEN35_TEMPLATE.empty()) << "Failed to read qwen3_5.jinja";
    ChatTemplate ct(QWEN35_TEMPLATE, "<bos>", "<eos>");

    json messages = json::array({
        {{"role", "user"},
         {"content", json::array({
             {{"type", "text"}, {"text", "Describe the picture."}},
             {{"type", "image"}},
         })}},
        {{"role", "assistant"}, {"content", "Sure."}},
        {{"role", "user"}, {"content", "Thanks!"}},
    });

    auto result = ct.apply_json(messages, true);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_TRUE(result->find("Describe the picture.") != std::string::npos);
    EXPECT_TRUE(result->find("<|vision_start|><|image_pad|><|vision_end|>") != std::string::npos);
    EXPECT_TRUE(result->find("<|im_start|>assistant") != std::string::npos);
}
