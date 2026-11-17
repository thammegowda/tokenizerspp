#include "tokenizers/pattern.h"

#include <uni_algo/case.h>
#include <uni_algo/prop.h>
#include <uni_algo/script.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tokenizers {

namespace {

// Decode one UTF-8 code point at `p` (with `remaining` > 0 bytes available).
// Invalid sequences return (U+FFFD, 1).
std::pair<char32_t, size_t> decode_utf8(const char* p, size_t remaining) noexcept {
    const auto b0 = static_cast<uint8_t>(p[0]);
    if (b0 < 0x80) return {b0, 1};
    if ((b0 & 0xE0) == 0xC0) {
        if (remaining >= 2) {
            const auto b1 = static_cast<uint8_t>(p[1]);
            if ((b1 & 0xC0) == 0x80) {
                const char32_t cp = (static_cast<char32_t>(b0 & 0x1F) << 6) | (b1 & 0x3F);
                if (cp >= 0x80) return {cp, 2};
            }
        }
        return {0xFFFD, 1};
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (remaining >= 3) {
            const auto b1 = static_cast<uint8_t>(p[1]);
            const auto b2 = static_cast<uint8_t>(p[2]);
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
                const char32_t cp = (static_cast<char32_t>(b0 & 0x0F) << 12) |
                                    (static_cast<char32_t>(b1 & 0x3F) << 6) | (b2 & 0x3F);
                if (cp >= 0x800 && (cp < 0xD800 || cp > 0xDFFF)) return {cp, 3};
            }
        }
        return {0xFFFD, 1};
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (remaining >= 4) {
            const auto b1 = static_cast<uint8_t>(p[1]);
            const auto b2 = static_cast<uint8_t>(p[2]);
            const auto b3 = static_cast<uint8_t>(p[3]);
            if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
                const char32_t cp = (static_cast<char32_t>(b0 & 0x07) << 18) |
                                    (static_cast<char32_t>(b1 & 0x3F) << 12) |
                                    (static_cast<char32_t>(b2 & 0x3F) << 6) | (b3 & 0x3F);
                if (cp >= 0x10000 && cp <= 0x10FFFF) return {cp, 4};
            }
        }
        return {0xFFFD, 1};
    }
    return {0xFFFD, 1};
}

// Decode the UTF-8 code point immediately preceding `pos`.
std::pair<char32_t, size_t> decode_utf8_prev(const char* data, size_t pos) noexcept {
    size_t start = pos - 1;
    const size_t limit = pos >= 4 ? pos - 4 : 0;
    while (start > limit && (static_cast<uint8_t>(data[start]) & 0xC0) == 0x80) {
        --start;
    }
    auto [cp, len] = decode_utf8(data + start, pos - start);
    if (start + len == pos) return {cp, len};
    return {0xFFFD, 1};
}

size_t next_cp_offset(std::string_view s, size_t pos) noexcept {
    if (pos >= s.size()) return s.size() + 1;
    const auto b = static_cast<uint8_t>(s[pos]);
    if (b < 0x80) return pos + 1;
    return pos + decode_utf8(s.data() + pos, s.size() - pos).second;
}

using GC = una::codepoint::general_category;

constexpr uint32_t gc_bit(GC cat) noexcept { return uint32_t{1} << static_cast<unsigned>(cat); }

constexpr uint32_t GC_LU = gc_bit(GC::Lu);
constexpr uint32_t GC_LL = gc_bit(GC::Ll);
constexpr uint32_t GC_LT = gc_bit(GC::Lt);
constexpr uint32_t GC_LM = gc_bit(GC::Lm);
constexpr uint32_t GC_LO = gc_bit(GC::Lo);
constexpr uint32_t GC_LC = GC_LU | GC_LL | GC_LT;
constexpr uint32_t GC_L = GC_LC | GC_LM | GC_LO;

constexpr uint32_t GC_MN = gc_bit(GC::Mn);
constexpr uint32_t GC_MC = gc_bit(GC::Mc);
constexpr uint32_t GC_ME = gc_bit(GC::Me);
constexpr uint32_t GC_M = GC_MN | GC_MC | GC_ME;

constexpr uint32_t GC_ND = gc_bit(GC::Nd);
constexpr uint32_t GC_NL = gc_bit(GC::Nl);
constexpr uint32_t GC_NO = gc_bit(GC::No);
constexpr uint32_t GC_N = GC_ND | GC_NL | GC_NO;

constexpr uint32_t GC_PC = gc_bit(GC::Pc);
constexpr uint32_t GC_PD = gc_bit(GC::Pd);
constexpr uint32_t GC_PS = gc_bit(GC::Ps);
constexpr uint32_t GC_PE = gc_bit(GC::Pe);
constexpr uint32_t GC_PI = gc_bit(GC::Pi);
constexpr uint32_t GC_PF = gc_bit(GC::Pf);
constexpr uint32_t GC_PO = gc_bit(GC::Po);
constexpr uint32_t GC_P = GC_PC | GC_PD | GC_PS | GC_PE | GC_PI | GC_PF | GC_PO;

constexpr uint32_t GC_SM = gc_bit(GC::Sm);
constexpr uint32_t GC_SC = gc_bit(GC::Sc);
constexpr uint32_t GC_SK = gc_bit(GC::Sk);
constexpr uint32_t GC_SO = gc_bit(GC::So);
constexpr uint32_t GC_S = GC_SM | GC_SC | GC_SK | GC_SO;

constexpr uint32_t GC_ZS = gc_bit(GC::Zs);
constexpr uint32_t GC_ZL = gc_bit(GC::Zl);
constexpr uint32_t GC_ZP = gc_bit(GC::Zp);
constexpr uint32_t GC_Z = GC_ZS | GC_ZL | GC_ZP;

constexpr uint32_t GC_CC = gc_bit(GC::Cc);
constexpr uint32_t GC_CF = gc_bit(GC::Cf);
constexpr uint32_t GC_CS = gc_bit(GC::Cs);
constexpr uint32_t GC_CO = gc_bit(GC::Co);
constexpr uint32_t GC_CN = gc_bit(GC::Cn);
constexpr uint32_t GC_C = GC_CC | GC_CF | GC_CS | GC_CO | GC_CN;

enum PropFlags : uint32_t {
    PROP_NONE = 0,
    PROP_WHITESPACE = 1u << 0,
    PROP_ALPHABETIC = 1u << 1,
    PROP_WORD = 1u << 2,
    PROP_ASCII = 1u << 3,
    PROP_ANY = 1u << 4,
    PROP_ASSIGNED = 1u << 5,
};

bool is_unicode_word(char32_t cp, const una::codepoint::prop& p) noexcept {
    if (p.Alphabetic()) return true;
    const uint32_t b = gc_bit(una::codepoint::get_general_category(p));
    if ((b & (GC_M | GC_ND | GC_PC)) != 0) return true;
    return cp == 0x200Cu || cp == 0x200Du;
}

bool is_word_cp(char32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
               (cp >= '0' && cp <= '9') || cp == '_';
    }
    return is_unicode_word(cp, una::codepoint::prop{cp});
}

struct PropertySpec {
    uint32_t gc_mask = 0;
    uint32_t flags = PROP_NONE;
    char32_t script = 0;
    bool negated = false;

    bool matches(char32_t cp, const una::codepoint::prop& p, uint32_t cp_gc_bit) const noexcept {
        bool hit = false;
        if ((gc_mask & cp_gc_bit) != 0) {
            hit = true;
        } else if (flags != PROP_NONE) {
            if ((flags & PROP_WHITESPACE) && p.White_Space()) hit = true;
            else if ((flags & PROP_ALPHABETIC) && p.Alphabetic()) hit = true;
            else if ((flags & PROP_WORD) && is_unicode_word(cp, p)) hit = true;
            else if ((flags & PROP_ASCII) && cp < 0x80) hit = true;
            else if (flags & PROP_ANY) hit = true;
            else if ((flags & PROP_ASSIGNED) && (cp_gc_bit & GC_CN) == 0) hit = true;
        } else if (script != 0) {
            hit = static_cast<char32_t>(una::codepoint::get_script(cp)) == script;
        }
        return negated ? !hit : hit;
    }
};

std::string normalize_prop_key(std::string_view raw) {
    std::string key;
    key.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (ch == ' ' || ch == '_' || ch == '-') continue;
        key.push_back(static_cast<char>(std::tolower(ch)));
    }
    for (std::string_view prefix : {"generalcategory=", "gc=", "script=", "sc="}) {
        if (key.starts_with(prefix)) {
            key.erase(0, prefix.size());
            break;
        }
    }
    return key;
}

constexpr char32_t make_script_tag(char a, char b, char c, char d) noexcept {
    const char buf[5]{a, b, c, d, '\0'};
    return static_cast<char32_t>(una::locale::script{buf});
}

PropertySpec parse_unicode_property(std::string_view raw_name, bool negated) {
    const std::string key = normalize_prop_key(raw_name);
    PropertySpec spec;
    spec.negated = negated;

    struct Entry {
        std::string_view name;
        uint32_t gc;
        uint32_t flags;
    };
    static constexpr std::array<Entry, 62> TABLE{{
        {"l", GC_L, 0}, {"letter", GC_L, 0},
        {"lu", GC_LU, 0}, {"uppercaseletter", GC_LU, 0},
        {"ll", GC_LL, 0}, {"lowercaseletter", GC_LL, 0},
        {"lt", GC_LT, 0}, {"titlecaseletter", GC_LT, 0},
        {"lm", GC_LM, 0}, {"modifierletter", GC_LM, 0},
        {"lo", GC_LO, 0}, {"otherletter", GC_LO, 0},
        {"lc", GC_LC, 0}, {"l&", GC_LC, 0}, {"casedletter", GC_LC, 0},
        {"m", GC_M, 0}, {"mark", GC_M, 0}, {"combiningmark", GC_M, 0},
        {"mn", GC_MN, 0}, {"nonspacingmark", GC_MN, 0},
        {"mc", GC_MC, 0}, {"spacingmark", GC_MC, 0},
        {"me", GC_ME, 0}, {"enclosingmark", GC_ME, 0},
        {"n", GC_N, 0}, {"number", GC_N, 0},
        {"nd", GC_ND, 0}, {"decimalnumber", GC_ND, 0}, {"digit", GC_ND, 0},
        {"nl", GC_NL, 0}, {"letternumber", GC_NL, 0},
        {"no", GC_NO, 0}, {"othernumber", GC_NO, 0},
        {"p", GC_P, 0}, {"punct", GC_P, 0}, {"punctuation", GC_P, 0},
        {"pc", GC_PC, 0}, {"connectorpunctuation", GC_PC, 0},
        {"pd", GC_PD, 0}, {"dashpunctuation", GC_PD, 0},
        {"ps", GC_PS, 0}, {"openpunctuation", GC_PS, 0},
        {"pe", GC_PE, 0}, {"closepunctuation", GC_PE, 0},
        {"pi", GC_PI, 0}, {"initialpunctuation", GC_PI, 0},
        {"pf", GC_PF, 0}, {"finalpunctuation", GC_PF, 0},
        {"po", GC_PO, 0}, {"otherpunctuation", GC_PO, 0},
        {"s", GC_S, 0}, {"symbol", GC_S, 0},
        {"sm", GC_SM, 0}, {"mathsymbol", GC_SM, 0},
        {"sc", GC_SC, 0}, {"currencysymbol", GC_SC, 0},
        {"sk", GC_SK, 0}, {"modifiersymbol", GC_SK, 0},
        {"so", GC_SO, 0}, {"othersymbol", GC_SO, 0},
        {"z", GC_Z, 0}, {"separator", GC_Z, 0},
    }};
    for (const auto& e : TABLE) {
        if (key == e.name) {
            spec.gc_mask = e.gc;
            spec.flags = e.flags;
            return spec;
        }
    }
    if (key == "zs" || key == "spaceseparator") { spec.gc_mask = GC_ZS; return spec; }
    if (key == "zl" || key == "lineseparator") { spec.gc_mask = GC_ZL; return spec; }
    if (key == "zp" || key == "paragraphseparator") { spec.gc_mask = GC_ZP; return spec; }
    if (key == "c" || key == "other") { spec.gc_mask = GC_C; return spec; }
    if (key == "cc" || key == "control" || key == "cntrl") { spec.gc_mask = GC_CC; return spec; }
    if (key == "cf" || key == "format") { spec.gc_mask = GC_CF; return spec; }
    if (key == "cs" || key == "surrogate") { spec.gc_mask = GC_CS; return spec; }
    if (key == "co" || key == "privateuse") { spec.gc_mask = GC_CO; return spec; }
    if (key == "cn" || key == "unassigned") { spec.gc_mask = GC_CN; return spec; }

    if (key == "whitespace" || key == "wspace" || key == "space") { spec.flags = PROP_WHITESPACE; return spec; }
    if (key == "alphabetic" || key == "alpha") { spec.flags = PROP_ALPHABETIC; return spec; }
    if (key == "word") { spec.flags = PROP_WORD; return spec; }
    if (key == "ascii") { spec.flags = PROP_ASCII; return spec; }
    if (key == "any") { spec.flags = PROP_ANY; return spec; }
    if (key == "assigned") { spec.flags = PROP_ASSIGNED; return spec; }

    struct ScriptAlias {
        std::string_view name;
        char tag[4];
    };
    static constexpr std::array<ScriptAlias, 36> SCRIPTS{{
        {"latin", {'L', 'a', 't', 'n'}},      {"latn", {'L', 'a', 't', 'n'}},
        {"greek", {'G', 'r', 'e', 'k'}},      {"grek", {'G', 'r', 'e', 'k'}},
        {"cyrillic", {'C', 'y', 'r', 'l'}},   {"cyrl", {'C', 'y', 'r', 'l'}},
        {"armenian", {'A', 'r', 'm', 'n'}},   {"hebrew", {'H', 'e', 'b', 'r'}},
        {"arabic", {'A', 'r', 'a', 'b'}},     {"syriac", {'S', 'y', 'r', 'c'}},
        {"thaana", {'T', 'h', 'a', 'a'}},     {"devanagari", {'D', 'e', 'v', 'a'}},
        {"bengali", {'B', 'e', 'n', 'g'}},    {"gurmukhi", {'G', 'u', 'r', 'u'}},
        {"gujarati", {'G', 'u', 'j', 'r'}},   {"oriya", {'O', 'r', 'y', 'a'}},
        {"tamil", {'T', 'a', 'm', 'l'}},      {"telugu", {'T', 'e', 'l', 'u'}},
        {"kannada", {'K', 'n', 'd', 'a'}},    {"malayalam", {'M', 'l', 'y', 'm'}},
        {"sinhala", {'S', 'i', 'n', 'h'}},    {"thai", {'T', 'h', 'a', 'i'}},
        {"lao", {'L', 'a', 'o', 'o'}},        {"tibetan", {'T', 'i', 'b', 't'}},
        {"myanmar", {'M', 'y', 'm', 'r'}},    {"georgian", {'G', 'e', 'o', 'r'}},
        {"hangul", {'H', 'a', 'n', 'g'}},     {"ethiopic", {'E', 't', 'h', 'i'}},
        {"cherokee", {'C', 'h', 'e', 'r'}},   {"khmer", {'K', 'h', 'm', 'r'}},
        {"mongolian", {'M', 'o', 'n', 'g'}},  {"hiragana", {'H', 'i', 'r', 'a'}},
        {"katakana", {'K', 'a', 'n', 'a'}},   {"bopomofo", {'B', 'o', 'p', 'o'}},
        {"han", {'H', 'a', 'n', 'i'}},        {"hani", {'H', 'a', 'n', 'i'}},
    }};
    for (const auto& s : SCRIPTS) {
        if (key == s.name) {
            spec.script = make_script_tag(s.tag[0], s.tag[1], s.tag[2], s.tag[3]);
            return spec;
        }
    }
    if (key == "common" || key == "zyyy") { spec.script = make_script_tag('Z', 'y', 'y', 'y'); return spec; }
    if (key == "inherited" || key == "zinh") { spec.script = make_script_tag('Z', 'i', 'n', 'h'); return spec; }
    if (key == "unknown" || key == "zzzz") { spec.script = make_script_tag('Z', 'z', 'z', 'z'); return spec; }
    if (key.size() == 4 && std::all_of(key.begin(), key.end(), [](unsigned char c) { return std::isalpha(c); })) {
        spec.script = make_script_tag(
            static_cast<char>(std::toupper(static_cast<unsigned char>(key[0]))),
            key[1], key[2], key[3]);
        return spec;
    }

    throw std::runtime_error("Unsupported Unicode property in regex: " + std::string(raw_name));
}

struct CharClass {
    uint64_t ascii[2]{0, 0};
    uint32_t pos_gc_mask = 0;
    std::vector<std::pair<char32_t, char32_t>> ranges;
    std::vector<PropertySpec> props;
    bool icase = false;
    bool negated = false;

    void add_prop(const PropertySpec& spec) {
        if (!spec.negated && spec.flags == PROP_NONE && spec.script == 0) {
            pos_gc_mask |= spec.gc_mask;
        } else {
            props.push_back(spec);
        }
    }

    bool matches_inner_one(char32_t cp) const noexcept {
        if (!ranges.empty()) {
            if (ranges.size() <= 4) {
                for (const auto& [lo, hi] : ranges) {
                    if (cp >= lo && cp <= hi) return true;
                }
            } else {
                const auto it = std::upper_bound(
                    ranges.begin(), ranges.end(), cp,
                    [](char32_t val, const std::pair<char32_t, char32_t>& r) { return val < r.first; });
                if (it != ranges.begin()) {
                    const auto& prev = *(it - 1);
                    if (cp >= prev.first && cp <= prev.second) return true;
                }
            }
        }
        if (pos_gc_mask != 0 || !props.empty()) {
            const una::codepoint::prop p{cp};
            const uint32_t cbit = gc_bit(una::codepoint::get_general_category(p));
            if ((pos_gc_mask & cbit) != 0) return true;
            for (const auto& sp : props) {
                if (sp.matches(cp, p, cbit)) return true;
            }
        }
        return false;
    }

    bool matches_raw(char32_t cp) const noexcept {
        bool hit = matches_inner_one(cp);
        if (!hit && icase) {
            const char32_t lo = una::codepoint::to_simple_lowercase(cp);
            if (lo != cp && matches_inner_one(lo)) hit = true;
            if (!hit) {
                const char32_t up = una::codepoint::to_simple_uppercase(cp);
                if (up != cp && matches_inner_one(up)) hit = true;
            }
        }
        return negated ? !hit : hit;
    }

    void finalize() {
        if (ranges.size() > 1) {
            std::sort(ranges.begin(), ranges.end());
            size_t out = 0;
            for (size_t i = 1; i < ranges.size(); ++i) {
                if (ranges[i].first <= ranges[out].second + 1) {
                    ranges[out].second = std::max(ranges[out].second, ranges[i].second);
                } else {
                    ranges[++out] = ranges[i];
                }
            }
            ranges.resize(out + 1);
        }
        ascii[0] = 0;
        ascii[1] = 0;
        for (char32_t c = 0; c < 128; ++c) {
            if (matches_raw(c)) {
                ascii[c >> 6] |= (uint64_t{1} << (c & 63));
            }
        }
    }

    bool matches_ascii(uint8_t b) const noexcept {
        return ((ascii[b >> 6] >> (b & 63)) & 1u) != 0;
    }

    bool matches(char32_t cp) const noexcept {
        if (cp < 0x80) return matches_ascii(static_cast<uint8_t>(cp));
        return matches_raw(cp);
    }
};

enum class AnchorKind : uint8_t {
    StartLine,
    EndLine,
    StartText,
    EndText,
    WordBoundary,
    NotWordBoundary,
};

enum class Op : uint8_t {
    Match,
    Jump,
    Split,
    Char,
    CharICase,
    Any,
    Class,
    Anchor,
    Lookaround,
};

struct Inst {
    Op op = Op::Match;
    bool flag = false;       // Any: dotall; Anchor: multiline; Lookaround: positive
    bool behind = false;     // Lookaround: true = lookbehind
    uint8_t anchor = 0;
    uint32_t a = 0;          // Jump: target; Split: first; Char: cp; Class: idx; Lookaround: sub_idx
    uint32_t b = 0;          // Split: second; CharICase: upper_cp; Lookbehind: fixed_cps
};

struct Flags {
    bool icase = false;
    bool multiline = false;
    bool dotall = false;
};

struct AstNode {
    enum class Kind : uint8_t {
        Empty,
        Char,
        Any,
        Class,
        Anchor,
        Concat,
        Alt,
        Repeat,
        Lookaround,
    };
    Kind kind = Kind::Empty;
    Flags flags{};
    char32_t cp = 0;
    uint32_t class_idx = 0;
    AnchorKind anchor = AnchorKind::StartText;
    uint32_t min_rep = 0;
    uint32_t max_rep = 0; // UINT32_MAX = unbounded
    bool greedy = true;
    bool positive = true;
    bool behind = false;
    std::vector<AstNode> children;
};

constexpr uint32_t INF_REP = std::numeric_limits<uint32_t>::max();
constexpr uint32_t MAX_BOUNDED_REP = 1024;

struct Program {
    std::vector<Inst> insts;
    std::vector<CharClass> classes;
    uint64_t start_ascii[2]{ ~uint64_t{0}, ~uint64_t{0} };
    bool has_ascii_filter = false;
    bool anchored_start = false;
};

class Compiler {
public:
    Compiler(std::string_view pat, std::vector<CharClass>& classes)
        : pat_(pat), classes_(classes) {}

    AstNode parse_root(Flags initial = {}) {
        flags_ = initial;
        AstNode node = parse_alt();
        if (pos_ != pat_.size()) {
            throw std::runtime_error("Unexpected ')' in regex");
        }
        return node;
    }

    static std::optional<uint32_t> fixed_cp_length(const AstNode& node) {
        switch (node.kind) {
            case AstNode::Kind::Empty:
            case AstNode::Kind::Anchor:
            case AstNode::Kind::Lookaround:
                return 0u;
            case AstNode::Kind::Char:
            case AstNode::Kind::Any:
            case AstNode::Kind::Class:
                return 1u;
            case AstNode::Kind::Concat: {
                uint32_t sum = 0;
                for (const auto& c : node.children) {
                    auto sub = fixed_cp_length(c);
                    if (!sub) return std::nullopt;
                    sum += *sub;
                }
                return sum;
            }
            case AstNode::Kind::Alt: {
                if (node.children.empty()) return 0u;
                auto first = fixed_cp_length(node.children[0]);
                if (!first) return std::nullopt;
                for (size_t i = 1; i < node.children.size(); ++i) {
                    if (fixed_cp_length(node.children[i]) != first) return std::nullopt;
                }
                return first;
            }
            case AstNode::Kind::Repeat: {
                if (node.min_rep != node.max_rep) return std::nullopt;
                auto inner = fixed_cp_length(node.children[0]);
                if (!inner) return std::nullopt;
                return (*inner) * node.min_rep;
            }
        }
        return std::nullopt;
    }

    static void emit(const AstNode& node, Program& prog) {
        switch (node.kind) {
            case AstNode::Kind::Empty:
                return;
            case AstNode::Kind::Char: {
                Inst inst;
                if (node.flags.icase) {
                    const char32_t lo = una::codepoint::to_simple_lowercase(node.cp);
                    const char32_t up = una::codepoint::to_simple_uppercase(node.cp);
                    if (lo != up) {
                        inst.op = Op::CharICase;
                        inst.a = lo;
                        inst.b = up;
                        prog.insts.push_back(inst);
                        return;
                    }
                }
                inst.op = Op::Char;
                inst.a = node.cp;
                prog.insts.push_back(inst);
                return;
            }
            case AstNode::Kind::Any: {
                Inst inst;
                inst.op = Op::Any;
                inst.flag = node.flags.dotall;
                prog.insts.push_back(inst);
                return;
            }
            case AstNode::Kind::Class: {
                Inst inst;
                inst.op = Op::Class;
                inst.a = node.class_idx;
                prog.insts.push_back(inst);
                return;
            }
            case AstNode::Kind::Anchor: {
                Inst inst;
                inst.op = Op::Anchor;
                inst.anchor = static_cast<uint8_t>(node.anchor);
                inst.flag = node.flags.multiline;
                prog.insts.push_back(inst);
                return;
            }
            case AstNode::Kind::Lookaround: {
                uint32_t fixed_len = 0;
                if (node.behind) {
                    auto fl = fixed_cp_length(node.children[0]);
                    if (!fl) {
                        throw std::runtime_error("Variable-length lookbehind is not supported");
                    }
                    fixed_len = *fl;
                }
                const auto look_pc = static_cast<uint32_t>(prog.insts.size());
                prog.insts.push_back(Inst{
                    .op = Op::Lookaround,
                    .flag = node.positive,
                    .behind = node.behind,
                    .a = 0,
                    .b = fixed_len,
                });
                const auto jump_over_pc = static_cast<uint32_t>(prog.insts.size());
                prog.insts.push_back(Inst{.op = Op::Jump, .a = 0});
                const auto sub_entry_pc = static_cast<uint32_t>(prog.insts.size());
                emit(node.children[0], prog);
                prog.insts.push_back(Inst{.op = Op::Match});
                const auto after_pc = static_cast<uint32_t>(prog.insts.size());
                prog.insts[look_pc].a = sub_entry_pc;
                prog.insts[jump_over_pc].a = after_pc;
                return;
            }
            case AstNode::Kind::Concat: {
                for (const auto& child : node.children) emit(child, prog);
                return;
            }
            case AstNode::Kind::Alt: {
                if (node.children.empty()) return;
                if (node.children.size() == 1) {
                    emit(node.children[0], prog);
                    return;
                }
                std::vector<uint32_t> end_jumps;
                end_jumps.reserve(node.children.size() - 1);
                for (size_t i = 0; i + 1 < node.children.size(); ++i) {
                    const auto split_pc = static_cast<uint32_t>(prog.insts.size());
                    prog.insts.push_back(Inst{.op = Op::Split, .a = split_pc + 1, .b = 0});
                    emit(node.children[i], prog);
                    const auto jmp_pc = static_cast<uint32_t>(prog.insts.size());
                    prog.insts.push_back(Inst{.op = Op::Jump, .a = 0});
                    end_jumps.push_back(jmp_pc);
                    prog.insts[split_pc].b = static_cast<uint32_t>(prog.insts.size());
                }
                emit(node.children.back(), prog);
                const auto after = static_cast<uint32_t>(prog.insts.size());
                for (uint32_t jpc : end_jumps) prog.insts[jpc].a = after;
                return;
            }
            case AstNode::Kind::Repeat: {
                const auto& body = node.children[0];
                if (node.max_rep == 0) return;
                for (uint32_t i = 0; i < node.min_rep; ++i) {
                    emit(body, prog);
                }
                if (node.max_rep == INF_REP) {
                    if (node.min_rep == 0) {
                        const auto split_pc = static_cast<uint32_t>(prog.insts.size());
                        prog.insts.push_back(Inst{.op = Op::Split});
                        const auto body_pc = static_cast<uint32_t>(prog.insts.size());
                        emit(body, prog);
                        prog.insts.push_back(Inst{.op = Op::Jump, .a = split_pc});
                        const auto after_pc = static_cast<uint32_t>(prog.insts.size());
                        prog.insts[split_pc].a = node.greedy ? body_pc : after_pc;
                        prog.insts[split_pc].b = node.greedy ? after_pc : body_pc;
                    } else {
                        // Already emitted min_rep copies; add trailing * loop via single Split back to last body or new loop
                        const auto split_pc = static_cast<uint32_t>(prog.insts.size());
                        prog.insts.push_back(Inst{.op = Op::Split});
                        const auto body_pc = static_cast<uint32_t>(prog.insts.size());
                        emit(body, prog);
                        prog.insts.push_back(Inst{.op = Op::Jump, .a = split_pc});
                        const auto after_pc = static_cast<uint32_t>(prog.insts.size());
                        prog.insts[split_pc].a = node.greedy ? body_pc : after_pc;
                        prog.insts[split_pc].b = node.greedy ? after_pc : body_pc;
                    }
                } else {
                    const uint32_t opt_count = node.max_rep - node.min_rep;
                    std::vector<uint32_t> splits;
                    splits.reserve(opt_count);
                    for (uint32_t i = 0; i < opt_count; ++i) {
                        const auto split_pc = static_cast<uint32_t>(prog.insts.size());
                        prog.insts.push_back(Inst{.op = Op::Split});
                        const auto body_pc = static_cast<uint32_t>(prog.insts.size());
                        emit(body, prog);
                        prog.insts[split_pc].a = body_pc; // temporary
                        splits.push_back(split_pc);
                    }
                    const auto after_pc = static_cast<uint32_t>(prog.insts.size());
                    for (uint32_t spc : splits) {
                        const uint32_t body_pc = prog.insts[spc].a;
                        prog.insts[spc].a = node.greedy ? body_pc : after_pc;
                        prog.insts[spc].b = node.greedy ? after_pc : body_pc;
                    }
                }
                return;
            }
        }
    }

private:
    std::string_view pat_;
    size_t pos_ = 0;
    Flags flags_{};
    std::vector<CharClass>& classes_;

    AstNode parse_alt() {
        std::vector<AstNode> branches;
        branches.push_back(parse_concat());
        while (pos_ < pat_.size() && pat_[pos_] == '|') {
            ++pos_;
            branches.push_back(parse_concat());
        }
        if (branches.size() == 1) return std::move(branches[0]);
        AstNode alt;
        alt.kind = AstNode::Kind::Alt;
        alt.children = std::move(branches);
        return alt;
    }

    AstNode parse_concat() {
        std::vector<AstNode> items;
        while (pos_ < pat_.size() && pat_[pos_] != '|' && pat_[pos_] != ')') {
            AstNode item = parse_quantified();
            if (item.kind != AstNode::Kind::Empty) {
                items.push_back(std::move(item));
            }
        }
        if (items.empty()) return AstNode{};
        if (items.size() == 1) return std::move(items[0]);
        AstNode cat;
        cat.kind = AstNode::Kind::Concat;
        cat.children = std::move(items);
        return cat;
    }

    AstNode parse_quantified() {
        AstNode atom = parse_atom();
        if (pos_ >= pat_.size()) return atom;

        uint32_t min_r = 0;
        uint32_t max_r = 0;
        bool has_quant = false;
        const char ch = pat_[pos_];
        if (ch == '*') {
            min_r = 0; max_r = INF_REP; has_quant = true; ++pos_;
        } else if (ch == '+') {
            min_r = 1; max_r = INF_REP; has_quant = true; ++pos_;
        } else if (ch == '?') {
            min_r = 0; max_r = 1; has_quant = true; ++pos_;
        } else if (ch == '{') {
            ++pos_;
            min_r = parse_uint_bound();
            if (pos_ < pat_.size() && pat_[pos_] == '}') {
                max_r = min_r;
                ++pos_;
            } else if (pos_ < pat_.size() && pat_[pos_] == ',') {
                ++pos_;
                if (pos_ < pat_.size() && pat_[pos_] == '}') {
                    max_r = INF_REP;
                    ++pos_;
                } else {
                    max_r = parse_uint_bound();
                    if (pos_ >= pat_.size() || pat_[pos_] != '}') {
                        throw std::runtime_error("Unclosed '{m,n}' quantifier in regex");
                    }
                    ++pos_;
                }
            } else {
                throw std::runtime_error("Invalid '{...}' quantifier in regex");
            }
            if (max_r < min_r || min_r > MAX_BOUNDED_REP || (max_r != INF_REP && max_r > MAX_BOUNDED_REP)) {
                throw std::runtime_error("Quantifier bounds out of range in regex");
            }
            has_quant = true;
        }

        if (!has_quant) return atom;
        if (atom.kind == AstNode::Kind::Anchor || atom.kind == AstNode::Kind::Lookaround ||
            atom.kind == AstNode::Kind::Empty) {
            if (atom.kind == AstNode::Kind::Empty) return atom;
            throw std::runtime_error("Cannot quantify anchor or assertion in regex");
        }

        bool greedy = true;
        if (pos_ < pat_.size() && pat_[pos_] == '?') {
            greedy = false;
            ++pos_;
        }

        AstNode rep;
        rep.kind = AstNode::Kind::Repeat;
        rep.min_rep = min_r;
        rep.max_rep = max_r;
        rep.greedy = greedy;
        rep.children.push_back(std::move(atom));
        return rep;
    }

    uint32_t parse_uint_bound() {
        if (pos_ >= pat_.size() || !std::isdigit(static_cast<unsigned char>(pat_[pos_]))) {
            throw std::runtime_error("Expected integer in regex quantifier");
        }
        uint64_t val = 0;
        while (pos_ < pat_.size() && std::isdigit(static_cast<unsigned char>(pat_[pos_]))) {
            val = val * 10 + static_cast<uint64_t>(pat_[pos_] - '0');
            if (val > MAX_BOUNDED_REP) {
                throw std::runtime_error("Regex quantifier bound exceeds limit");
            }
            ++pos_;
        }
        return static_cast<uint32_t>(val);
    }

    AstNode parse_atom() {
        const char ch = pat_[pos_];
        if (ch == '^') {
            ++pos_;
            return AstNode{.kind = AstNode::Kind::Anchor, .flags = flags_,
                           .anchor = flags_.multiline ? AnchorKind::StartLine : AnchorKind::StartText};
        }
        if (ch == '$') {
            ++pos_;
            return AstNode{.kind = AstNode::Kind::Anchor, .flags = flags_,
                           .anchor = flags_.multiline ? AnchorKind::EndLine : AnchorKind::EndText};
        }
        if (ch == '.') {
            ++pos_;
            return AstNode{.kind = AstNode::Kind::Any, .flags = flags_};
        }
        if (ch == '[') {
            return parse_bracket_class();
        }
        if (ch == '(') {
            return parse_group();
        }
        if (ch == '*' || ch == '+' || ch == '?' || ch == '{') {
            throw std::runtime_error("Nothing to repeat in regex");
        }
        if (ch == '\\') {
            return parse_escape_atom();
        }
        auto [cp, len] = decode_utf8(pat_.data() + pos_, pat_.size() - pos_);
        pos_ += len;
        return AstNode{.kind = AstNode::Kind::Char, .flags = flags_, .cp = cp};
    }

    AstNode parse_group() {
        ++pos_; // skip '('
        if (pos_ < pat_.size() && pat_[pos_] == '?') {
            ++pos_;
            if (pos_ >= pat_.size()) throw std::runtime_error("Incomplete '(?' group in regex");
            if (pat_[pos_] == ':') {
                ++pos_;
                const Flags saved = flags_;
                AstNode inner = parse_alt();
                flags_ = saved;
                expect_char(')');
                return inner;
            }
            if (pat_[pos_] == '=' || pat_[pos_] == '!') {
                const bool positive = (pat_[pos_] == '=');
                ++pos_;
                const Flags saved = flags_;
                AstNode inner = parse_alt();
                flags_ = saved;
                expect_char(')');
                AstNode la;
                la.kind = AstNode::Kind::Lookaround;
                la.positive = positive;
                la.behind = false;
                la.children.push_back(std::move(inner));
                return la;
            }
            if (pat_[pos_] == '<' && pos_ + 1 < pat_.size() &&
                (pat_[pos_ + 1] == '=' || pat_[pos_ + 1] == '!')) {
                const bool positive = (pat_[pos_ + 1] == '=');
                pos_ += 2;
                const Flags saved = flags_;
                AstNode inner = parse_alt();
                flags_ = saved;
                expect_char(')');
                AstNode lb;
                lb.kind = AstNode::Kind::Lookaround;
                lb.positive = positive;
                lb.behind = true;
                lb.children.push_back(std::move(inner));
                return lb;
            }
            // Inline flags: (?ims-ims) or (?ims-ims:...)
            Flags modified = flags_;
            bool negate_flag = false;
            bool saw_flag = false;
            while (pos_ < pat_.size() && pat_[pos_] != ':' && pat_[pos_] != ')') {
                const char f = pat_[pos_++];
                if (f == '-') {
                    negate_flag = true;
                } else if (f == 'i') {
                    modified.icase = !negate_flag; saw_flag = true;
                } else if (f == 'm') {
                    modified.multiline = !negate_flag; saw_flag = true;
                } else if (f == 's') {
                    modified.dotall = !negate_flag; saw_flag = true;
                } else if (f == 'u') {
                    saw_flag = true;
                } else {
                    throw std::runtime_error(std::string("Unsupported inline flag in regex: ") + f);
                }
            }
            if (!saw_flag || pos_ >= pat_.size()) {
                throw std::runtime_error("Invalid group syntax in regex");
            }
            if (pat_[pos_] == ')') {
                ++pos_;
                flags_ = modified;
                return AstNode{};
            }
            ++pos_; // ':'
            const Flags saved = flags_;
            flags_ = modified;
            AstNode inner = parse_alt();
            flags_ = saved;
            expect_char(')');
            return inner;
        }

        const Flags saved = flags_;
        AstNode inner = parse_alt();
        flags_ = saved;
        expect_char(')');
        return inner;
    }

    void expect_char(char expected) {
        if (pos_ >= pat_.size() || pat_[pos_] != expected) {
            throw std::runtime_error(std::string("Expected '") + expected + "' in regex");
        }
        ++pos_;
    }

    uint32_t make_single_prop_class(const PropertySpec& spec) {
        CharClass cc;
        cc.icase = false;
        cc.negated = spec.negated;
        PropertySpec pos_spec = spec;
        pos_spec.negated = false;
        cc.add_prop(pos_spec);
        cc.finalize();
        const auto idx = static_cast<uint32_t>(classes_.size());
        classes_.push_back(std::move(cc));
        return idx;
    }

    PropertySpec parse_escape_property(char esc) {
        const bool negated = (esc == 'P');
        if (pos_ >= pat_.size()) {
            throw std::runtime_error("Incomplete \\p escape in regex");
        }
        if (pat_[pos_] == '{') {
            ++pos_;
            const size_t close = pat_.find('}', pos_);
            if (close == std::string_view::npos || close == pos_) {
                throw std::runtime_error("Invalid \\p{...} in regex");
            }
            const std::string_view name = pat_.substr(pos_, close - pos_);
            pos_ = close + 1;
            return parse_unicode_property(name, negated);
        }
        const std::string_view single = pat_.substr(pos_, 1);
        ++pos_;
        return parse_unicode_property(single, negated);
    }

    char32_t parse_hex_escape(size_t exact_digits, bool braced_allowed) {
        if (braced_allowed && pos_ < pat_.size() && pat_[pos_] == '{') {
            ++pos_;
            const size_t close = pat_.find('}', pos_);
            if (close == std::string_view::npos || close == pos_ || close - pos_ > 8) {
                throw std::runtime_error("Invalid braced hex escape in regex");
            }
            uint32_t val = 0;
            for (size_t i = pos_; i < close; ++i) {
                const unsigned char c = static_cast<unsigned char>(pat_[i]);
                if (!std::isxdigit(c)) throw std::runtime_error("Non-hex digit in regex escape");
                val = (val << 4) | static_cast<uint32_t>(
                    std::isdigit(c) ? (c - '0') : (std::tolower(c) - 'a' + 10));
            }
            pos_ = close + 1;
            if (val > 0x10FFFFu || (val >= 0xD800u && val <= 0xDFFFu)) {
                throw std::runtime_error("Hex escape out of Unicode range in regex");
            }
            return static_cast<char32_t>(val);
        }
        if (pos_ + exact_digits > pat_.size()) {
            throw std::runtime_error("Truncated hex escape in regex");
        }
        uint32_t val = 0;
        for (size_t i = 0; i < exact_digits; ++i) {
            const unsigned char c = static_cast<unsigned char>(pat_[pos_++]);
            if (!std::isxdigit(c)) throw std::runtime_error("Invalid hex escape in regex");
            val = (val << 4) | static_cast<uint32_t>(
                std::isdigit(c) ? (c - '0') : (std::tolower(c) - 'a' + 10));
        }
        if (val > 0x10FFFFu || (val >= 0xD800u && val <= 0xDFFFu)) {
            throw std::runtime_error("Hex escape out of Unicode range in regex");
        }
        return static_cast<char32_t>(val);
    }

    // Parses either a literal char32_t or a PropertySpec inside escapes.
    bool parse_class_or_char_escape(char32_t& out_cp, PropertySpec& out_prop) {
        if (pos_ >= pat_.size()) throw std::runtime_error("Trailing '\\' in regex");
        const char e = pat_[pos_++];
        switch (e) {
            case 's': out_prop = PropertySpec{.flags = PROP_WHITESPACE, .negated = false}; return true;
            case 'S': out_prop = PropertySpec{.flags = PROP_WHITESPACE, .negated = true}; return true;
            case 'd': out_prop = PropertySpec{.gc_mask = GC_ND, .negated = false}; return true;
            case 'D': out_prop = PropertySpec{.gc_mask = GC_ND, .negated = true}; return true;
            case 'w': out_prop = PropertySpec{.flags = PROP_WORD, .negated = false}; return true;
            case 'W': out_prop = PropertySpec{.flags = PROP_WORD, .negated = true}; return true;
            case 'p':
            case 'P': out_prop = parse_escape_property(e); return true;
            case 'n': out_cp = U'\n'; return false;
            case 'r': out_cp = U'\r'; return false;
            case 't': out_cp = U'\t'; return false;
            case 'f': out_cp = U'\f'; return false;
            case 'v': out_cp = U'\v'; return false;
            case 'a': out_cp = U'\a'; return false;
            case '0': out_cp = U'\0'; return false;
            case 'x': out_cp = parse_hex_escape(2, true); return false;
            case 'u': out_cp = parse_hex_escape(4, true); return false;
            case 'U': out_cp = parse_hex_escape(8, false); return false;
            default:
                if (std::isalnum(static_cast<unsigned char>(e))) {
                    throw std::runtime_error(std::string("Unsupported escape '\\") + e + "' in regex");
                }
                out_cp = static_cast<unsigned char>(e);
                return false;
        }
    }

    AstNode parse_escape_atom() {
        ++pos_; // skip '\'
        if (pos_ >= pat_.size()) throw std::runtime_error("Trailing '\\' in regex");
        const char peek = pat_[pos_];
        if (peek == 'A') {
            ++pos_;
            return AstNode{.kind = AstNode::Kind::Anchor, .flags = flags_, .anchor = AnchorKind::StartText};
        }
        if (peek == 'z') {
            ++pos_;
            return AstNode{.kind = AstNode::Kind::Anchor, .flags = flags_, .anchor = AnchorKind::EndText};
        }
        if (peek == 'b') {
            ++pos_;
            return AstNode{.kind = AstNode::Kind::Anchor, .flags = flags_, .anchor = AnchorKind::WordBoundary};
        }
        if (peek == 'B') {
            ++pos_;
            return AstNode{.kind = AstNode::Kind::Anchor, .flags = flags_, .anchor = AnchorKind::NotWordBoundary};
        }

        char32_t cp = 0;
        PropertySpec prop;
        if (parse_class_or_char_escape(cp, prop)) {
            return AstNode{.kind = AstNode::Kind::Class, .flags = flags_,
                           .class_idx = make_single_prop_class(prop)};
        }
        return AstNode{.kind = AstNode::Kind::Char, .flags = flags_, .cp = cp};
    }

    AstNode parse_bracket_class() {
        ++pos_; // skip '['
        CharClass cc;
        cc.icase = flags_.icase;
        if (pos_ < pat_.size() && pat_[pos_] == '^') {
            cc.negated = true;
            ++pos_;
        }

        bool first = true;
        bool has_prev_cp = false;
        char32_t prev_cp = 0;

        while (pos_ < pat_.size()) {
            if (pat_[pos_] == ']' && !first) {
                ++pos_;
                if (has_prev_cp) cc.ranges.emplace_back(prev_cp, prev_cp);
                cc.finalize();
                const auto idx = static_cast<uint32_t>(classes_.size());
                classes_.push_back(std::move(cc));
                return AstNode{.kind = AstNode::Kind::Class, .flags = flags_, .class_idx = idx};
            }
            first = false;

            if (pat_[pos_] == '-' && has_prev_cp && pos_ + 1 < pat_.size() && pat_[pos_ + 1] != ']') {
                ++pos_; // consume '-'
                char32_t end_cp = 0;
                if (pat_[pos_] == '\\') {
                    ++pos_;
                    PropertySpec prop;
                    if (parse_class_or_char_escape(end_cp, prop)) {
                        throw std::runtime_error("Invalid property escape as range endpoint in regex");
                    }
                } else {
                    auto [c, len] = decode_utf8(pat_.data() + pos_, pat_.size() - pos_);
                    end_cp = c;
                    pos_ += len;
                }
                if (end_cp < prev_cp) {
                    throw std::runtime_error("Out-of-order character class range in regex");
                }
                cc.ranges.emplace_back(prev_cp, end_cp);
                has_prev_cp = false;
                continue;
            }

            if (has_prev_cp) {
                cc.ranges.emplace_back(prev_cp, prev_cp);
                has_prev_cp = false;
            }

            if (pat_[pos_] == '\\') {
                ++pos_;
                char32_t cp = 0;
                PropertySpec prop;
                if (parse_class_or_char_escape(cp, prop)) {
                    cc.add_prop(prop);
                } else {
                    prev_cp = cp;
                    has_prev_cp = true;
                }
            } else {
                auto [cp, len] = decode_utf8(pat_.data() + pos_, pat_.size() - pos_);
                pos_ += len;
                prev_cp = cp;
                has_prev_cp = true;
            }
        }
        throw std::runtime_error("Unclosed '[' character class in regex");
    }
};

// Compute conservative ASCII start-byte filter: if false for byte b < 0x80,
// no non-empty or empty match can begin at an ASCII byte `b`.
bool compute_start_filter(const Program& prog, uint32_t pc, uint64_t out_ascii[2], uint32_t depth = 0) {
    if (depth > 32 || pc >= prog.insts.size()) return false; // false = cannot filter (allow all)
    const Inst& inst = prog.insts[pc];
    switch (inst.op) {
        case Op::Match:
        case Op::Anchor:
        case Op::Lookaround:
            return false;
        case Op::Jump:
            return compute_start_filter(prog, inst.a, out_ascii, depth + 1);
        case Op::Split: {
            uint64_t a[2]{0, 0}, b[2]{0, 0};
            if (!compute_start_filter(prog, inst.a, a, depth + 1)) return false;
            if (!compute_start_filter(prog, inst.b, b, depth + 1)) return false;
            out_ascii[0] |= a[0] | b[0];
            out_ascii[1] |= a[1] | b[1];
            return true;
        }
        case Op::Char:
            if (inst.a < 0x80) {
                out_ascii[inst.a >> 6] |= (uint64_t{1} << (inst.a & 63));
            }
            return true;
        case Op::CharICase:
            if (inst.a < 0x80) out_ascii[inst.a >> 6] |= (uint64_t{1} << (inst.a & 63));
            if (inst.b < 0x80) out_ascii[inst.b >> 6] |= (uint64_t{1} << (inst.b & 63));
            return true;
        case Op::Any:
            out_ascii[0] = ~uint64_t{0};
            out_ascii[1] = ~uint64_t{0};
            if (!inst.flag) out_ascii[0] &= ~(uint64_t{1} << '\n');
            return true;
        case Op::Class: {
            const auto& cc = prog.classes[inst.a];
            out_ascii[0] |= cc.ascii[0];
            out_ascii[1] |= cc.ascii[1];
            return true;
        }
    }
    return false;
}

Program compile_regex(std::string_view pattern) {
    Program prog;
    Compiler compiler(pattern, prog.classes);
    AstNode root = compiler.parse_root();
    if (root.kind == AstNode::Kind::Concat && !root.children.empty() &&
        root.children[0].kind == AstNode::Kind::Anchor &&
        root.children[0].anchor == AnchorKind::StartText) {
        prog.anchored_start = true;
    } else if (root.kind == AstNode::Kind::Anchor && root.anchor == AnchorKind::StartText) {
        prog.anchored_start = true;
    }
    Compiler::emit(root, prog);
    prog.insts.push_back(Inst{.op = Op::Match});

    uint64_t mask[2]{0, 0};
    if (compute_start_filter(prog, 0, mask)) {
        prog.start_ascii[0] = mask[0];
        prog.start_ascii[1] = mask[1];
        prog.has_ascii_filter = (mask[0] != ~uint64_t{0} || mask[1] != ~uint64_t{0});
    }
    return prog;
}

struct Frame {
    uint32_t pc;
    size_t pos;
};

class VisitedTable {
public:
    void next_run() {
        ++gen_;
        if (gen_ == 0) {
            slots_.fill({});
            gen_ = 1;
        }
        if (!overflow_.empty()) overflow_.clear();
    }

    // Returns true if `key` was newly inserted, false if already visited in this run.
    bool insert(uint64_t key) {
        constexpr size_t MASK = TABLE_SIZE - 1;
        size_t idx = static_cast<size_t>((key * 11400714819323198485ull) >> 53) & MASK;
        for (size_t probe = 0; probe < 4; ++probe) {
            Slot& s = slots_[(idx + probe) & MASK];
            if (s.gen != gen_) {
                s.key = key;
                s.gen = gen_;
                return true;
            }
            if (s.key == key) return false;
        }
        return overflow_.insert(key).second;
    }

private:
    static constexpr size_t TABLE_SIZE = 2048;
    struct Slot {
        uint64_t key = 0;
        uint32_t gen = 0;
    };
    std::array<Slot, TABLE_SIZE> slots_{};
    std::unordered_set<uint64_t> overflow_;
    uint32_t gen_ = 0;
};

struct VmScratch {
    std::vector<Frame> stack;
    VisitedTable visited;
    std::unique_ptr<VmScratch> sub_scratch;

    VmScratch& child() {
        if (!sub_scratch) sub_scratch = std::make_unique<VmScratch>();
        return *sub_scratch;
    }
};

constexpr size_t NO_MATCH = std::numeric_limits<size_t>::max();

bool eval_anchor(AnchorKind kind, bool multiline, std::string_view input, size_t pos) noexcept {
    switch (kind) {
        case AnchorKind::StartText:
            return pos == 0;
        case AnchorKind::EndText:
            return pos == input.size();
        case AnchorKind::StartLine:
            return pos == 0 || (multiline && input[pos - 1] == '\n');
        case AnchorKind::EndLine:
            return pos == input.size() || (multiline && input[pos] == '\n');
        case AnchorKind::WordBoundary:
        case AnchorKind::NotWordBoundary: {
            const bool left_word = pos > 0 && is_word_cp(decode_utf8_prev(input.data(), pos).first);
            const bool right_word =
                pos < input.size() &&
                is_word_cp(decode_utf8(input.data() + pos, input.size() - pos).first);
            const bool boundary = (left_word != right_word);
            return kind == AnchorKind::WordBoundary ? boundary : !boundary;
        }
    }
    return false;
}

// Execute `prog` starting at `entry_pc` anchored at `origin`.
// If `exact_end != NO_MATCH`, only accept Match at `pos == exact_end`.
size_t run_vm(const Program& prog, std::string_view input, size_t origin, VmScratch& scratch,
              size_t exact_end = NO_MATCH, uint32_t entry_pc = 0) {
    scratch.stack.clear();
    scratch.visited.next_run();

    const Inst* insts = prog.insts.data();
    const char* data = input.data();
    const size_t len = input.size();

    uint32_t pc = entry_pc;
    size_t pos = origin;

    for (;;) {
        const Inst& ins = insts[pc];
        switch (ins.op) {
            case Op::Match:
                if (exact_end == NO_MATCH || pos == exact_end) return pos;
                break;

            case Op::Jump:
                pc = ins.a;
                continue;

            case Op::Split: {
                const uint64_t key = (static_cast<uint64_t>(pos - origin) << 16) | pc;
                if (!scratch.visited.insert(key)) break;
                scratch.stack.push_back(Frame{ins.b, pos});
                pc = ins.a;
                continue;
            }

            case Op::Char:
                if (pos < len) {
                    const auto b0 = static_cast<uint8_t>(data[pos]);
                    if (ins.a < 0x80) {
                        if (b0 == ins.a) {
                            ++pos;
                            ++pc;
                            continue;
                        }
                    } else if (b0 >= 0x80) {
                        auto [cp, clen] = decode_utf8(data + pos, len - pos);
                        if (cp == ins.a) {
                            pos += clen;
                            ++pc;
                            continue;
                        }
                    }
                }
                break;

            case Op::CharICase:
                if (pos < len) {
                    const auto b0 = static_cast<uint8_t>(data[pos]);
                    if (b0 < 0x80 && ins.a < 0x80 && ins.b < 0x80) {
                        if (b0 == ins.a || b0 == ins.b) {
                            ++pos;
                            ++pc;
                            continue;
                        }
                    } else {
                        auto [cp, clen] = decode_utf8(data + pos, len - pos);
                        const char32_t lo = una::codepoint::to_simple_lowercase(cp);
                        if (cp == ins.a || cp == ins.b || lo == ins.a) {
                            pos += clen;
                            ++pc;
                            continue;
                        }
                    }
                }
                break;

            case Op::Any:
                if (pos < len) {
                    const auto b0 = static_cast<uint8_t>(data[pos]);
                    if (b0 < 0x80) {
                        if (ins.flag || b0 != '\n') {
                            ++pos;
                            ++pc;
                            continue;
                        }
                    } else {
                        pos += decode_utf8(data + pos, len - pos).second;
                        ++pc;
                        continue;
                    }
                }
                break;

            case Op::Class:
                if (pos < len) {
                    const auto& cc = prog.classes[ins.a];
                    const auto b0 = static_cast<uint8_t>(data[pos]);
                    if (b0 < 0x80) {
                        if (cc.matches_ascii(b0)) {
                            ++pos;
                            ++pc;
                            continue;
                        }
                    } else {
                        auto [cp, clen] = decode_utf8(data + pos, len - pos);
                        if (cc.matches_raw(cp)) {
                            pos += clen;
                            ++pc;
                            continue;
                        }
                    }
                }
                break;

            case Op::Anchor:
                if (eval_anchor(static_cast<AnchorKind>(ins.anchor), ins.flag, input, pos)) {
                    ++pc;
                    continue;
                }
                break;

            case Op::Lookaround: {
                bool matched = false;
                if (!ins.behind) {
                    matched = run_vm(prog, input, pos, scratch.child(), NO_MATCH, ins.a) != NO_MATCH;
                } else {
                    size_t back_pos = pos;
                    bool ok = true;
                    for (uint32_t k = 0; k < ins.b; ++k) {
                        if (back_pos == 0) { ok = false; break; }
                        back_pos -= decode_utf8_prev(data, back_pos).second;
                    }
                    if (ok) {
                        matched = run_vm(prog, input, back_pos, scratch.child(), pos, ins.a) != NO_MATCH;
                    }
                }
                if (matched == ins.flag) {
                    ++pc;
                    continue;
                }
                break;
            }
        }

        if (scratch.stack.empty()) return NO_MATCH;
        const Frame next = scratch.stack.back();
        scratch.stack.pop_back();
        pc = next.pc;
        pos = next.pos;
    }
}

} // namespace

// ===== InvertPattern =====

Result<std::vector<PatternMatch>>
InvertPattern::find_matches(std::string_view inside) const {
    auto result = inner_->find_matches(inside);
    if (!result) return std::unexpected(result.error());
    for (auto& [offsets, flag] : *result) {
        flag = !flag;
    }
    return result;
}

// ===== CharPattern =====

Result<std::vector<PatternMatch>>
CharPattern::find_matches(std::string_view inside) const {
    if (inside.empty()) {
        return std::vector<PatternMatch>{{{0, 0}, false}};
    }

    std::vector<PatternMatch> matches;
    size_t last_offset = 0;
    size_t pos = 0;
    while (pos < inside.size()) {
        auto [cp, len] = decode_utf8(inside.data() + pos, inside.size() - pos);
        if (cp == ch_) {
            if (last_offset < pos) {
                matches.push_back({{last_offset, pos}, false});
            }
            matches.push_back({{pos, pos + len}, true});
            last_offset = pos + len;
        }
        pos += len;
    }
    if (last_offset < inside.size()) {
        matches.push_back({{last_offset, inside.size()}, false});
    }
    return matches;
}

// ===== StringPattern =====

Result<std::vector<PatternMatch>>
StringPattern::find_matches(std::string_view inside) const {
    if (pattern_.empty()) {
        return std::vector<PatternMatch>{{{0, inside.size()}, false}};
    }

    if (inside.empty()) {
        return std::vector<PatternMatch>{{{0, 0}, false}};
    }

    std::vector<PatternMatch> matches;
    size_t prev = 0;
    size_t pos = 0;
    while ((pos = inside.find(pattern_, prev)) != std::string_view::npos) {
        if (prev != pos) {
            matches.push_back({{prev, pos}, false});
        }
        matches.push_back({{pos, pos + pattern_.size()}, true});
        prev = pos + pattern_.size();
    }
    if (prev != inside.size()) {
        matches.push_back({{prev, inside.size()}, false});
    }
    return matches;
}

// ===== FuncPattern =====

Result<std::vector<PatternMatch>>
FuncPattern::find_matches(std::string_view inside) const {
    if (inside.empty()) {
        return std::vector<PatternMatch>{{{0, 0}, false}};
    }

    std::vector<PatternMatch> matches;
    size_t last_offset = 0;
    size_t last_seen = 0;
    size_t pos = 0;

    while (pos < inside.size()) {
        auto [cp, len] = decode_utf8(inside.data() + pos, inside.size() - pos);
        last_seen = pos + len;
        if (pred_(cp)) {
            if (last_offset < pos) {
                matches.push_back({{last_offset, pos}, false});
            }
            matches.push_back({{pos, pos + len}, true});
            last_offset = pos + len;
        }
        pos += len;
    }
    if (last_seen > last_offset) {
        matches.push_back({{last_offset, last_seen}, false});
    }
    return matches;
}

// ===== RegexPattern =====

struct RegexPattern::Impl {
    Program prog;
};

RegexPattern::RegexPattern(const std::string& pattern)
    : impl_(std::make_unique<Impl>()), pattern_str_(pattern) {
    impl_->prog = compile_regex(pattern);
}

RegexPattern::~RegexPattern() = default;
RegexPattern::RegexPattern(RegexPattern&&) noexcept = default;
RegexPattern& RegexPattern::operator=(RegexPattern&&) noexcept = default;

Result<std::vector<PatternMatch>>
RegexPattern::find_matches(std::string_view inside) const {
    std::vector<PatternMatch> splits;
    splits.reserve(inside.size() / 4 + 2);

    const Program& prog = impl_->prog;
    VmScratch scratch;
    scratch.stack.reserve(32);

    size_t prev = 0;
    size_t search_pos = 0;
    const size_t n = inside.size();
    const char* data = inside.data();

    while (search_pos <= n) {
        if (prog.anchored_start && search_pos > 0) break;

        if (prog.has_ascii_filter) {
            while (search_pos < n) {
                const auto b = static_cast<uint8_t>(data[search_pos]);
                if (b >= 0x80 || ((prog.start_ascii[b >> 6] >> (b & 63)) & 1u) != 0) break;
                ++search_pos;
            }
        }

        size_t match_start = NO_MATCH;
        size_t match_end = NO_MATCH;
        for (size_t pos = search_pos; pos <= n; pos = next_cp_offset(inside, pos)) {
            if (prog.anchored_start && pos > 0) break;
            if (prog.has_ascii_filter && pos < n) {
                const auto b = static_cast<uint8_t>(data[pos]);
                if (b < 0x80 && ((prog.start_ascii[b >> 6] >> (b & 63)) & 1u) == 0) continue;
            }
            const size_t end = run_vm(prog, inside, pos, scratch);
            if (end != NO_MATCH) {
                match_start = pos;
                match_end = end;
                break;
            }
        }

        if (match_start == NO_MATCH) break;

        if (prev != match_start) {
            splits.push_back({{prev, match_start}, false});
        }
        splits.push_back({{match_start, match_end}, true});
        prev = match_end;

        if (match_start == match_end) {
            search_pos = next_cp_offset(inside, match_end);
        } else {
            search_pos = match_end;
        }
    }

    if (prev != n) {
        splits.push_back({{prev, n}, false});
    }
    if (splits.empty() && inside.empty()) {
        splits.push_back({{0, 0}, false});
    }
    return splits;
}

} // namespace tokenizers
