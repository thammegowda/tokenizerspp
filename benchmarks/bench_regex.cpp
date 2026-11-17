/// Microbenchmark for tokenizerspp custom UTF-8 Unicode RegexPattern & UnicodeScripts.
/// Measures pre-tokenization throughput (MB/s, Mchars/s) across LLM regexes and corpora.

#include <tokenizers/pattern.h>
#include <tokenizers/pre_tokenized_string.h>
#include <tokenizers/pre_tokenizers.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

size_t count_utf8_chars(std::string_view s) noexcept {
    size_t count = 0;
    for (unsigned char b : s) {
        if ((b & 0xC0) != 0x80) ++count;
    }
    return count;
}

std::string repeat_to_size(std::string_view sample, size_t target_bytes) {
    std::string out;
    out.reserve(target_bytes + sample.size());
    while (out.size() < target_bytes) {
        out.append(sample);
    }
    return out;
}

struct BenchCase {
    const char* name;
    const char* pattern;
};

constexpr BenchCase REGEX_CASES[] = {
    {"GPT2_ByteLevel",
     R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"},
    {"Llama3",
     R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"},
    {"Qwen3",
     R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"},
    {"WordPunct", R"(\w+|[^\w\s]+)"},
    {"Whitespace", R"(\s+)"},
    {"NLCodec_WS",
     R"([\x{0009}-\x{000D}\x{001C}-\x{0020}\x{0085}\x{00A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]+)"},
};

struct Corpus {
    const char* name;
    std::string text;
    size_t chars;
};

void run_regex_bench(const BenchCase& bc, const Corpus& corpus, int warmups, int repeats) {
    const tokenizers::RegexPattern regex(bc.pattern);
    size_t slices = 0;

    for (int i = 0; i < warmups; ++i) {
        auto res = regex.find_matches(corpus.text);
        if (!res) std::abort();
        slices = res->size();
    }

    std::vector<double> ms_samples;
    ms_samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = Clock::now();
        auto res = regex.find_matches(corpus.text);
        const auto t1 = Clock::now();
        if (!res) std::abort();
        slices = res->size();
        ms_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms_samples.begin(), ms_samples.end());
    const double median_ms = ms_samples[ms_samples.size() / 2];
    const double mb_per_s = (static_cast<double>(corpus.text.size()) / (1024.0 * 1024.0)) / (median_ms / 1000.0);
    const double mchars_per_s = (static_cast<double>(corpus.chars) / 1e6) / (median_ms / 1000.0);

    std::printf("%-16s | %-14s | %7.2f ms | %7.1f MB/s | %7.2f Mchar/s | %8zu slices\n",
                bc.name, corpus.name, median_ms, mb_per_s, mchars_per_s, slices);
}

void run_scripts_bench(const Corpus& corpus, int warmups, int repeats) {
    const tokenizers::pre_tokenizers::UnicodeScripts scripts;
    size_t splits = 0;

    for (int i = 0; i < warmups; ++i) {
        tokenizers::PreTokenizedString pts(corpus.text);
        auto res = scripts.pre_tokenize(pts);
        if (!res) std::abort();
        splits = pts.get_splits(tokenizers::OffsetReferential::Original, tokenizers::OffsetType::Byte).size();
    }

    std::vector<double> ms_samples;
    ms_samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        tokenizers::PreTokenizedString pts(corpus.text);
        const auto t0 = Clock::now();
        auto res = scripts.pre_tokenize(pts);
        const auto t1 = Clock::now();
        if (!res) std::abort();
        splits = pts.get_splits(tokenizers::OffsetReferential::Original, tokenizers::OffsetType::Byte).size();
        ms_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms_samples.begin(), ms_samples.end());
    const double median_ms = ms_samples[ms_samples.size() / 2];
    const double mb_per_s = (static_cast<double>(corpus.text.size()) / (1024.0 * 1024.0)) / (median_ms / 1000.0);
    const double mchars_per_s = (static_cast<double>(corpus.chars) / 1e6) / (median_ms / 1000.0);

    std::printf("%-16s | %-14s | %7.2f ms | %7.1f MB/s | %7.2f Mchar/s | %8zu splits\n",
                "UnicodeScripts", corpus.name, median_ms, mb_per_s, mchars_per_s, splits);
}

} // namespace

int main() {
    constexpr size_t TARGET_BYTES = 2 * 1024 * 1024; // 2 MiB per corpus

    constexpr std::string_view EN_SAMPLE =
        "It's 2026, and we're benchmarking Kidi's lightweight C++23 runtime! "
        "Don't forget: binary search runs in O(log n) time across 1,048,576 sorted elements, "
        "whereas linear scans take O(n).    if (lo <= hi) { mid = lo + ((hi - lo) >> 1); }\r\n";

    constexpr std::string_view MULTI_SAMPLE =
        "English text, 日本語のテキストとカタカナ・ひらがなー漢字12345! "
        "Привет мир, как дела? مرحبا بالعالم ٤٥٦! नमस्ते दुनिया ७८९! "
        "한국어 텍스트 테스트 🚀✨ Café naïve résumé.\n";

    const std::vector<Corpus> corpora = {
        {"ASCII_ProseCode", repeat_to_size(EN_SAMPLE, TARGET_BYTES), 0},
        {"Multilingual", repeat_to_size(MULTI_SAMPLE, TARGET_BYTES), 0},
    };

    std::vector<Corpus> prepared = corpora;
    for (auto& c : prepared) c.chars = count_utf8_chars(c.text);

    std::printf("%-16s | %-14s | %10s | %12s | %15s | %15s\n",
                "Pattern", "Corpus (2 MiB)", "Median", "Throughput", "Char Rate", "Output");
    std::printf("-----------------+----------------+------------+--------------+-----------------+----------------\n");

    for (const auto& bc : REGEX_CASES) {
        for (const auto& corpus : prepared) {
            run_regex_bench(bc, corpus, 1, 5);
        }
    }
    for (const auto& corpus : prepared) {
        run_scripts_bench(corpus, 1, 5);
    }
    return 0;
}
