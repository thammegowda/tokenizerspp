/// Benchmark for tokenizerpp — measures encode and decode throughput.
/// Output format: key:value pairs for bench_h2h.py compatibility.
/// Usage: bench_cpp.out <tokenizer.json> <input.txt>

#include <tokenizers/tokenizer.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static size_t count_chars(const std::string& s) {
    size_t count = 0;
    for (size_t i = 0; i < s.size(); ) {
        auto b = static_cast<uint8_t>(s[i]);
        if (b < 0x80) i += 1;
        else if ((b & 0xE0) == 0xC0) i += 2;
        else if ((b & 0xF0) == 0xE0) i += 3;
        else i += 4;
        ++count;
    }
    return count;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer.json> <input.txt>\n";
        return 1;
    }

    const std::string tokenizer_path = argv[1];
    const std::string input_path = argv[2];

    try {
        // Load tokenizer
        auto load_start = std::chrono::high_resolution_clock::now();
        auto tok_result = tokenizers::Tokenizer::from_file(tokenizer_path);
        if (!tok_result) {
            std::cerr << "Failed to load tokenizer: " << tok_result.error().message() << "\n";
            return 1;
        }
        auto load_end = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();

        auto& tokenizer = *tok_result;

        // Read input
        std::string text = read_file(input_path);
        size_t num_chars = count_chars(text);

        // Benchmark encode
        auto encode_start = std::chrono::high_resolution_clock::now();
        auto enc_result = tokenizer.encode(text, false);
        auto encode_end = std::chrono::high_resolution_clock::now();

        if (!enc_result) {
            std::cerr << "Encode failed: " << enc_result.error().message() << "\n";
            return 1;
        }

        double encode_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
        size_t num_tokens = enc_result->len();
        double encode_tps = num_tokens / (encode_ms / 1000.0);

        // Benchmark decode
        auto decode_start = std::chrono::high_resolution_clock::now();
        auto dec_result = tokenizer.decode(enc_result->get_ids(), false);
        auto decode_end = std::chrono::high_resolution_clock::now();

        if (!dec_result) {
            std::cerr << "Decode failed: " << dec_result.error().message() << "\n";
            return 1;
        }

        double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
        double decode_tps = num_tokens / (decode_ms / 1000.0);

        // Output in key:value format
        std::printf("load_time_ms:%ld\n", load_ms);
        std::printf("encode_time_ms:%.0f\n", encode_ms);
        std::printf("decode_time_ms:%.0f\n", decode_ms);
        std::printf("num_tokens:%zu\n", num_tokens);
        std::printf("num_chars:%zu\n", num_chars);
        std::printf("tokens_per_sec:%.2f\n", encode_tps);
        std::printf("decode_tokens_per_sec:%.2f\n", decode_tps);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
