cmake_minimum_required(VERSION 3.20)
project(tokenizerpp VERSION 0.1.0 LANGUAGES CXX)

# C++23 is set per-target (not globally) so that parent projects embedding
# tokenizerpp alongside C++17 libraries (e.g. ONNX Runtime) are not affected.
# When building standalone, default to C++23.
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TOKENIZERPP_BUILD_TESTS "Build tests" ON)
option(TOKENIZERPP_BUILD_BENCHMARKS "Build benchmarks" OFF)

# === Dependencies via FetchContent ===
include(FetchContent)

# nlohmann/json for JSON parsing
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# uni-algo for Unicode normalization (small footprint C++ Unicode library)
FetchContent_Declare(
    uni_algo
    GIT_REPOSITORY https://github.com/uni-algo/uni-algo.git
    GIT_TAG v1.2.0
)
FetchContent_MakeAvailable(uni_algo)
if(TARGET uni-algo)
    target_compile_definitions(uni-algo PUBLIC
        UNI_ALGO_DISABLE_COLLATE
        UNI_ALGO_DISABLE_SEGMENT_GRAPHEME
        UNI_ALGO_DISABLE_SEGMENT_WORD
        UNI_ALGO_DISABLE_SYSTEM_LOCALE
    )
endif()

# === Source files ===
set(TOKENIZERPP_SOURCES
    src/core/error.cpp
    src/core/normalized_string.cpp
    src/core/pre_tokenized_string.cpp
    src/core/encoding.cpp
    src/core/added_vocabulary.cpp
    src/core/post_processor.cpp
    src/core/pattern.cpp
    src/core/tokenizer.cpp
    src/core/serialization.cpp
    src/core/tokenizer_config.cpp

    src/normalizers/bert.cpp
    src/normalizers/unicode.cpp
    src/normalizers/strip.cpp
    src/normalizers/replace.cpp
    src/normalizers/prepend.cpp
    src/normalizers/byte_level.cpp
    src/normalizers/precompiled.cpp
    src/normalizers/sequence.cpp

    src/pre_tokenizers/bert.cpp
    src/pre_tokenizers/byte_level.cpp
    src/pre_tokenizers/whitespace.cpp
    src/pre_tokenizers/delimiter.cpp
    src/pre_tokenizers/digits.cpp
    src/pre_tokenizers/metaspace.cpp
    src/pre_tokenizers/punctuation.cpp
    src/pre_tokenizers/split.cpp
    src/pre_tokenizers/unicode_scripts.cpp
    src/pre_tokenizers/sequence.cpp
    src/pre_tokenizers/fixed_length.cpp

    src/models/bpe/model.cpp
    src/models/bpe/word.cpp
    src/models/wordpiece/model.cpp
    src/models/unigram/model.cpp
    src/models/unigram/lattice.cpp
    src/models/unigram/trie.cpp
    src/models/wordlevel/model.cpp

    src/processors/bert.cpp
    src/processors/roberta.cpp
    src/processors/byte_level.cpp
    src/processors/template.cpp
    src/processors/sequence.cpp

    src/decoders/bpe.cpp
    src/decoders/byte_level.cpp
    src/decoders/byte_fallback.cpp
    src/decoders/ctc.cpp
    src/decoders/fuse.cpp
    src/decoders/metaspace.cpp
    src/decoders/strip.cpp
    src/decoders/wordpiece.cpp
    src/decoders/sequence.cpp
    src/decoders/replace.cpp

    src/utils/regex.cpp
    src/utils/padding.cpp
    src/utils/truncation.cpp

    src/chat_template/jinja.cpp
)

add_library(tokenizerpp STATIC ${TOKENIZERPP_SOURCES})
target_compile_features(tokenizerpp PUBLIC cxx_std_23)
target_include_directories(tokenizerpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(tokenizerpp
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        uni-algo::uni-algo
)

# === Benchmarks ===
if(TOKENIZERPP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# === Tests ===
if(TOKENIZERPP_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(tokenizerpp_tests
        tests/test_pattern.cpp
        tests/test_normalized_string.cpp
        tests/test_encoding.cpp
        tests/test_pre_tokenizers.cpp
        tests/test_models.cpp
        tests/test_tokenizer.cpp
        tests/test_chat_template.cpp
    )
    target_link_libraries(tokenizerpp_tests PRIVATE tokenizerpp GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(tokenizerpp_tests)
endif()
