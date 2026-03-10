cmake_minimum_required(VERSION 3.20)
project(tokenizerpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
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

# PCRE2 for regex with lookahead/lookbehind support (needed by byte-level pre-tokenizer)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PCRE2 IMPORTED_TARGET libpcre2-8)
endif()
if(NOT PCRE2_FOUND)
    find_library(PCRE2_LIBRARY NAMES pcre2-8)
    find_path(PCRE2_INCLUDE_DIR NAMES pcre2.h)
    if(PCRE2_LIBRARY AND PCRE2_INCLUDE_DIR)
        set(PCRE2_FOUND TRUE)
    endif()
endif()

if(NOT PCRE2_FOUND)
    message(STATUS "PCRE2 not found on system, building from source")
    FetchContent_Declare(
        pcre2
        URL https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.44/pcre2-10.44.tar.gz
    )
    set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(pcre2)
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

    src/utils/regex.cpp
    src/utils/padding.cpp
    src/utils/truncation.cpp

    src/chat_template/jinja.cpp
)

add_library(tokenizerpp STATIC ${TOKENIZERPP_SOURCES})
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

# Link PCRE2
if(PCRE2_FOUND AND TARGET PkgConfig::PCRE2)
    target_link_libraries(tokenizerpp PRIVATE PkgConfig::PCRE2)
elseif(PCRE2_FOUND)
    target_link_libraries(tokenizerpp PRIVATE ${PCRE2_LIBRARY})
    target_include_directories(tokenizerpp PRIVATE ${PCRE2_INCLUDE_DIR})
else()
    target_link_libraries(tokenizerpp PRIVATE pcre2-8)
endif()

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
