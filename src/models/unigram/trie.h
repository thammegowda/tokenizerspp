#pragma once
/// @file models/unigram/trie.h
/// Byte-level prefix trie for Unigram model.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tokenizers {

class Trie {
public:
    /// Insert a word (as bytes) into the trie.
    void push(const std::string& key);

    /// Return lengths of all prefixes of `text` present in the trie.
    std::vector<size_t> common_prefix_search(std::string_view text) const;

private:
    struct Node {
        bool is_leaf = false;
        std::unordered_map<uint8_t, std::unique_ptr<Node>> children;
    };
    Node root_;
};

} // namespace tokenizers
