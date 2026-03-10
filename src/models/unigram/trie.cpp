#include "trie.h"

namespace tokenizers {

void Trie::push(const std::string& key) {
    Node* node = &root_;
    for (uint8_t byte : key) {
        auto it = node->children.find(byte);
        if (it == node->children.end()) {
            auto [inserted, _] = node->children.emplace(byte, std::make_unique<Node>());
            node = inserted->second.get();
        } else {
            node = it->second.get();
        }
    }
    node->is_leaf = true;
}

std::vector<size_t> Trie::common_prefix_search(std::string_view text) const {
    std::vector<size_t> results;
    const Node* node = &root_;
    for (size_t i = 0; i < text.size(); ++i) {
        auto it = node->children.find(static_cast<uint8_t>(text[i]));
        if (it == node->children.end()) break;
        node = it->second.get();
        if (node->is_leaf) {
            results.push_back(i + 1);
        }
    }
    return results;
}

} // namespace tokenizers
