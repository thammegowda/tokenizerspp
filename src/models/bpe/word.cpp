#include "models/bpe/word.h"

namespace tokenizers {
namespace models {

void BPEWord::add(TokenId c, TokenId len) {
    int idx = static_cast<int>(symbols_.size());
    int prev = idx > 0 ? idx - 1 : -1;
    if (prev >= 0) {
        symbols_[prev].next = idx;
    }
    symbols_.push_back({c, prev, -1, len});
}

void BPEWord::merge_all(const MergeMap& merges) {
    using PQ = std::priority_queue<BPEMerge, std::vector<BPEMerge>, BPEMergeGreater>;
    PQ queue;

    // Seed the queue with initial adjacent pairs
    for (size_t i = 0; i + 1 < symbols_.size(); ++i) {
        auto it = merges.find({symbols_[i].c, symbols_[i + 1].c});
        if (it != merges.end()) {
            queue.push({it->second.first, static_cast<TokenId>(i), it->second.second});
        }
    }

    while (!queue.empty()) {
        auto top = queue.top();
        queue.pop();

        int pos = static_cast<int>(top.pos);

        // Skip if this symbol was removed
        if (symbols_[pos].len == 0) continue;
        // Skip if no next
        if (symbols_[pos].next == -1) continue;

        int right = symbols_[pos].next;

        // Verify the pair still matches
        auto it = merges.find({symbols_[pos].c, symbols_[right].c});
        if (it == merges.end() || it->second.first != top.rank) continue;

        // Perform merge: left absorbs right
        symbols_[pos].c = top.new_id;
        symbols_[pos].len += symbols_[right].len;
        symbols_[pos].next = symbols_[right].next;

        // Mark right as removed
        symbols_[right].len = 0;

        // Update the prev pointer of the node after right
        if (symbols_[pos].next >= 0) {
            symbols_[symbols_[pos].next].prev = pos;
        }

        // Check new pair with left neighbor
        if (symbols_[pos].prev >= 0) {
            int left_n = symbols_[pos].prev;
            auto it2 = merges.find({symbols_[left_n].c, symbols_[pos].c});
            if (it2 != merges.end()) {
                queue.push({it2->second.first, static_cast<TokenId>(left_n), it2->second.second});
            }
        }

        // Check new pair with right neighbor
        if (symbols_[pos].next >= 0) {
            int right_n = symbols_[pos].next;
            auto it2 = merges.find({symbols_[pos].c, symbols_[right_n].c});
            if (it2 != merges.end()) {
                queue.push({it2->second.first, static_cast<TokenId>(pos), it2->second.second});
            }
        }
    }
}

std::vector<Token> BPEWord::to_tokens(
    const std::unordered_map<TokenId, std::string>& vocab_r) const {
    std::vector<Token> tokens;
    size_t offset = 0;
    for (const auto& sym : symbols_) {
        if (sym.len == 0) continue;
        auto it = vocab_r.find(sym.c);
        std::string value = (it != vocab_r.end()) ? it->second : "";
        tokens.emplace_back(sym.c, std::move(value), Offsets{offset, offset + sym.len});
        offset += sym.len;
    }
    return tokens;
}

} // namespace models
} // namespace tokenizers
