#pragma once
/// @file models/bpe/word.h
/// Internal BPE Word data structure (linked-list of symbols with merge_all).

#include "tokenizers/models.h"
#include <queue>
#include <vector>

namespace tokenizers {
namespace models {

struct BPESymbol {
    TokenId c;    // token ID
    int prev;      // index of previous symbol (-1 if none)
    int next;      // index of next symbol (-1 if none)
    TokenId len;  // byte length this symbol covers
};

struct BPEMerge {
    TokenId rank;
    TokenId pos;   // index of left symbol
    TokenId new_id;
};

struct BPEMergeGreater {
    bool operator()(const BPEMerge& a, const BPEMerge& b) const {
        if (a.rank != b.rank) return a.rank > b.rank;
        return a.pos > b.pos;
    }
};

/// Vector-based linked list of symbols supporting BPE merge_all.
class BPEWord {
public:
    void add(TokenId c, TokenId len);
    void merge_all(const MergeMap& merges);
    std::vector<Token> to_tokens(
        const std::unordered_map<TokenId, std::string>& vocab_r) const;
    bool empty() const { return symbols_.empty(); }

private:
    std::vector<BPESymbol> symbols_;
};

} // namespace models
} // namespace tokenizers
