#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <pretokenizer/pretokenizer.h>


namespace InfernoTokenizer {

    struct TokenizerConfig {
        std::string merges_file;
        std::string vocab_file;
    };

    struct MergeEntry {
        uint32_t token;
        uint32_t rank;
    };

  

    class BPETokenizer {
    public:

        bool Initialize(const TokenizerConfig& config);

        std::vector<uint32_t> encode(const std::string& text);
        std::string decode(const std::vector<uint32_t>& tokens);
        

        void load_merges(const std::string& filename);
        void load_vocab(const std::string& file);

    private:
        
        Tokenizer::PreTokenizer tok;
        std::unordered_map<uint64_t, MergeEntry> m_mergemap;
        std::vector<std::string> m_vocablist;
    };

}