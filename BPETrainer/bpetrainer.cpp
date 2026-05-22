#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "bpetrainer.h"



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function train()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BPETrainer::train(const BPETrainerConfig& config) {

    m_config = config;	

	// Process the input file.
	std::unordered_map<std::string, uint64_t> piece_freqs = process_file(m_config.input_file);
    std::cout << "Parsed input file into " << piece_freqs.size() << " pieces" << std::endl;

	//we now have a list of unique strings with the number of times they appeared in the corpus.
	std::cout << "Building corpus from map of piece frequencies" << std::endl;
	std::vector<CorpusEntry> corpus = build_corpus_from_piece_freqs(piece_freqs);


	std::cout << "Building pair map from corpus" << std::endl;
	PairMap pairmap = build_pairmap_from_corpus(corpus);

	std::cout << "Built pairmap with: " << pairmap.size() << " entries" << std::endl;

	
	size_t maxvocab = config.target_vocab_size + 256 - 1;      //whatever -s was
	size_t vocabsize = config.initial_token_count;   //starts off at 256

    //Initialize initial vocabulary
    for (uint32_t i = 0; i < 256; i++) {
        m_vocab[i] = { i };
    }

	uint64_t merges;
	while ((merges = merge_best_pair_optimized(corpus, pairmap, m_rules)) && (vocabsize < maxvocab)) {		
		vocabsize++;
		std::cout << "Merges done: " << merges << " Vocabsize is now: " << vocabsize << std::endl;
	}
}

void BPETrainer::add_normal_text(const std::string& text, std::unordered_map<std::string, uint64_t>& piece_map) {

    if (text.empty()) {
        return;
    }

    std::vector<std::string> pieces = m_tokenizer.split(text);

    for (const std::string& piece : pieces) {
        piece_map[piece]++;
    }
}


bool BPETrainer::find_special_at(const std::string& text, size_t pos, std::string& matched) const {

    matched.clear();

    for (const std::string& special : m_config.special_tokens) {

        if (special.empty()) {
            continue;
        }

        if (pos + special.size() > text.size()) {
            continue;
        }

        if (text.compare(pos, special.size(), special) == 0) {

            // Prefer longest match
            if (special.size() > matched.size()) {
                matched = special;
            }
        }
    }

    return !matched.empty();
}


void BPETrainer::process_special_aware_text(
    const std::string& text,
    std::unordered_map<std::string, uint64_t>& piece_map
) {

    std::string normal_buffer;

    size_t i = 0;

    while (i < text.size()) {

        std::string matched_special;

        if (find_special_at(text, i, matched_special)) {

            // Flush normal text before special token
            add_normal_text(normal_buffer, piece_map);

            normal_buffer.clear();

            // Skip special token entirely for BPE training
            i += matched_special.size();
        }
        else {

            normal_buffer.push_back(text[i]);

            i++;
        }
    }

    // Flush remaining normal text
    add_normal_text(normal_buffer, piece_map);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function process_file()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, uint64_t> BPETrainer::process_file(std::string& filename) {

    std::unordered_map<std::string, uint64_t> piece_map;

    std::ifstream in(filename, std::ios::binary);

    if (!in) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    const size_t BUFFER_SIZE = 1 << 24;

    size_t total_bytes = 0;

    std::vector<char> buffer(BUFFER_SIZE);

    in.seekg(0, std::ios::end);
    uint64_t filesize = static_cast<uint64_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::cout << filesize << std::endl;

    size_t max_special_len = 0;

    for (const std::string& special : m_config.special_tokens) {
        max_special_len = std::max(max_special_len, special.size());
    }

    size_t carry_len = 32;

    if (max_special_len > 0) {
        carry_len = std::max<size_t>(32, max_special_len - 1);
    }

    std::string carry;

    while (in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || in.gcount() > 0) {

        std::streamsize n = in.gcount();

        std::string chunk(buffer.data(), static_cast<size_t>(n));

        std::string data = carry + chunk;

        carry.clear();

        if (data.size() > carry_len) {

            std::string safe = data.substr(0, data.size() - carry_len);

            carry = data.substr(data.size() - carry_len);

            process_special_aware_text(safe, piece_map);
        }
        else {
            carry = data;
        }

        total_bytes += static_cast<size_t>(n);

        double percent =
            (static_cast<double>(total_bytes) / static_cast<double>(filesize)) * 100.0;

        std::cout
            << "Bytes Processed: "
            << total_bytes
            << "  Percent complete: "
            << std::fixed
            << std::setprecision(2)
            << percent
            << "%"
            << std::endl;
    }

    if (!carry.empty()) {
        process_special_aware_text(carry, piece_map);
    }

    return piece_map;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function build_corpus_from_piece_freqs()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<CorpusEntry> BPETrainer::build_corpus_from_piece_freqs(const std::unordered_map<std::string, uint64_t>& pf) {

    std::vector<CorpusEntry> corpus;
    corpus.reserve(pf.size());

    for (const auto& [tokenstring, count] : pf) {

        CorpusEntry ce;
        for (unsigned char c : tokenstring) {
            ce.symbols.push_back(c);
        }
        ce.freq = count;
        corpus.push_back(std::move(ce));
    }

    return corpus;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function build_pairmap_from_corpus()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PairMap BPETrainer::build_pairmap_from_corpus(std::vector<CorpusEntry>& corpus) {

    //Phase 1 - Iterate through the corpus and build the pair map
    PairMap pair_map;

    for (CorpusEntry& entry : corpus) {
        if (entry.symbols.size() >= 2) {
            for (size_t i = 0; i < entry.symbols.size() - 1; i++) {
                uint64_t flat = ((uint64_t)(entry.symbols[i]) << 32) | entry.symbols[i + 1];
                pair_map[flat] += entry.freq;
            }
        }
    }

    return pair_map;

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function merge_best_pair_optimized()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t BPETrainer::merge_best_pair_optimized(std::vector<CorpusEntry>& corpus, PairMap& pairmap, std::vector<MergeRule>& rules) {
    if (pairmap.empty()) {
        return 0;
    }

    auto pack_pair = [](uint32_t a, uint32_t b) -> uint64_t {
        return (uint64_t(a) << 32) | uint64_t(b);
    };

    auto decrement_pair = [&](uint64_t key, uint64_t amount) {
        auto it = pairmap.find(key);
        if (it == pairmap.end()) {
            std::cout << "ERROR: tried to decrement missing pair key " << key << std::endl;
            std::exit(1);
        }

        if (it->second < amount) {
            std::cout << "ERROR: pair count underflow for key " << key
                << " count=" << it->second
                << " amount=" << amount << std::endl;
            std::exit(1);
        }

        it->second -= amount;

        if (it->second == 0) {
            pairmap.erase(it);
        }
    };

    auto increment_pair = [&](uint64_t key, uint64_t amount) {
        pairmap[key] += amount;
    };

    // Phase 1 - find highest frequency pair
    std::pair<uint64_t, uint64_t> highestpair{0, 0};

    for (const std::pair<const uint64_t, uint64_t>& entry : pairmap) {
        if (entry.second > highestpair.second) {
            highestpair = entry;
        }
    }

    if (highestpair.second == 0) {
        return 0;
    }

    const uint32_t firsttoken = uint32_t(highestpair.first >> 32);
    const uint32_t secondtoken = uint32_t(highestpair.first & 0xFFFFFFFFu);

    const uint32_t merge_token = m_newtokenid;

    uint64_t mergesperformed = 0;

    // Phase 2 - scan corpus and merge without erase()
    for (CorpusEntry& entry : corpus) {
        if (entry.symbols.size() < 2) {
            continue;
        }

        const uint64_t weight = entry.freq;
        std::vector<uint32_t>& symbols = entry.symbols;

        size_t write = 0;
        size_t read = 0;

        while (read < symbols.size()) {
            if (read + 1 < symbols.size() &&
                symbols[read] == firsttoken &&
                symbols[read + 1] == secondtoken)
            {
                const bool hasleft = (write > 0);
                const bool hasright = (read + 2 < symbols.size());

                uint32_t left = 0;
                uint32_t right = 0;

                if (hasleft) {
                    left = symbols[write - 1];
                }

                if (hasright) {
                    right = symbols[read + 2];
                }

                // Remove old adjacent pairs
                if (hasleft) {
                    decrement_pair(pack_pair(left, firsttoken), weight);
                }

                decrement_pair(pack_pair(firsttoken, secondtoken), weight);

                if (hasright) {
                    decrement_pair(pack_pair(secondtoken, right), weight);
                }

                // Write merged token in-place
                symbols[write] = merge_token;
                write++;

                mergesperformed += weight;

                // Add new adjacent pairs
                if (hasleft) {
                    increment_pair(pack_pair(left, merge_token), weight);
                }

                if (hasright) {
                    increment_pair(pack_pair(merge_token, right), weight);
                }

                // Skip both tokens that got merged
                read += 2;
            }
            else {
                if (write != read) {
                    symbols[write] = symbols[read];
                }
                write++;
                read++;
            }
        }

        // Shrink once at the end
        symbols.resize(write);
    }

    // Phase 3 - record merge rule + vocab entry
    if (mergesperformed > 0) {
        rules.emplace_back(firsttoken, secondtoken);
        m_vocab[m_newtokenid] = concat(m_vocab[firsttoken], m_vocab[secondtoken]);
        m_newtokenid++;
    }

    return mergesperformed;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function concat()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<uint32_t> BPETrainer::concat(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;

    result.reserve(a.size() + b.size());

    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());

    return result;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function save()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BPETrainer::save() {


    
    std::ofstream mergesfile(m_config.mergerules_output_file, std::ios::binary);
    std::ofstream vocabfile(m_config.vocab_output_file, std::ios::binary);

    std::cout << "Writing merge file -> ";
    // Print all pieces and their frequencies.     
    uint32_t token = m_config.initial_token_count;
    for (const auto& rule : m_rules) {
        //rulesfile << "[" << rule.first << ", " << rule.second << "] --> " << token << std::endl;
        mergesfile << rule.first << " " << rule.second << std::endl;
        //std::cout << "[" << rule.first << ", " << rule.second << "] --> " << token << std::endl;
        token++;
    }
    std::cout << token-256 << " Merges written." << std::endl;

    

    std::cout << "Writing vocab file -> ";
    // Write vocab
    uint32_t count = 0;
    for (const auto& [tok, bytes] : m_vocab) { 

        vocabfile << tok << " : ";

        for (uint32_t b : bytes) {
            vocabfile << b << " ";
        }
        count++;

        vocabfile << "\n";
    }
    std::cout << count << " words written." << std::endl;

    mergesfile.close();
    vocabfile.close();

}
