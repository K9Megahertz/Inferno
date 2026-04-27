#include <iostream>
#include <fstream>
#include <iomanip>
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

	
	size_t maxvocab = config.target_vocab_size;
	size_t vocabsize = config.initial_token_count;

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


    std::unordered_map<std::string, uint64_t>  piece_map;
    // Open the file in binary mode.
    // Binary mode is important because we want the exact bytes from disk.
    // We do not want newline translation or any text-mode behavior.
    std::ifstream in(filename, std::ios::binary);

    // If the file failed to open, throw an error.
    if (!in) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    // Size of each chunk we read from the file.
    // 1 << 20 = 1,048,576 bytes = 1 MB
    // 1 << 24 = 16,777,216 bytes = 16 MB
    const size_t BUFFER_SIZE = 1 << 24;
    size_t total_bytes = 0;
    // Temporary buffer that will hold each chunk from the stream.
    std::vector<char> buffer(BUFFER_SIZE);

    in.seekg(0, std::ios::end);
    uint64_t filesize = (uint64_t)in.tellg();
    in.seekg(0, std::ios::beg);

    std::cout << filesize << std::endl;
    // Try to read a full chunk.
    // If that fails because we hit the end of file, gcount() may still be > 0
    // for the final partial chunk, so we keep processing in that case too.
    while (in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || in.gcount() > 0) {

        // How many bytes were actually read this time.
        std::streamsize n = in.gcount();

        std::string chunk(buffer.data(), n);

        std::vector<std::string> pieces = m_tokenizer.split(chunk);


        for (auto& piece : pieces) {
            piece_map[piece]++;
        }



        /*// Process each byte in the chunk.
        for (std::streamsize i = 0; i < n; ++i) {

            // Convert char to unsigned char before classification.
            // This avoids negative-char problems.
            unsigned char b = static_cast<unsigned char>(buffer[i]);

            // Hand this byte to the tokenizer logic.
            process_byte(b);
        }*/
        total_bytes += n;
        double percent = (static_cast<double>(total_bytes) / static_cast<double>(filesize)) * 100.0;
        std::cout << "Bytes Processed: " << total_bytes << "  Percent complete: " << std::fixed << std::setprecision(2) << percent << "%" << std::endl;
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
