#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <infernotokenizer/bpetokenizer.h>





int main(int argc, char* argv[]) {


    std::string input_file;
    std::string output_file;
    std::string merges_file;
    std::string vocab_file;
    size_t vocab_size = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-i") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -i\n";
                return 1;
            }
            input_file = argv[++i];
        }
        else if (arg == "-m") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -m\n";
                return 1;
            }
            merges_file = argv[++i];
        }
        else if (arg == "-v") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -v\n";
                return 1;
            }
            vocab_file = argv[++i];
        }
        else if (arg == "-o") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -o\n";
                return 1;
            }
            output_file = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    // Validate required args
    if (input_file.empty() || output_file.empty() ||merges_file.empty() || vocab_file.empty()) {
        std::cerr << "Usage: -i <input> -o <output> -m <merges> -v <vocab>\n";
        return 1;
    }





	InfernoTokenizer::BPETokenizer tokenizer;

	InfernoTokenizer::TokenizerConfig config;

    config.merges_file = merges_file;
    config.vocab_file = vocab_file;

	tokenizer.Initialize(config);


    size_t total_bytes = 0;
    size_t total_tokens = 0;

    std::ifstream in(input_file, std::ios::binary);
    std::ofstream out(output_file, std::ios::binary);

    if (!in) {
        std::cerr << "Failed to open input file: " << input_file << "\n";
        exit(1);
    }

    if (!out) {
        std::cerr << "Failed to open output file: " << output_file << "\n";
        exit(1);
    }

    //setup sizes
    const size_t CHUNK_SIZE = 8 * 1024 * 1024; // 8 MB
    const size_t CARRY_SIZE = 1 * 1024 * 1024; // 1 MB

    //read buffer for reading in data
    std::vector<char> buffer(CHUNK_SIZE);


    //figure out how big the file is
    in.seekg(0, std::ios::end);
    uint64_t filesize = (uint64_t)in.tellg();
    in.seekg(0, std::ios::beg);



    std::string carry;

    std::cout << "Total bytes to process: " << filesize << std::endl;

    while (in.read(buffer.data(), CHUNK_SIZE) || in.gcount() > 0) {
        size_t n = in.gcount();

        std::string chunkstr(buffer.data(), n);

        std::string data = carry + chunkstr;

        carry.clear();


        if (data.size() > CARRY_SIZE) {

            //get the first part of the string, this is safe to process
            std::string safe = data.substr(0, data.size() - CARRY_SIZE);

            //strip off the last number of carrysize characters and save those for the next round
            carry = data.substr(data.size() - CARRY_SIZE);

            ///// PROCESS DATA //////
            std::vector<uint32_t> tokens = tokenizer.encode(safe);

            out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(uint32_t));

            total_bytes += n;
            total_tokens += tokens.size();

            double percent = (static_cast<double>(total_bytes) / static_cast<double>(filesize)) * 100.0;
            std::cout << "Bytes Processed: " << total_bytes << "  Percent complete: " << std::fixed << std::setprecision(2) << percent << "%" << std::endl;
        }
        else {
            carry = data;
        }
    }

    if (!carry.empty()) {

        ///// PROCESS DATA //////
        std::vector<uint32_t> tokens = tokenizer.encode(carry);

        out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(uint32_t));
        
        total_tokens += tokens.size();

        double percent = (static_cast<double>(total_bytes) / static_cast<double>(filesize)) * 100.0;
        std::cout << "Bytes Processed: " << total_bytes << "  Percent complete: " << std::fixed << std::setprecision(2) << percent << "%" << std::endl;
        
    }

    std::cout << "Tokens written: " << total_tokens << std::endl;

    in.close();
    out.close();
   

	return 0;
}