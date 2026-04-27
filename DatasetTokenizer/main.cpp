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

    const size_t CHUNK_SIZE = 8 * 1024 * 1024; // 8 MB
    std::vector<char> buffer(CHUNK_SIZE);

    in.seekg(0, std::ios::end);
    uint64_t filesize = (uint64_t)in.tellg();
    in.seekg(0, std::ios::beg);

    std::cout << "Total bytes to process: " << filesize << std::endl;

    while (in.read(buffer.data(), CHUNK_SIZE) || in.gcount() > 0) {
        size_t n = in.gcount();

        std::string chunk(buffer.data(), n);

        std::vector<uint32_t> tokens = tokenizer.encode(chunk);

        out.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(uint32_t));

        total_bytes += n;
        total_tokens += tokens.size();

        double percent = (static_cast<double>(total_bytes) / static_cast<double>(filesize)) * 100.0;
        std::cout << "Bytes Processed: " << total_bytes << "  Percent complete: " << std::fixed << std::setprecision(2) << percent << "%" << std::endl;
    }

    std::cout << "Tokens written: " << total_tokens << std::endl;

    in.close();
    out.close();
   

	return 0;
}