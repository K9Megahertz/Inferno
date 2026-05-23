#pragma once
#include <fstream>


class Tensor;   // forward declaration



class DataLoader {
public:
    DataLoader(const std::string& token_file, size_t batch_size, size_t context_size, size_t steps_per_chunk);

    std::pair<Inferno::Tensor, Inferno::Tensor> next_batch();
    void load_random_chunk();

private:

    std::vector<uint32_t> m_buffer;
    std::ifstream m_file;
    size_t m_batch_size;
    size_t m_context_size;
    size_t m_num_tokens;

    size_t m_chunk_bytes;
    size_t m_chunk_tokens;
    size_t m_steps_per_chunk;
    size_t m_chunk_step;

    std::mt19937 m_rng;

    
};

class DataLoader2 {
public:
    DataLoader2(const std::string& token_file, size_t batch_size, size_t context_size);

    std::pair<Inferno::Tensor, Inferno::Tensor> next_batch();
    

private:    
    
    size_t m_batch_size;
    size_t m_context_size;
    size_t m_num_tokens;

    std::mt19937 m_rng;

    Inferno::MemmappedFile m_file;
};