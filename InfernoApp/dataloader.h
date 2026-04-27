#pragma once
#include <fstream>


class Tensor;   // forward declaration



class DataLoader {
public:
    DataLoader(const std::string& token_file, size_t batch_size, size_t context_size);

    std::pair<Inferno::Tensor, Inferno::Tensor> next_batch();

private:

    std::vector<uint32_t> m_buffer;
    std::ifstream m_file;
    size_t m_batch_size;
    size_t m_context_size;
    size_t m_num_tokens;
};