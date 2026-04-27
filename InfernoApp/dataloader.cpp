#include <inferno/inferno.h>
#include "dataloader.h"


DataLoader::DataLoader(const std::string& token_file, size_t batch_size, size_t context_size) : m_batch_size(batch_size), m_context_size(context_size) {


	size_t chunk_bytes = 2ull * 1024ull * 1024ull;
	size_t chunk_tokens = chunk_bytes / sizeof(uint32_t);

	m_file.open(token_file, std::ios::binary);

	if (!m_file) {
		std::cerr << "Failed to open merges file: " << token_file << "\n";
		exit(1);
	}


	
	m_buffer.resize(chunk_tokens);
	m_file.read(reinterpret_cast<char*>(m_buffer.data()), chunk_bytes);	
	size_t tokens_read = m_file.gcount() / sizeof(uint32_t);
	if (tokens_read != chunk_tokens) 
		m_buffer.resize(tokens_read);




	


}

std::pair<Inferno::Tensor, Inferno::Tensor> DataLoader::next_batch() {

	std::mt19937 rng(std::random_device{}());

	auto random_int = [&](size_t min, size_t max) {
		std::uniform_int_distribution<size_t> dist(min, max);
		return dist(rng);
	};


	std::vector<uint32_t> xvec(m_batch_size * m_context_size, 0);
	std::vector<uint32_t> yvec(m_batch_size * m_context_size, 0);

	for (size_t b = 0; b < m_batch_size; b++) {
		size_t start = random_int(0, m_buffer.size() - m_context_size - 1);

		for (size_t t = 0; t < m_context_size; t++) {
			size_t idx = b * m_context_size + t;

			xvec[idx] = static_cast<int32_t>(m_buffer[start + t]);
			yvec[idx] = static_cast<int32_t>(m_buffer[start + t + 1]);
		}
	}

	Inferno::Tensor x(Inferno::DType::Int32, std::move(xvec), { m_batch_size, m_context_size }, "x_batch", Inferno::Device::cpu());
	Inferno::Tensor y(Inferno::DType::Int32, std::move(yvec), { m_batch_size, m_context_size }, "y_batch", Inferno::Device::cpu());

	return { x,y };

}
