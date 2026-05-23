#include <inferno/inferno.h>
#include "dataloader.h"


DataLoader::DataLoader(const std::string& token_file, size_t batch_size, size_t context_size, size_t steps_per_chunk) : 
	m_batch_size(batch_size),
	m_context_size(context_size),
	m_steps_per_chunk(steps_per_chunk),
	m_chunk_step(0),
	m_rng(std::random_device{}()) {



	m_chunk_bytes = 256ull * 1024ull * 1024ull;
	m_chunk_tokens = m_chunk_bytes / sizeof(uint32_t);

	m_file.open(token_file, std::ios::binary);

	if (!m_file) {
		std::cerr << "Failed to open merges file: " << token_file << "\n";
		exit(1);
	}	
	
	m_buffer.resize(m_chunk_tokens);	
	load_random_chunk();

}


void DataLoader::load_random_chunk() {
	

	

	m_file.seekg(0, std::ios::end);
	uint64_t file_size = static_cast<uint64_t>(m_file.tellg());

	//every 4 bytes is a token
	uint64_t total_tokens = file_size / sizeof(uint32_t);

	//backup at least one chunk
	uint64_t max_start_token = total_tokens - m_chunk_tokens;

	//get start token
	std::uniform_int_distribution<uint64_t> dist(0, max_start_token);
	uint64_t start_token = dist(m_rng);


	//get offset in bytes
	uint64_t offset = start_token * sizeof(int32_t);
	
	//goto offset
	m_file.seekg(offset, std::ios::beg);


	m_file.read(reinterpret_cast<char*>(m_buffer.data()), m_chunk_tokens * sizeof(uint32_t));
	size_t tokens_read = m_file.gcount() / sizeof(uint32_t);
	if (tokens_read != m_chunk_tokens)
		m_buffer.resize(tokens_read);

	
	
}


std::pair<Inferno::Tensor, Inferno::Tensor> DataLoader::next_batch() {

	

	std::vector<uint32_t> xvec(m_batch_size * m_context_size, 0);
	std::vector<uint32_t> yvec(m_batch_size * m_context_size, 0);

	std::uniform_int_distribution<size_t> dist(0,m_buffer.size() - m_context_size - 1);

	for (size_t b = 0; b < m_batch_size; b++) {
		size_t start = dist(m_rng);

		for (size_t t = 0; t < m_context_size; t++) {
			size_t idx = b * m_context_size + t;

			xvec[idx] = static_cast<int32_t>(m_buffer[start + t]);
			yvec[idx] = static_cast<int32_t>(m_buffer[start + t + 1]);
		}
	}

	Inferno::Tensor x(Inferno::DType::Int32, std::move(xvec), { m_batch_size, m_context_size }, "x_batch", Inferno::Device::cpu());
	Inferno::Tensor y(Inferno::DType::Int32, std::move(yvec), { m_batch_size, m_context_size }, "y_batch", Inferno::Device::cpu());

	m_chunk_step++;

	if ((m_chunk_step % m_steps_per_chunk) == 0) {
		load_random_chunk();
		m_chunk_step = 0;
	}




	return { x,y };

}





DataLoader2::DataLoader2(const std::string& token_file, size_t batch_size, size_t context_size) :
	m_batch_size(batch_size),
	m_context_size(context_size),		
	m_rng(std::random_device{}()),
	m_file(token_file) {

	


	

}





std::pair<Inferno::Tensor, Inferno::Tensor> DataLoader2::next_batch() {


	std::vector<uint32_t> xvec(m_batch_size * m_context_size, 0);
	std::vector<uint32_t> yvec(m_batch_size * m_context_size, 0);

	size_t count = m_file.count_as<int32_t>();
	int32_t *tokens = m_file.data_as_ptr<int32_t>();
	

	std::uniform_int_distribution<size_t> dist(0, count - m_context_size - 1);

	for (size_t b = 0; b < m_batch_size; b++) {
		size_t start = dist(m_rng);

		for (size_t t = 0; t < m_context_size; t++) {
			size_t idx = b * m_context_size + t;

			xvec[idx] = tokens[start + t];
			yvec[idx] = tokens[start + t + 1];
		}
	}

	Inferno::Tensor x(Inferno::DType::Int32, std::move(xvec), { m_batch_size, m_context_size }, "x_batch", Inferno::Device::cpu());
	Inferno::Tensor y(Inferno::DType::Int32, std::move(yvec), { m_batch_size, m_context_size }, "y_batch", Inferno::Device::cpu());

	

	return { x,y };

}
