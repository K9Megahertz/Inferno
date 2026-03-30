#include "embeddingbackward.h"


namespace Inferno {


	EmbeddingBackward::EmbeddingBackward(const Tensor& A, const Tensor& B) : m_embeddings(A), m_token_ids(B) {


	}

	void EmbeddingBackward::backward() {

		NoGradGuard guard;
		// upstream gradient dL/d(output)
		Tensor g_out = Engine::grad_in(this, 0);
		
		Tensor g_a = scatter_add_embedding(m_embeddings, m_token_ids, g_out);

		// find parent nodes
		auto na = GetImpl(m_embeddings)->grad_edge();
		

		// send gradients upstream
		if (na)
			Engine::accumulate(na.get(), 0, g_a);

		


	}

	void EmbeddingBackward::release() {
		// drop references so graph can free
		m_embeddings = Tensor{};
		m_token_ids = Tensor{};

	}

	void EmbeddingBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_embeddings);
		out.push_back(m_token_ids);

		
	}


}