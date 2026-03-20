#include "transposebackward.h"


namespace Inferno {


	TransposeBackward::TransposeBackward(const Tensor& A, int dima, int dimb) : m_A(A), m_dima(dima), m_dimb(dimb) {


	}

	void TransposeBackward::backward() {

		Tensor g_out = Engine::grad_in(this, 0);

		Tensor g_a = g_out.transpose(m_dima, m_dimb);

		// send gradient upstream
		auto na = GetImpl(m_A)->grad_edge();

		if (na) {
			Engine::accumulate(na.get(), 0, g_a);
		}


	}

	void TransposeBackward::release() {
		// drop references so graph can free
		m_A = Tensor{};
		
	}

	void TransposeBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);
		
	}


}