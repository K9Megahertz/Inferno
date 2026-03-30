#include "reshapebackward.h"


namespace Inferno {

	ReshapeBackward::ReshapeBackward(const Tensor& A) : m_A(A) {


	}

	void ReshapeBackward::backward() {
		NoGradGuard guard;

		// upstream gradient dL/d(out)
		Tensor g_out = Engine::grad_in(this, 0);

		// reshape grad back to original input shape
		Tensor g_a = g_out.reshape(m_A.shape());

		// send gradient upstream
		auto na = GetImpl(m_A)->grad_edge();
		if (na) {
			Engine::accumulate(na.get(), 0, g_a);
		}
	}

	void ReshapeBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);
	}

	void ReshapeBackward::release() {
		m_A = Tensor{};
	}

}