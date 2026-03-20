#include "mmbackward.h"


namespace Inferno {


	MMBackward::MMBackward(const Tensor& A, const Tensor& B) : m_A(A), m_B(B) {


	}

	void MMBackward::backward() {
        // upstream gradient dL/dY
        Tensor g_out = Engine::grad_in(this, 0);


        bool a_vec = (m_A.ndim() == 1);
        bool b_vec = (m_B.ndim() == 1);


		Tensor A2 = make_view(m_A, m_A.shape(), m_A.strides(), 0, "A2");
		Tensor B2 = make_view(m_B, m_B.shape(), m_B.strides(), 0, "B2");
		Tensor G2 = g_out;


		if (a_vec) {
			A2.shape() = { 1, A2.shape()[0] };
			A2.strides() = A2.calculate_strides(A2.shape());
		}

		if (b_vec) {
			B2.shape() = { B2.shape()[0], 1 };
			B2.strides() = B2.calculate_strides(B2.shape());
		}

		if (a_vec && b_vec) {
			G2.shape() = { 1, 1 };
			G2.strides() = G2.calculate_strides(G2.shape());
		}
		else if (a_vec) {
			G2.shape() = { 1, g_out.shape()[0]};
			G2.strides() = G2.calculate_strides(G2.shape());
		}
		else if (b_vec) {
			G2.shape() = { g_out.shape()[0], 1 };
			G2.strides() = G2.calculate_strides(G2.shape());
		}


        // dA = g_out @ B^T
        Tensor g_a = matmul(G2, B2.transpose(-1, -2));        

        // dB = A^T @ g_out
        Tensor g_b = matmul(A2.transpose(-1, -2), G2);
        

        // Reduce back to original input shapes in case batch broadcasting occurred
        g_a = sum_to_shape(g_a, GetImpl(m_A)->shape());
        g_b = sum_to_shape(g_b, GetImpl(m_B)->shape());


        // send upstream
        auto na = GetImpl(m_A)->grad_edge();
        auto nb = GetImpl(m_B)->grad_edge();

        if (na)
            Engine::accumulate(na.get(), 0, g_a);

        if (nb)
            Engine::accumulate(nb.get(), 0, g_b);

	}

	void MMBackward::release() {
		// drop references so graph can free
		m_A = Tensor{};
		m_B = Tensor{};
	}

	void MMBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);
		out.push_back(m_B);
	}


}