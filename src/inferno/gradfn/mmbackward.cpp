#include "mmbackward.h"
#include "inferno/gradengine/engine.h"
#include "inferno/core/ops_impl.h"

namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  CTORS / DTORS
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	MMBackward::MMBackward(const Tensor& A, const Tensor& B) : m_A(A), m_B(B) {
		set_name("MMBackward");

	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function backward
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void MMBackward::backward() {
		NoGradGuard guard;

		Tensor g_out = Engine::grad_in(this, 0);
		

		const bool a_vec = (m_A.ndim() == 1);
		const bool b_vec = (m_B.ndim() == 1);

		Tensor A2 = make_view(m_A, m_A.shape(), m_A.strides(), m_A.offset(), "A2_backward");
		Tensor B2 = make_view(m_B, m_B.shape(), m_B.strides(), m_B.offset(), "B2_backward");
		Tensor G2 = make_view(g_out, g_out.shape(), g_out.strides(), g_out.offset(), "G2_backward");

		if (a_vec) {
			A2.shape() = { 1, m_A.shape()[0] };
			A2.strides() = { 0, m_A.strides()[0] };
		}

		if (b_vec) {
			B2.shape() = { m_B.shape()[0], 1 };
			B2.strides() = { m_B.strides()[0], 0 };
		}

		if (a_vec && b_vec) {
			// scalar grad -> [1,1]
			G2.shape() = { 1, 1 };
			G2.strides() = { 0, 0 };
		}
		else if (a_vec) {
			// output was [N], treat as [1,N]
			G2.shape() = { 1, g_out.shape()[0] };
			G2.strides() = { 0, g_out.strides()[0] };
		}
		else if (b_vec) {
			// output was [..., M], treat as [..., M, 1]
			std::vector<size_t> new_shape = g_out.shape();
			std::vector<size_t> new_strides = g_out.strides();

			new_shape.push_back(1);
			new_strides.push_back(0);

			G2.shape() = new_shape;
			G2.strides() = new_strides;
		}

		// dA = G @ B^T
		//Tensor BT = B2.transpose(-1, -2).contiguous();
		Tensor g_a = matmul_impl(G2, B2, "MMBackward_dA", false, true);
		
		//Tensor g_a = matmul_impl(G2, BT, "MMBackward_dA", false, false);
		
		// dB = A^T @ G
		//Tensor AT = A2.transpose(-1, -2).contiguous();
		Tensor g_b = matmul_impl(A2, G2, "MMBackward_dB", true, false);
		//Tensor g_b = matmul_impl(AT, G2, "MMBackward_dB", false, false);

		// Restore vector gradient shapes before reduction
		if (a_vec) {
			g_a.shape() = m_A.shape();
			g_a.strides() = g_a.calculate_strides(g_a.shape());
		}

		if (b_vec) {
			g_b.shape() = m_B.shape();
			g_b.strides() = g_b.calculate_strides(g_b.shape());
		}

		g_a = sum_to_shape(g_a, GetImpl(m_A)->shape());
		g_b = sum_to_shape(g_b, GetImpl(m_B)->shape());

		auto na = GetImpl(m_A)->grad_edge();
		auto nb = GetImpl(m_B)->grad_edge();

		if (na) {
			Engine::accumulate(na.get(), 0, g_a);
		}

		if (nb) {
			Engine::accumulate(nb.get(), 0, g_b);
		}
	}

	/*void MMBackward::backward() {

		NoGradGuard guard;
        // upstream gradient dL/dY
        Tensor g_out = Engine::grad_in(this, 0);

		INFERNO_LOG_DEBUG() << "g_out" << std::endl;
		INFERNO_LOG_DEBUG() << g_out << std::endl;

		INFERNO_LOG_DEBUG() << "m_A" << std::endl;
		INFERNO_LOG_DEBUG() << m_A << std::endl;

		INFERNO_LOG_DEBUG() << "m_B" << std::endl;
		INFERNO_LOG_DEBUG() << m_B << std::endl;


        bool a_vec = (m_A.ndim() == 1);
        bool b_vec = (m_B.ndim() == 1);


		Tensor A2 = make_view(m_A, m_A.shape(), m_A.strides(), m_A.offset(), "A2");
		Tensor B2 = make_view(m_B, m_B.shape(), m_B.strides(), m_B.offset(), "B2");
		Tensor G2 = g_out;


		INFERNO_LOG_DEBUG() << "G2" << std::endl;
		INFERNO_LOG_DEBUG() << G2 << std::endl;

		INFERNO_LOG_DEBUG() << "A2" << std::endl;
		INFERNO_LOG_DEBUG() << A2 << std::endl;

		INFERNO_LOG_DEBUG() << "B2" << std::endl;
		INFERNO_LOG_DEBUG() << B2 << std::endl;

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
		INFERNO_LOG_DEBUG() << "After vecs" << std::endl;
		INFERNO_LOG_DEBUG() << "G2" << std::endl;
		INFERNO_LOG_DEBUG() << G2 << std::endl;

		INFERNO_LOG_DEBUG() << "A2" << std::endl;
		INFERNO_LOG_DEBUG() << A2 << std::endl;

		INFERNO_LOG_DEBUG() << "B2" << std::endl;
		INFERNO_LOG_DEBUG() << B2 << std::endl;

		
        // dA = g_out @ B^T
        //Tensor g_a = matmul(G2, B2.transpose(-1, -2).contiguous(), "Backward");
		//Tensor g_a = matmul(G2, B2.transpose(-1, -2), "Backward");		
		Tensor g_a = matmul_impl(G2, B2, "Backward", false, true);
		

		
        // dB = A^T @ g_out
        //Tensor g_b = matmul(A2.transpose(-1, -2).contiguous(), G2, "Backward");
		//Tensor g_b = matmul(A2.transpose(-1, -2), G2, "Backward");
		Tensor g_b = matmul_impl(A2, G2, "Backward", true, false);
		
		//std::cout << g_a << std::endl;
		//std::cout << g_b << std::endl;
        

        // Reduce back to original input shapes in case batch broadcasting occurred

		INFERNO_LOG_DEBUG() << "sum_to_shape" << std::endl;
		INFERNO_LOG_DEBUG() << g_a << std::endl;
		INFERNO_LOG_DEBUG() << "to" << std::endl;
		INFERNO_LOG_DEBUG() << m_A << std::endl;
		
		

		INFERNO_LOG_DEBUG() << "sum_to_shape" << std::endl;
		INFERNO_LOG_DEBUG() << g_b << std::endl;
		INFERNO_LOG_DEBUG() << "to" << std::endl;
		INFERNO_LOG_DEBUG() << m_B << std::endl;
		
        g_a = sum_to_shape(g_a, GetImpl(m_A)->shape());
		//std::cout << "second" << std::endl;
        g_b = sum_to_shape(g_b, GetImpl(m_B)->shape());
		//std::cout << "done" << std::endl;

        // send upstream
        auto na = GetImpl(m_A)->grad_edge();
        auto nb = GetImpl(m_B)->grad_edge();

        if (na)
            Engine::accumulate(na.get(), 0, g_a);

        if (nb)
            Engine::accumulate(nb.get(), 0, g_b);
		//std::cout << g_a << std::endl;
		//std::cout << g_b << std::endl;

	}*/


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function release
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void MMBackward::release() {
		// drop references so graph can free
		m_A = Tensor{};
		m_B = Tensor{};
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function get_inputs
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void MMBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);
		out.push_back(m_B);
	}


}