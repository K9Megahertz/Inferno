#include "mselossbackward.h"


namespace Inferno {


	MSELossBackward::MSELossBackward(const Tensor& A, const Tensor& B) : m_A(A), m_B(B) {


	}

	void MSELossBackward::backward() {

        Tensor g_out = Engine::grad_in(this, 0); // usually shape {1}

        dispatchTwo(m_A.dtype(), m_B.dtype(), [&](auto TA, auto TB) {
            using AT = typename decltype(TA)::type;
            using BT = typename decltype(TB)::type;
            using RT = promote_t<AT, BT>;

            

            Tensor g_a(dtype_of_v<RT>, m_A.shape(), "mse_grad_a", m_A.device());
            Tensor g_b(dtype_of_v<RT>, m_B.shape(), "mse_grad_b", m_B.device());

            auto aptr = GetImpl(m_A)->data_as_ptr<AT>();
            auto bptr = GetImpl(m_B)->data_as_ptr<BT>();
            auto gout = GetImpl(g_out)->data_as_ptr<RT>();
            auto gaptr = GetImpl(g_a)->data_as_ptr<RT>();
            auto gbptr = GetImpl(g_b)->data_as_ptr<RT>();


			switch (g_out.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_mse_loss_backward(aptr, bptr, gaptr, gbptr, gout, m_A.numel());

				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				//cuda_mse_loss_backward(aptr, bptr, gaptr, gbptr, gout, m_A.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}
            
            auto na = GetImpl(m_A)->grad_edge();
            auto nb = GetImpl(m_B)->grad_edge();

            if (na)
                Engine::accumulate(na.get(), 0, g_a);

            if (nb)
                Engine::accumulate(nb.get(), 0, g_b);
            });

	}

	void MSELossBackward::release() {
		// drop references so graph can free
		m_A = Tensor{};
		m_B = Tensor{};
	}

	void MSELossBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);
		out.push_back(m_B);
	}


}