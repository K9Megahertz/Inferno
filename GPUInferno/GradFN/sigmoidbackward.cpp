#include "sigmoidbackward.h"


namespace Inferno {


	SigmoidBackward::SigmoidBackward(const Tensor& A, const Tensor& out) : m_A(A), m_out(out) {


	}
	void SigmoidBackward::backward() {
		Tensor g_out = Engine::grad_in(this, 0);

		Tensor g_a = dispatchTwo(m_A.dtype(), g_out.dtype(), [&](auto TA, auto TG) {
			using AT = typename decltype(TA)::type;
			using GT = typename decltype(TG)::type;
			using RT = promote_t<AT, GT>;			

				


			Inferno::Tensor out(dtype_of_v<RT>, m_A.shape(), "SigmoidBackward", m_A.device());				


			//get pointers to data
			AT* aptr = GetImpl(m_out)->data_as_ptr<AT>();
			GT* gptr = GetImpl(g_out)->data_as_ptr<GT>();
			RT* optr = GetImpl(out)->data_as_ptr<RT>();


			switch (g_out.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_sigmoid_backward(aptr, gptr, optr, out.numel());

				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_sigmoid_backward<AT,GT,RT>(aptr, gptr, optr, out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			return out;

		});


		// find parent nodes
		auto na = GetImpl(m_A)->grad_edge();


		// send gradients upstream
		if (na)
			Engine::accumulate(na.get(), 0, g_a);
			

	}

	void SigmoidBackward::release() {
		// drop references so graph can free
		m_A = Tensor{};
		m_out = Tensor{};
		
	}

	void SigmoidBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_A);		
		
	}


}