#include "sigmoid.h"
#include "../cuda/cudaops.h"

namespace Inferno {


	Tensor Sigmoid::forward(const Tensor& A) {


		return dispatchOne(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;
			using RT = promote_t<AT, float>;

			auto implA = GetImpl(A);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();

			Inferno::Tensor out(dtype_of_v<RT>, A.shape(), "sigmoid", A.device());

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();


			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_sigmoid(aptr, optr, out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_sigmoid<AT>(aptr, optr, out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}


			implout->gradfn() = std::make_shared<SigmoidBackward>(A,out);


			return out;
			});

	}


}