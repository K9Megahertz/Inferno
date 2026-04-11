#include <inferno/functional/sigmoid.h>
#include "functional/sigmoid_kernels.h"
#include "core/tensorimpl.h"
#include "gradengine/engine.h"
#include "cuda/cudaops.h"
#include "gradfn/sigmoidbackward.h"
#include "core/dtype_dispatch.h"

namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function forward
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor Sigmoid::forward(const Tensor& A) {


		return dispatchAny(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;
			using RT = promote_t<AT, float>;

			auto implA = GetImpl(A);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();

			Inferno::Tensor out(dtype_of_v<RT>, A.shape(), "sigmoid", A.device(), true);

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();


			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using normal sigmoid path");
				cpu_sigmoid<AT,RT>(aptr, optr, out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using normal sigmoid path");
				cuda_sigmoid<AT,RT>(aptr, optr, out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad())) {
				implout->gradfn() = std::make_shared<SigmoidBackward>(A, out);
			}


			return out;
			});

	}


}