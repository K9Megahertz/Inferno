#include <inferno/functional/gelu.h>
#include "inferno/functional/gelu_kernels.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/gradengine/engine.h"
#include "inferno/core/tensorimpl.h"
#include "inferno/gradfn/gelubackward.h"
#include "inferno/core/dtype_dispatch.h"

namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function gelu
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor gelu(const Tensor& A) {

		return dispatchAny(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;
			using RT = promote_t<AT, float>;
					

			Inferno::Tensor out(dtype_of_v<RT>, A.shape(), "gelu", A.device(), true);

			auto implA = GetImpl(A);
			auto implout = GetImpl(out);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto optr = implout->data_as_ptr<RT>();


			std::vector<size_t> shape = implA->shape();
			std::vector<size_t> astrides = implA->strides();
			size_t aoffset = implA->offset();
			std::vector<size_t> ostrides = implout->strides();
			size_t ooffset = implout->offset();		


			const size_t N = A.numel();
			const size_t off = A.offset();

			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:				
				if (A.is_contiguous() && out.is_contiguous()) {
					Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using optimized gulu path");
					cpu_gelu<AT,RT>(aptr, optr, N, aoffset);
				}
				else {
					Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using strided gulu path");
					cpu_gelu_strided<AT, RT>(aptr, optr, shape, astrides, ostrides, aoffset, ooffset);
				}
				
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:				
				if (A.is_contiguous() && out.is_contiguous()) {
					Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using optimized gelu path");
					cuda_gelu<AT, RT>(aptr, optr, N, aoffset);
					
				}
				else {
					Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using strided gelu path");
					cuda_gelu_strided<AT, RT>(aptr, optr, shape, astrides, ostrides, aoffset, ooffset);
				}				
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad())) {
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Gelu - Making an GeluBackward node");
				implout->gradfn() = std::make_shared<GeluBackward>(A, out);
			}


			return out;
			});

	}


}