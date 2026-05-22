#include <inferno/functional/softmax.h>
#include "inferno/functional/softmax_kernels.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/gradengine/engine.h"
#include "inferno/core/tensorimpl.h"
#include "inferno/gradfn/softmaxbackward.h"
#include "inferno/core/dtype_dispatch.h"

namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function softmax
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor Softmax(Tensor& A, int axis) {


		return dispatchAny(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;
			using RT = promote_t<AT, float>;

			const size_t ndim = A.ndim();

			//adjust negative axis specification
			int ax = (axis < 0) ? int(ndim) + axis : axis;

			//valid axis?
			if (ax < 0 || ax >= int(ndim)) {
				INFERNO_LOG_DEBUG() << "softmax: invalid axis";
				exit(1);
			}

			bool req_grad = Inferno::grad_enabled && A.requires_grad();

			Inferno::Tensor out(dtype_of_v<RT>, A.shape(), "softmax", A.device(), req_grad);
			

			//get pointers to data
			auto aptr = GetImpl(A)->data_as_ptr<AT>();			
			auto optr = GetImpl(out)->data_as_ptr<RT>();		

			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal softmax path";
				cpu_softmax<AT,RT>(aptr, optr, A.shape(), A.strides(), out.strides(), A.offset(), out.offset(), ax);					
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal softmax path";
				//cuda_softmax(aptr, optr, outer, dim, inner, off, N);
				cuda_softmax<AT,RT>(aptr, optr, A.shape(), A.strides(), out.strides(), A.offset(), out.offset(), ax);
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type";
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad())) {
				INFERNO_LOG_DEBUG() << "Softmax - Making a SoftmaxBackward node";
				GetImpl(out)->gradfn() = std::make_shared<SoftmaxBackward>(A, out, ax);
			}


			return out;
			});

	}

	


}