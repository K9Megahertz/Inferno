#include <inferno/core/tensor.h>
#include <inferno/loss/loss.h>
#include "inferno/gradfn/mselossbackward.h"
#include "inferno/gradengine/engine.h"
#include "inferno/core/cpuops.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/core/dtype_dispatch.h"

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

	Tensor MSELoss::forward(Tensor& prediction, Tensor& target) {
		if (prediction.device() != target.device()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Incompatible device types on tensor parameters in mse_loss");
			exit(1);
		}

		if (prediction.shape() != target.shape()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Shape mismatch on tensor parameters in mse_loss");
			exit(1);
		}

		return dispatchFloatTwo(prediction.dtype(), target.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;

			auto implpred = GetImpl(prediction);
			auto impltarget = GetImpl(target);

			auto aptr = implpred->data_as_ptr<AT>();
			auto bptr = impltarget->data_as_ptr<BT>();

			Tensor out(dtype_of_v<RT>, std::vector<size_t>{1}, "mse_loss", prediction.device(), true);
			auto implOut = GetImpl(out);
			auto optr = implOut->data_as_ptr<RT>();

			switch (prediction.device().m_type) {

			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using normal mse_loss path");
				cpu_mse_loss<AT, BT, RT>(aptr, bptr, optr, prediction.numel());
				break;

			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using normal mse_loss path");
				cuda_mse_loss<AT, BT, RT>(aptr, bptr, optr, prediction.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}


			if ((Inferno::grad_enabled) && (prediction.requires_grad())) {
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MSELoss - Making a MSELossBackward node");
				implOut->gradfn() = std::make_shared<MSELossBackward>(prediction, target);
			}

			return out;
			});
	}		
}