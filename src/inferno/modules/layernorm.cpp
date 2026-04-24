#include <inferno/modules/layernorm.h>
#include "inferno/modules/layernorm_kernels.h"
#include "inferno/gradfn/layernormbackward.h"
#include "inferno/gradengine/engine.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/gradfn/layernormbackward.h"
#include "inferno/core/dtype_dispatch.h"

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

    LayerNorm::LayerNorm(size_t normalized_dim, float eps, Device device, DType dtype) : m_dim(normalized_dim), m_eps(eps) {
        dispatchFloat(dtype, [&](auto TagA) {
            using AT = typename decltype(TagA)::type;
            NoGradGuard guard; //incase we want to scale these later or something
            m_beta = Tensor(Inferno::DType::Float32, std::vector<float>(normalized_dim, 0.0f), { normalized_dim }, "ln_beta", device, true);
            m_gamma = Tensor(Inferno::DType::Float32, std::vector<float>(normalized_dim, 1.0f), { normalized_dim }, "ln_gamma", device, true);           
                
            });
        register_parameter("beta",&m_beta);
        register_parameter("gamma", &m_gamma);
        
    }


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

    Tensor LayerNorm::forward(Tensor& input) {

        const size_t num_batches = std::accumulate(input.shape().begin(), input.shape().end() - 1, static_cast<size_t>(1), std::multiplies<size_t>());

        return dispatchFloat(input.dtype(), [&](auto TA) {
            using AT = typename decltype(TA)::type;

            bool req = input.requires_grad() || m_gamma.requires_grad() || m_beta.requires_grad();

            Tensor out(input.dtype(), input.shape(), "layernorm", input.device(), req);

            // saved tensors for backward
            Tensor saved_mean(DType::Float32, { num_batches }, "ln_mean", input.device(), false);
            Tensor saved_invstd(DType::Float32, { num_batches }, "ln_invstd", input.device(), false);

            auto iptr = GetImpl(input)->data_as_ptr<AT>();
            auto optr = GetImpl(out)->data_as_ptr<AT>();

            auto gptr = GetImpl(m_gamma)->data_as_ptr<float>();
            auto bptr = GetImpl(m_beta)->data_as_ptr<float>();

            auto meanptr = GetImpl(saved_mean)->data_as_ptr<float>();
            auto invstdptr = GetImpl(saved_invstd)->data_as_ptr<float>();

            switch (input.device().m_type) {

            case DeviceType::CPU:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using normal layernorm path");
                cpu_layer_normalization<AT>(iptr,optr,gptr,bptr,meanptr,invstdptr,num_batches,m_dim,m_eps);
                break;

            case DeviceType::CUDA:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using normal layernorm path");
                cuda_layer_normalization<AT>(iptr,optr,gptr,bptr,meanptr,invstdptr,num_batches,m_dim,m_eps);
                break;

            default:
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                exit(1);
            }

            if (Inferno::grad_enabled && req) {
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "LayerNorm - Making a LayerNormBackward node");
                GetImpl(out)->gradfn() = std::make_shared<LayerNormBackward>(input, m_gamma, m_beta, saved_mean, saved_invstd, m_dim);
            }

            return out;
            });
    }
}