
#include <inferno/optim/adamw.h>
#include <inferno/gradengine/engine.h>
#include <inferno/core/dtype_dispatch.h>
#include "inferno/cuda/cudaops.h"

namespace Inferno {


    void OptimizerAdamW::step() {
        NoGradGuard guard;

        float lr = m_lr;

        int warmup_steps = 2000;  // start with this

        if (m_step < warmup_steps) {
            lr = m_lr * (float(m_step + 1)  / float(warmup_steps));
        }

        INFERNO_LOG_DEBUG() << "Step: " << m_step << "  LR: " << lr << std::endl;

        m_step++;

        float bias_correction1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_step));
        float bias_correction2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_step));

        for (auto& [name, tensor] : m_parameters) {
            if (tensor == nullptr || !tensor->grad()) {
                continue;
            }

            Tensor* grad = tensor->grad().get();

            AdamWState& state = m_state[name];

            if (!state.initialized) {
                state.m = Tensor::zeros_like(*tensor);
                state.v = Tensor::zeros_like(*tensor);
                state.initialized = true;
            }

            dispatchFloatTwo(tensor->dtype(), grad->dtype(), [&](auto TA, auto TB) {
                using AT = typename decltype(TA)::type;
                using BT = typename decltype(TB)::type;

                AT* pptr = GetImpl(*tensor)->data_as_ptr<AT>();
                BT* gptr = GetImpl(*grad)->data_as_ptr<BT>();

                AT* mptr = GetImpl(state.m)->data_as_ptr<AT>();
                AT* vptr = GetImpl(state.v)->data_as_ptr<AT>();

                size_t count = tensor->numel();

                switch (tensor->device().m_type) {
                case DeviceType::CPU:
                    cpu_adamw_step_impl<AT, BT>(
                        pptr,
                        gptr,
                        mptr,
                        vptr,
                        count,
                        lr,
                        m_beta1,
                        m_beta2,
                        m_eps,
                        m_weight_decay,
                        bias_correction1,
                        bias_correction2
                    );
                    break;

                case DeviceType::CUDA:
                    cuda_adamw_step_impl<AT, BT>(
                        pptr,
                        gptr,
                        mptr,
                        vptr,
                        count,
                        lr,
                        m_beta1,
                        m_beta2,
                        m_eps,
                        m_weight_decay,
                        bias_correction1,
                        bias_correction2
                    );
                    break;

                default:
                    INFERNO_LOG_ERROR() << "Invalid device type";
                    std::exit(1);
                }
                });
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function zero_grad
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void OptimizerAdamW::zero_grad() {        
        for (auto& [name, tensor] : m_parameters) {
            if (tensor != nullptr) {
                tensor->grad() = nullptr;
            }
        }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function load_state_dict
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void OptimizerAdamW::load_state_dict(const AdamWStateDict& sd) {
        m_step = sd.step;
        m_lr = sd.lr;
        m_beta1 = sd.beta1;
        m_beta2 = sd.beta2;
        m_eps = sd.eps;
        m_weight_decay = sd.weight_decay;

        m_state.clear();

        for (const std::pair<std::string, Tensor*>& entry : m_parameters) {
            const std::string& name = entry.first;
            Tensor* param = entry.second;

            if (param == nullptr) {
                continue;
            }

            auto it = sd.states.find(name);

            if (it == sd.states.end()) {
                continue;
            }

            const AdamWParamState& loaded = it->second;

            if (loaded.m.shape() != param->shape()) {
                throw std::runtime_error("AdamW load_state_dict: m shape mismatch for " + name);
            }

            if (loaded.v.shape() != param->shape()) {
                throw std::runtime_error("AdamW load_state_dict: v shape mismatch for " + name);
            }

            if (loaded.m.dtype() != param->dtype()) {
                throw std::runtime_error("AdamW load_state_dict: m dtype mismatch for " + name);
            }

            if (loaded.v.dtype() != param->dtype()) {
                throw std::runtime_error("AdamW load_state_dict: v dtype mismatch for " + name);
            }

            AdamWState state;

            state.m = loaded.m.to(param->device());
            state.v = loaded.v.to(param->device());
            state.initialized = true;

            m_state[name] = state;
        }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function state_dict
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    AdamWStateDict OptimizerAdamW::state_dict() const {
        AdamWStateDict sd;

        sd.step = m_step;
        sd.lr = m_lr;
        sd.beta1 = m_beta1;
        sd.beta2 = m_beta2;
        sd.eps = m_eps;
        sd.weight_decay = m_weight_decay;

        for (const auto& [name, tensor] : m_parameters) {
            auto it = m_state.find(name);

            if (it == m_state.end() || !it->second.initialized) {
                continue;
            }

            sd.states[name] = {
                it->second.m,
                it->second.v
            };
        }

        return sd;
    }



}