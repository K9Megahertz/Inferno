#pragma once
#include <vector>
#include "../tensor.h"

namespace Inferno {
    class OptimizerSGD {
        std::vector<Tensor*> params;
        float lr;

    public:
        OptimizerSGD(const std::vector<Tensor*>& parameters, float learning_rate)
            : params(parameters), lr(learning_rate) {}

        template <typename AT, typename BT>
        void step_impl_cpu(Tensor& p, Tensor& g) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Stepping on: " + p.name());

            if (p.shape() != g.shape()) {
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD: param/grad shape mismatch");
                exit(1);
            }

            AT* dptr = GetImpl(p)->data_as_ptr<AT>();
            BT* gptr = GetImpl(g)->data_as_ptr<BT>();

            size_t count = p.numel();
            for (size_t i = 0; i < count; i++) {
                dptr[i] = static_cast<AT>(
                    static_cast<double>(dptr[i]) - static_cast<double>(lr) * static_cast<double>(gptr[i])
                    );
            }
        }

        void step() {
            for (auto& p : params) {
                auto grad = GetImpl(*p)->grad();

                if (!grad) {
                    continue;
                }

                if (p->device() != grad->device()) {
                    Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD: param/grad device mismatch");
                    exit(1);
                }

                dispatchTwo(p->dtype(), grad->dtype(), [&](auto TA, auto TB) {
                    using AT = typename decltype(TA)::type;
                    using BT = typename decltype(TB)::type;

                    if (p->device().is_cpu()) {
                        step_impl_cpu<AT, BT>(*p, *grad);
                    }
                    else {
                        Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD CUDA path not implemented");
                        exit(1);
                    }
                    });
            }
        }

        void zero_grad() {
            for (auto& p : params) {
                p->grad() = nullptr;
                //std::fill(p->m_node->m_grad.begin(), p->m_node->m_grad.end(), T(0));
                switch (p->dtype()) {

                case DType::Int32:
                case DType::Float32: {
                    //auto& gvec = p->grad_as<float>();
                    //std::fill(gvec.begin(), gvec.end(), 0.0f);
                }
                                   break;
                case DType::Float64: {
                    //auto& gvec = p->grad_as<double>();
                    //std::fill(gvec.begin(), gvec.end(), 0.0);
                }
                                   break;
                }
            }
        }
    };

}
