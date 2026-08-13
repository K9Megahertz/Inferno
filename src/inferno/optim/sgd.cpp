#pragma once


#include <inferno/optim/sgd.h>
#include <inferno/core/tensor.h>
#include "inferno/cuda/cudaops.h"
#include "inferno/core/tensorimpl.h"
#include "inferno/core/dtype_dispatch.h"

namespace Inferno {


    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  CTORS // DTORS
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    OptimizerSGD::OptimizerSGD(const std::vector<Tensor*>& parameters, float learning_rate) : m_params(parameters), m_lr(learning_rate) {}



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function step
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void OptimizerSGD::step() {

        INFERNO_LOG_DEBUG() << "Performing SGD optmizier step" << std::endl;
        for (auto& p : m_params) {
            auto grad = GetImpl(*p)->grad();

            if (!grad) {
                continue;
            }

            if (p->device() != grad->device()) {
                INFERNO_LOG_ERROR() << "OptimizerSGD: param/grad device mismatch" << std::endl;
                exit(1);
            }

            INFERNO_LOG_DEBUG() << "Stepping on: " << p->name() << std::endl;

            if (p->shape() != grad->shape()) {
                INFERNO_LOG_ERROR() << "OptimizerSGD: param/grad shape mismatch" << std::endl;
                exit(1);
            }

            dispatchFloatTwo(p->dtype(), grad->dtype(), [&](auto TA, auto TB) {
                using AT = typename decltype(TA)::type;
                using BT = typename decltype(TB)::type;


                AT* dptr = GetImpl(*p)->data_as_ptr<AT>();
                BT* gptr = GetImpl(*grad)->data_as_ptr<BT>();

                size_t count = p->numel();

                switch (p->device().m_type) {

                    ////////////////////////////////////////////////////
                    // CPU Code Path
                    ////////////////////////////////////////////////////
                case DeviceType::CPU:
                    INFERNO_LOG_DEBUG() << "CPU Code path - Using normal step path" << std::endl;
                    cpu_sgd_step_impl<AT,BT>(dptr, gptr, count);
                    break;

                    ////////////////////////////////////////////////////
                    // CUDA Code Path
                    ////////////////////////////////////////////////////
                case DeviceType::CUDA:
                    INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal step path" << std::endl;
                    cuda_sgd_step_impl<AT, BT>(dptr, gptr, count, m_lr);
                    break;

                default:
                    INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
                    exit(1);
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

    void OptimizerSGD::zero_grad() {
        for (auto& p : m_params) {
            if (p != nullptr) {
                p->grad() = nullptr;
            }
        }
    }   

}
