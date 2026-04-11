#pragma once


#include <inferno/optim/optimizers.h>
#include <inferno/core/tensor.h>
#include "cuda/cudaops.h"
#include "core/tensorimpl.h"
#include "core/dtype_dispatch.h"

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

        Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Performing SGD optmizier step");
        for (auto& p : m_params) {
            auto grad = GetImpl(*p)->grad();

            if (!grad) {
                continue;
            }

            if (p->device() != grad->device()) {
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD: param/grad device mismatch");
                exit(1);
            }

            Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Stepping on: " + p->name());

            if (p->shape() != grad->shape()) {
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "OptimizerSGD: param/grad shape mismatch");
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
                    Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using normal step path");
                    cpu_step_impl<AT,BT>(dptr, gptr, count);
                    break;

                    ////////////////////////////////////////////////////
                    // CUDA Code Path
                    ////////////////////////////////////////////////////
                case DeviceType::CUDA:
                    Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using normal step path");
                    cuda_step_impl<AT, BT>(dptr, gptr, count, m_lr);
                    break;

                default:
                    Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
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
            p->grad() = nullptr;
        }
    }   

}
