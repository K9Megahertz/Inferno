#include "crossentropylossbackward.h"
#include "inferno/gradengine/engine.h"
#include "inferno/cuda/cudaops.h"
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

    CrossEntropyLossBackward::CrossEntropyLossBackward(const Tensor& logits, const Tensor& target)
        : m_logits(logits), m_target(target) {
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function backward
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    void CrossEntropyLossBackward::backward() {
        Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,
            "CrossEntropyLossBackward::backward");

        Tensor g_out = Engine::grad_in(this, 0);

        /*if (!g_out.defined()) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "CrossEntropyLossBackward did not receive upstream gradient");
            exit(1);
        }*/

        if (m_target.dtype() != DType::Int32) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "CrossEntropyLossBackward requires Int32 targets");
            exit(1);
        }

        if (!m_logits.is_contiguous()) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "CrossEntropyLossBackward currently requires contiguous logits");
            exit(1);
        }

        if (!m_target.is_contiguous()) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "CrossEntropyLossBackward currently requires contiguous target");
            exit(1);
        }

        size_t rows = m_target.numel();
        size_t vocab_size = m_logits.shape().back();

        const int* tptr = GetImpl(m_target)->data_as_ptr<int>();

        dispatchFloat(m_logits.dtype(), [&](auto TLogits) {
            using LT = typename decltype(TLogits)::type;

            const LT* lptr = GetImpl(m_logits)->data_as_ptr<LT>();
            const LT* gptr = GetImpl(g_out)->data_as_ptr<LT>();

            Tensor grad_logits(
                dtype_of_v<LT>,
                m_logits.shape(),
                "cross_entropy_grad_logits",
                m_logits.device(),
                false
            );

            LT* grad_ptr = GetImpl(grad_logits)->data_as_ptr<LT>();

            switch (m_logits.device().m_type) {
            case DeviceType::CPU:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,
                    "CPU Code path - Using normal cross_entropy_loss_backward path");
                cpu_cross_entropy_loss_backward(
                    lptr,
                    tptr,
                    gptr,
                    grad_ptr,
                    rows,
                    vocab_size
                );
                break;

            case DeviceType::CUDA:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,
                    "CUDA Code path - Using normal cross_entropy_loss_backward path");
                cuda_cross_entropy_loss_backward(
                    lptr,
                    tptr,
                    gptr,
                    grad_ptr,
                    rows,
                    vocab_size
                );
                break;

            default:
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                exit(1);
            }

            Engine::accumulate(GetImpl(m_logits)->grad_edge().get(), 0, grad_logits);
            });
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function release
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void CrossEntropyLossBackward::release() {
        // drop references so graph can free
        m_logits = Tensor{};
        m_target = Tensor{};

    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function get_inputs
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void CrossEntropyLossBackward::get_inputs(std::vector<Tensor>& out) const {
        out.push_back(m_logits);
        // target does not receive gradients
    }



}