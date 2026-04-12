#include <inferno/core/tensor.h>
#include <inferno/loss/crossentropyloss.h>
#include "gradfn/crossentropylossbackward.h"
#include "gradengine/engine.h"
#include "core/cpuops.h"
#include "cuda/cudaops.h"
#include "core/dtype_dispatch.h"










namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function forward
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor CrossEntropyLoss::forward(Tensor& logits, Tensor& target) {
        if (logits.device() != target.device()) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"Incompatible device types on tensor parameters in cross_entropy_loss");
            exit(1);
        }

        if (logits.ndim() < 2) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"cross_entropy_loss requires logits rank >= 2");
            exit(1);
        }

        if (target.ndim() != logits.ndim() - 1) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"cross_entropy_loss requires target rank = logits rank - 1");
            exit(1);
        }

        // logits shape [..., V]
        // target shape [...]
        for (size_t i = 0; i < target.ndim(); i++) {
            if (target.shape()[i] != logits.shape()[i]) {
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"Shape mismatch on tensor parameters in cross_entropy_loss");
                exit(1);
            }
        }

        if (target.dtype() != DType::Int32) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"cross_entropy_loss currently requires target dtype = Int32");
            exit(1);
        }

        if (!(logits.dtype() == DType::Float32 || logits.dtype() == DType::Float64)) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"cross_entropy_loss currently requires logits dtype = Float32 or Float64");
            exit(1);
        }

        // Optional safety check if your raw kernels assume contiguous memory.
        if (!logits.is_contiguous()) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"cross_entropy_loss currently requires contiguous logits");
            exit(1);
        }

        if (!target.is_contiguous()) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,"cross_entropy_loss currently requires contiguous target");
            exit(1);
        }

        size_t rows = target.numel();
        size_t vocab_size = logits.shape().back();

        auto implTarget = GetImpl(target);
        const int* tptr = implTarget->data_as_ptr<int>();

        return dispatchFloat(logits.dtype(), [&](auto TLogits) {
            using LT = typename decltype(TLogits)::type;

            auto implLogits = GetImpl(logits);
            const LT* lptr = implLogits->data_as_ptr<LT>();

            Tensor out(dtype_of_v<LT>, std::vector<size_t>{1}, "cross_entropy_loss", logits.device(), true);
            auto implOut = GetImpl(out);
            LT* optr = implOut->data_as_ptr<LT>();

            switch (logits.device().m_type) {
            case DeviceType::CPU:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,"CPU Code path - Using normal cross_entropy_loss path");
                cpu_cross_entropy_loss(lptr, tptr, optr, rows, vocab_size);
                break;

            case DeviceType::CUDA:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,"CUDA Code path - Using normal cross_entropy_loss path");
                cuda_cross_entropy_loss(lptr, tptr, optr, rows, vocab_size);
                break;

            default:
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                exit(1);
            }

            if ((Inferno::grad_enabled) && (logits.requires_grad())) {
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,
                    "CrossEntropyLoss - Making a CrossEntropyLossBackward node");
                implOut->gradfn() = std::make_shared<CrossEntropyLossBackward>(logits, target);
            }

            return out;
            });
    }

}