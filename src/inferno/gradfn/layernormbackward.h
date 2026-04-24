#pragma once
#include "node.h"
#include <inferno/core/tensor.h>
#include "inferno/core/broadcastops.h"



namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class LayerNormBackward 
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class LayerNormBackward : public Node {

	public:

		LayerNormBackward(const Tensor& A, const Tensor& gamma, const Tensor& beta, const Tensor& mean, const Tensor& invstd, size_t dim);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_A;		
		Tensor m_gamma;
		Tensor m_beta;
		Tensor m_mean;
		Tensor m_invstd;
		size_t m_dim;


	};


    template<typename AT, typename GT>
    void cpu_layernorm_backward(
        const AT* aptr,          // input x
        const AT* goutptr,       // upstream grad dL/dY
        const float* gptr,       // gamma
        const float* meanptr,    // saved mean per batch
        const float* invstdptr,  // saved invstd per batch
        GT* gaptr,               // output grad for input dL/dX
        float* ggptr,            // output grad for gamma dL/dGamma
        float* gbptr,            // output grad for beta dL/dBeta
        size_t num_batches,
        size_t dim
    ) {
        // gamma and beta grads must accumulate across all batches,
        // so start them at zero.
        for (size_t i = 0; i < dim; ++i) {
            ggptr[i] = 0.0f;
            gbptr[i] = 0.0f;
        }

        // Process one normalized row at a time.
        for (size_t b = 0; b < num_batches; ++b) {
            const size_t base = b * dim;

            const float mean = meanptr[b];
            const float invstd = invstdptr[b];

            // We need:
            // x_hat_i = (x_i - mean) * invstd
            // dxhat_i = gout_i * gamma_i
            //
            // Then:
            // dx_i = (1/dim) * invstd * (dim * dxhat_i - sum(dxhat) - x_hat_i * sum(dxhat * x_hat))

            float sum_dxhat = 0.0f;
            float sum_dxhat_xhat = 0.0f;

            // First pass:
            // - accumulate beta grad
            // - accumulate gamma grad
            // - compute reduction terms needed for input grad
            for (size_t i = 0; i < dim; ++i) {
                const float x = static_cast<float>(aptr[base + i]);
                const float gout = static_cast<float>(goutptr[base + i]);
                const float gamma = gptr[i];

                const float xhat = (x - mean) * invstd;
                const float dxhat = gout * gamma;

                gbptr[i] += gout;           // dBeta = sum(gout)
                ggptr[i] += gout * xhat;    // dGamma = sum(gout * xhat)

                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * xhat;
            }

            const float dim_f = static_cast<float>(dim);

            // Second pass: compute input grad
            for (size_t i = 0; i < dim; ++i) {
                const float x = static_cast<float>(aptr[base + i]);
                const float gout = static_cast<float>(goutptr[base + i]);
                const float gamma = gptr[i];

                const float xhat = (x - mean) * invstd;
                const float dxhat = gout * gamma;

                const float gx =
                    (invstd / dim_f) *
                    (dim_f * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);

                gaptr[base + i] = static_cast<GT>(gx);
            }
        }
    }

    template void cpu_layernorm_backward<float, float>(
        const float* aptr,
        const float* goutptr,
        const float* gptr,
        const float* meanptr,
        const float* invstdptr,
        float* gaptr,
        float* ggptr,
        float* gbptr,
        size_t num_batches,
        size_t dim
    );

    template void cpu_layernorm_backward<double, double>(
        const double* aptr,
        const double* goutptr,
        const float* gptr,
        const float* meanptr,
        const float* invstdptr,
        double* gaptr,
        float* ggptr,
        float* gbptr,
        size_t num_batches,
        size_t dim
    );

    template void cpu_layernorm_backward<int, float>(
        const int* aptr,
        const int* goutptr,
        const float* gptr,
        const float* meanptr,
        const float* invstdptr,
        float* gaptr,
        float* ggptr,
        float* gbptr,
        size_t num_batches,
        size_t dim
    );


}