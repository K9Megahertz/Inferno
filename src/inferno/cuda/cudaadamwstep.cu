#include "cudaops.h"


namespace Inferno {

    template<typename PT, typename GT>
    __global__ void cuda_adamw_step_kernel(
        PT* p,
        const GT* g,
        PT* m,
        PT* v,
        size_t count,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float bias_correction1,
        float bias_correction2
    ) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= count) {
            return;
        }

        float grad = static_cast<float>(g[i]);

        float mi = static_cast<float>(m[i]);
        float vi = static_cast<float>(v[i]);
        float pi = static_cast<float>(p[i]);

        mi = beta1 * mi + (1.0f - beta1) * grad;
        vi = beta2 * vi + (1.0f - beta2) * grad * grad;

        float m_hat = mi / bias_correction1;
        float v_hat = vi / bias_correction2;

        float update = m_hat / (sqrtf(v_hat) + eps);

        if (weight_decay != 0.0f) {
            pi *= (1.0f - lr * weight_decay);
        }

        pi -= lr * update;

        p[i] = static_cast<PT>(pi);
        m[i] = static_cast<PT>(mi);
        v[i] = static_cast<PT>(vi);
    }


    template<typename PT, typename GT>
    void cuda_adamw_step_impl(
        PT* p,
        const GT* g,
        PT* m,
        PT* v,
        size_t count,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float bias_correction1,
        float bias_correction2
    ) {
        int threads = 256;
        int blocks = static_cast<int>((count + threads - 1) / threads);

        cuda_adamw_step_kernel<PT, GT> << <blocks, threads >> > (
            p,
            g,
            m,
            v,
            count,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2
            );
    }


    
    template void cuda_adamw_step_impl<float, float>(
        float*, const float*, float*, float*,
        size_t, float, float, float, float, float, float, float);
    
    template void cuda_adamw_step_impl<float, double>(
        float*, const double*, float*, float*,
        size_t, float, float, float, float, float, float, float);

    template void cuda_adamw_step_impl<double, float>(
        double*, const float*, double*, double*,
        size_t, float, float, float, float, float, float, float);

    template void cuda_adamw_step_impl<double, double>(
        double*, const double*, double*, double*,
        size_t, float, float, float, float, float, float, float);



}