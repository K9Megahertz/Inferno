#include "cudaops.h"

namespace  Inferno {

    template <typename AT, typename BT>
    __global__ void sgd_step_kernel(AT* dptr, const BT* gptr, size_t N, float lr)
    {
        size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < N) {
            dptr[i] = static_cast<AT>(
                static_cast<double>(dptr[i]) - static_cast<double>(lr) * static_cast<double>(gptr[i])
                );
        }
    }

    template <typename AT, typename BT>
    void cuda_step_impl(AT* dptr, const BT* gptr, size_t N, float lr)
    {
    
        constexpr int threads = 256;
        int blocks = static_cast<int>((N + threads - 1) / threads);

        sgd_step_kernel<AT, BT> << <blocks, threads >> > (dptr, gptr, N, lr);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,std::string("sgd_step_kernel launch failed: ") + cudaGetErrorString(err));
            exit(1);
        }

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,std::string("sgd_step_kernel execution failed: ") + cudaGetErrorString(err));
            exit(1);
        }
  
    }


    // explicit instantiations
    template void cuda_step_impl<int, int>(
        int*, const int*,
        size_t,
        float);

    template void cuda_step_impl<int, float>(
        int*, const float*,
        size_t,
        float);

    template void cuda_step_impl<int, double>(
        int*, const double*,
        size_t,
        float);



    template void cuda_step_impl<float, int>(
        float*, const int*,
        size_t,
        float);

    template void cuda_step_impl<float, float>(
        float*, const float*,
        size_t,
        float);

    template void cuda_step_impl<float, double>(
        float*, const double*,
        size_t,
        float);



    template void cuda_step_impl<double, int>(
        double*, const int*,
        size_t,
        float);

    template void cuda_step_impl<double, float>(
        double*, const float*,
        size_t,
        float);

    template void cuda_step_impl<double, double>(
        double*, const double*,
        size_t,
        float);



}