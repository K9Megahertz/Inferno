#include "cudaops.h"

namespace Inferno {



    template<typename AT, typename RT>
    __global__ void sigmoid_kernel(const AT* aptr, RT* outptr, size_t N) {
        size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < N) {
            RT x = static_cast<RT>(aptr[i]);
            outptr[i] = static_cast<RT>(1) / (static_cast<RT>(1) + exp(-x));
        }
    }

    template<typename AT, typename RT>
    void cuda_sigmoid(const AT* aptr, RT* outptr, size_t N) {
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        sigmoid_kernel << <grid, block >> > (aptr, outptr, N);
    }

    template void cuda_sigmoid<int, float>(const int*, float*, size_t);
    template void cuda_sigmoid<int, double>(const int*, double*, size_t);

    template void cuda_sigmoid<float, float>(const float*, float*, size_t);
    template void cuda_sigmoid<float, double>(const float*, double*, size_t);

    template void cuda_sigmoid<double, double>(const double*, double*, size_t);


    template<typename AT, typename GT, typename RT>
    __global__ void sigmoid_backward_kernel(
        const AT* yptr,   // sigmoid output
        const GT* gptr,   // upstream gradient
        RT* outptr,       // gradient wrt input
        size_t n)
    {
        size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < n)
        {
            RT y = static_cast<RT>(yptr[i]);
            RT g = static_cast<RT>(gptr[i]);

            outptr[i] = g * y * (static_cast<RT>(1) - y);
        }
    }

    template<typename AT, typename GT, typename RT>
    void cuda_sigmoid_backward(const AT* yptr, const GT* gptr, RT* outptr, size_t n) {

        constexpr int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);

        sigmoid_backward_kernel<AT, GT, RT> << <blocks, threads >> > (yptr, gptr, outptr, n);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                std::string("sigmoid_backward_kernel launch failed: ") + cudaGetErrorString(err));
            exit(1);
        }

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                std::string("sigmoid_backward_kernel execution failed: ") + cudaGetErrorString(err));
            exit(1);
        }
    }


    template void cuda_sigmoid_backward<int, int, int>(const int*, const int*, int*, size_t);
    template void cuda_sigmoid_backward<int, float, float>(const int*, const float*, float*, size_t);
    template void cuda_sigmoid_backward<int, double, double>(const int*, const double*, double*, size_t);

    template void cuda_sigmoid_backward<float, int, float>(const float*, const int*, float*, size_t);
    template void cuda_sigmoid_backward<float, float, float>(const float*, const float*, float*, size_t);
    template void cuda_sigmoid_backward<float, double, double>(const float*, const double*, double*, size_t);

    template void cuda_sigmoid_backward<double, int, double>(const double*, const int*, double*, size_t);
    template void cuda_sigmoid_backward<double, float, double>(const double*, const float*, double*, size_t);
    template void cuda_sigmoid_backward<double, double, double>(const double*, const double*, double*, size_t);


}