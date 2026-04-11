#include "cudaops.h"

namespace Inferno {


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function reduce_sum_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T>
    __global__ void reduce_sum_kernel(const T* input, T* output, size_t n)
    {
        extern __shared__ unsigned char shared_raw[];
        T* sdata = reinterpret_cast<T*>(shared_raw);


        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Load one element per thread into shared memory
        if (i < n)
            sdata[tid] = input[i];
        else
            sdata[tid] = static_cast<T>(0);

        __syncthreads();

        // Reduce within the block
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        // Write one result per block
        if (tid == 0)
        {
            output[blockIdx.x] = sdata[0];
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function mse_loss_elementwise_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    __global__ void mse_loss_elementwise_kernel(const AT* a, const BT* b, RT* temp, size_t numel) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < numel) {
            RT diff = static_cast<RT>(a[idx]) - static_cast<RT>(b[idx]);
            temp[idx] = diff * diff;
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_mse_loss
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    void cuda_mse_loss(const AT* a, const BT* b, RT* out, size_t numel) {
        constexpr int threads = 256;
        int blocks = static_cast<int>((numel + threads - 1) / threads);

        RT* temp = nullptr;
        cudaMalloc(&temp, numel * sizeof(RT));

        mse_loss_elementwise_kernel << <blocks, threads >> > (a, b, temp, numel);
        cudaDeviceSynchronize();

        RT* current_in = temp;
        size_t current_size = numel;

        std::vector<RT*> allocations;
        allocations.push_back(temp);

        while (current_size > 1) {
            int reduce_blocks = static_cast<int>((current_size + threads - 1) / threads);

            RT* partial = nullptr;
            cudaMalloc(&partial, reduce_blocks * sizeof(RT));
            allocations.push_back(partial);

            reduce_sum_kernel << <reduce_blocks, threads, threads * sizeof(RT) >> > (current_in, partial, current_size);
            cudaDeviceSynchronize();

            current_in = partial;
            current_size = reduce_blocks;
        }

        // divide by numel
        RT host_sum{};
        cudaMemcpy(&host_sum, current_in, sizeof(RT), cudaMemcpyDeviceToHost);
        host_sum /= static_cast<RT>(numel);
        cudaMemcpy(out, &host_sum, sizeof(RT), cudaMemcpyHostToDevice);

        for (RT* ptr : allocations) {
            cudaFree(ptr);
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    template void cuda_mse_loss<float, float, float>(const float*, const float*, float*, size_t);    
    template void cuda_mse_loss<float, double, double>(const float*, const double*, double*, size_t);

    template void cuda_mse_loss<double, float, double>(const double*, const float*, double*, size_t);
    template void cuda_mse_loss<double, double, double>(const double*, const double*, double*, size_t);
    
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function mse_loss_backward_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    __global__ void mse_loss_backward_kernel(
        const AT* aptr,
        const BT* bptr,
        RT* gaptr,
        RT* gbptr,
        const RT* gout,
        size_t numel)
    {
        size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i < numel)
        {
            RT upstream = gout[0];
            RT diff = static_cast<RT>(aptr[i]) - static_cast<RT>(bptr[i]);
            RT grad = upstream * static_cast<RT>(2) * diff / static_cast<RT>(numel);

            if (gaptr)
                gaptr[i] = grad;

            if (gbptr)
                gbptr[i] = -grad;
        }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_mse_loss_backward
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    void cuda_mse_loss_backward(
        const AT* aptr,
        const BT* bptr,
        RT* gaptr,
        RT* gbptr,
        const RT* gout,
        size_t numel)
    {
        constexpr int threads = 256;
        int blocks = static_cast<int>((numel + threads - 1) / threads);

        mse_loss_backward_kernel<AT, BT, RT> << <blocks, threads >> > (aptr, bptr, gaptr, gbptr, gout, numel);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, std::string("mse_loss_backward_kernel launch failed: ") + cudaGetErrorString(err)
            );
            exit(1);
        }

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, std::string("mse_loss_backward_kernel execution failed: ") + cudaGetErrorString(err)
            );
            exit(1);
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    template void cuda_mse_loss_backward<float, float, float>(const float*, const float*, float*, float*, const float*, size_t);
    template void cuda_mse_loss_backward<double, double, double>(const double*, const double*, double*, double*, const double*, size_t);

    template void cuda_mse_loss_backward<float, double, double>(const float*, const double*, double*, double*, const double*, size_t);
    template void cuda_mse_loss_backward<double, float, double>(const double*, const float*, double*, double*, const double*, size_t);

}