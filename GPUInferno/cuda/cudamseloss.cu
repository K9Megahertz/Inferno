#include "cudaops.h"

namespace Inferno {


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

    template<typename AT, typename BT, typename RT>
    __global__ void mse_loss_elementwise_kernel(const AT* a, const BT* b, RT* temp, size_t numel) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < numel) {
            RT diff = static_cast<RT>(a[idx]) - static_cast<RT>(b[idx]);
            temp[idx] = diff * diff;
        }
    }


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


    // explicit instantiations
    template void cuda_mse_loss<int, int, int>(
        const int*, const int*, int*,        
        size_t);

    template void cuda_mse_loss<float, float, float>(
        const float*, const float*, float*,        
        size_t);

    template void cuda_mse_loss<double, double, double>(
        const double*, const double*, double*,
        size_t);

    template void cuda_mse_loss<int, float, float>(
        const int*, const float*, float*,
        size_t);

    template void cuda_mse_loss<float, int, float>(
        const float*, const int*, float*,
        size_t);

    template void cuda_mse_loss<int, double, double>(
        const int*, const double*, double*,
        size_t);

    template void cuda_mse_loss<double, int, double>(
        const double*, const int*, double*,
        size_t);

    template void cuda_mse_loss<float, double, double>(
        const float*, const double*, double*,
        size_t);

    template void cuda_mse_loss<double, float, double>(
        const double*, const float*, double*,
        size_t);




}