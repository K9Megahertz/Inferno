#include <vector>
#include "cudaops.h"
#include "gpu_atomic.cuh"

namespace Inferno {

    
    template<typename AT>
    __global__ void sum_to_shape_kernel(const AT* src_ptr, AT* dst_ptr, size_t src_numel, int src_rank, const size_t* src_shape, const size_t* temp_dst_strides)
    {
        size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (linear >= src_numel) return;

        size_t src_idx[45];
        size_t tmp = linear;

        for (int d = src_rank - 1; d >= 0; --d) {
            src_idx[d] = tmp % src_shape[d];
            tmp /= src_shape[d];
        }

        size_t dst_offset = 0;
        for (int d = 0; d < src_rank; ++d) {
            dst_offset += src_idx[d] * temp_dst_strides[d];
        }

        gpu_atomic_add(&dst_ptr[dst_offset], src_ptr[linear]);
    }

    template <typename AT>
    void cuda_sum_to_shape(AT* dst_ptr, const AT* src_ptr, size_t src_numel, size_t src_rank, const std::vector<size_t>& src_shape, const std::vector<size_t>& temp_dst_strides, size_t out_numel) {
        // Zero destination tensor on device
        check_cuda(cudaMemset(dst_ptr, 0, out_numel * sizeof(AT)), "sum_to_shape cudaMemset failed");

        // Copy metadata arrays to device
        size_t* d_src_shape = nullptr;
        size_t* d_temp_dst_strides = nullptr;

        check_cuda(cudaMalloc(&d_src_shape, src_rank * sizeof(size_t)), "sum_to_shape cudaMalloc d_src_shape failed");
        check_cuda(cudaMalloc(&d_temp_dst_strides, src_rank * sizeof(size_t)), "sum_to_shape cudaMalloc d_temp_dst_strides failed");
        check_cuda(cudaMemcpy(d_src_shape, src_shape.data(), src_rank * sizeof(size_t), cudaMemcpyHostToDevice), "sum_to_shape cudaMemcpy d_src_shape failed");
        check_cuda(cudaMemcpy(d_temp_dst_strides, temp_dst_strides.data(), src_rank * sizeof(size_t), cudaMemcpyHostToDevice), "sum_to_shape cudaMemcpy d_temp_dst_strides failed");

        constexpr int threads = 256;
        int blocks = static_cast<int>((src_numel + threads - 1) / threads);

        sum_to_shape_kernel<AT> << <blocks, threads >> > (src_ptr, dst_ptr, src_numel, static_cast<int>(src_rank), d_src_shape, d_temp_dst_strides);

        check_cuda(cudaGetLastError(), "sum_to_shape kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "sum_to_shape kernel execution failed");

        check_cuda(cudaFree(d_src_shape), "cudaFree d_src_shape failed");
        check_cuda(cudaFree(d_temp_dst_strides), "cudaFree d_temp_dst_strides failed");
    }

    template void cuda_sum_to_shape<int>(int*, const int*, size_t, size_t, const std::vector<size_t>&, const std::vector<size_t>&, size_t);
    template void cuda_sum_to_shape<float>(float*, const float*, size_t, size_t, const std::vector<size_t>&, const std::vector<size_t>&, size_t);
    template void cuda_sum_to_shape<double>(double*, const double*, size_t, size_t, const std::vector<size_t>&, const std::vector<size_t>&, size_t);


}