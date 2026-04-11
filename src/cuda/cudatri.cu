#include "cudaops.h"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

namespace Inferno {

    template<typename AT>
    __global__ void cuda_triu_kernel(
        const AT* aptr,
        AT* optr,
        const size_t* shape,
        const size_t* strides,
        size_t rank,
        size_t offset,
        size_t out_numel,
        int diagonal)
    {
        size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= out_numel) return;

        // unravel linear index -> multi-index, and compute input storage index
        size_t tmp = linear;
        size_t storage_idx = offset;

        size_t row = 0;
        size_t col = 0;

        // We do not need to store the full index vector.
        // Just compute each dimension index on the fly.
        for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
            size_t idx_d = tmp % shape[d];
            tmp /= shape[d];

            storage_idx += idx_d * strides[d];

            if (d == static_cast<int>(rank) - 2) row = idx_d;
            if (d == static_cast<int>(rank) - 1) col = idx_d;
        }

        if (static_cast<int>(col) - static_cast<int>(row) >= diagonal) {
            optr[linear] = aptr[storage_idx];
        }
        else {
            optr[linear] = static_cast<AT>(0);
        }
    }

    template<typename AT>
    void cuda_triu(
        const AT* aptr,
        AT* optr,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides,
        size_t offset,
        size_t out_numel,
        int diagonal)
    {
        const size_t rank = shape.size();
        if (rank < 2) {
            throw std::runtime_error("cuda_triu requires rank >= 2");
        }

        size_t* d_shape = nullptr;
        size_t* d_strides = nullptr;

        cudaMalloc(&d_shape, rank * sizeof(size_t));
        cudaMalloc(&d_strides, rank * sizeof(size_t));

        cudaMemcpy(d_shape, shape.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, strides.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice);

        constexpr int threads = 256;
        int blocks = static_cast<int>((out_numel + threads - 1) / threads);

        cuda_triu_kernel << <blocks, threads >> > (
            aptr,
            optr,
            d_shape,
            d_strides,
            rank,
            offset,
            out_numel,
            diagonal
            );

        cudaFree(d_shape);
        cudaFree(d_strides);
    }

    template void cuda_triu<int>(const int*, int*, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, int);
    template void cuda_triu<float>(const float*, float*, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, int);
    template void cuda_triu<double>(const double*, double*, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, int);

}