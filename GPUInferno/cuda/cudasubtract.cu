#include "cudaops.h"

namespace Inferno {

    constexpr int MAX_DIMS = 12;

    template<typename AT, typename BT, typename RT>
    __global__ void subtract_kernel_broadcast(const AT* aptr, const BT* bptr, RT* outptr, size_t out_numel, int out_rank,
        const size_t* a_padded, const size_t* b_padded, const size_t* out_shape,
        const size_t* a_strides, const size_t* b_strides) {


        size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (linear >= out_numel) return;

        size_t out_idx[MAX_DIMS];
        size_t tmp = linear;


        //convert linear to multidimensional index  i.e. 88 -> {1,1,4}
        for (int d = out_rank - 1; d >= 0; --d) {
            out_idx[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }

        size_t a_offset = 0;
        size_t b_offset = 0;

        for (int d = 0; d < out_rank; ++d) {
            const size_t a_idx = (a_padded[d] == 1) ? 0 : out_idx[d];
            const size_t b_idx = (b_padded[d] == 1) ? 0 : out_idx[d];

            a_offset += a_idx * a_strides[d];
            b_offset += b_idx * b_strides[d];
        }

        outptr[linear] = static_cast<RT>(aptr[a_offset]) + static_cast<RT>(bptr[b_offset]);
    }

    template<typename AT, typename BT, typename RT>
    void cuda_subtract(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& bshape,
        const std::vector<size_t>& out_shape, size_t out_numel) {

        const size_t out_rank = out_shape.size();

        if (out_rank > MAX_DIMS) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "cuda_subtract: out rank exceeds MAX_DIMS");
            exit(1);
        }

        // Left-pad A and B shapes to output rank
        std::vector<size_t> a_padded(out_rank, 1);
        std::vector<size_t> b_padded(out_rank, 1);

        const size_t a_rank_offset = out_rank - ashape.size();
        const size_t b_rank_offset = out_rank - bshape.size();

        for (size_t i = 0; i < ashape.size(); ++i) {
            a_padded[a_rank_offset + i] = ashape[i];
        }

        for (size_t i = 0; i < bshape.size(); ++i) {
            b_padded[b_rank_offset + i] = bshape[i];
        }

        // Contiguous padded strides
        std::vector<size_t> a_strides(out_rank, 1);
        std::vector<size_t> b_strides(out_rank, 1);

        if (out_rank > 0) {
            for (int d = static_cast<int>(out_rank) - 2; d >= 0; --d) {
                a_strides[d] = a_strides[d + 1] * a_padded[d + 1];
                b_strides[d] = b_strides[d + 1] * b_padded[d + 1];
            }
        }

        size_t* d_a_padded = nullptr;
        size_t* d_b_padded = nullptr;
        size_t* d_out_shape = nullptr;
        size_t* d_a_strides = nullptr;
        size_t* d_b_strides = nullptr;

        check_cuda(cudaMalloc(&d_a_padded, out_rank * sizeof(size_t)), "cuda_subtract cudaMalloc d_a_padded failed");
        check_cuda(cudaMalloc(&d_b_padded, out_rank * sizeof(size_t)), "cuda_subtract cudaMalloc d_b_padded failed");
        check_cuda(cudaMalloc(&d_out_shape, out_rank * sizeof(size_t)), "cuda_subtract cudaMalloc d_out_shape failed");
        check_cuda(cudaMalloc(&d_a_strides, out_rank * sizeof(size_t)), "cuda_subtract cudaMalloc d_a_strides failed");
        check_cuda(cudaMalloc(&d_b_strides, out_rank * sizeof(size_t)), "cuda_subtract cudaMalloc d_b_strides failed");

        check_cuda(cudaMemcpy(d_a_padded, a_padded.data(), out_rank * sizeof(size_t), cudaMemcpyHostToDevice), "cuda_subtract cudaMemcpy d_a_padded failed");
        check_cuda(cudaMemcpy(d_b_padded, b_padded.data(), out_rank * sizeof(size_t), cudaMemcpyHostToDevice), "cuda_subtract cudaMemcpy d_b_padded failed");
        check_cuda(cudaMemcpy(d_out_shape, out_shape.data(), out_rank * sizeof(size_t), cudaMemcpyHostToDevice), "cuda_subtract cudaMemcpy d_out_shape failed");
        check_cuda(cudaMemcpy(d_a_strides, a_strides.data(), out_rank * sizeof(size_t), cudaMemcpyHostToDevice), "cuda_subtract cudaMemcpy d_a_strides failed");
        check_cuda(cudaMemcpy(d_b_strides, b_strides.data(), out_rank * sizeof(size_t), cudaMemcpyHostToDevice), "cuda_subtract cudaMemcpy d_b_strides failed");

        constexpr int threads = 256;
        int blocks = static_cast<int>((out_numel + threads - 1) / threads);

        subtract_kernel_broadcast<AT, BT, RT> << <blocks, threads >> > (
            aptr,
            bptr,
            outptr,
            out_numel,
            static_cast<int>(out_rank),
            d_a_padded,
            d_b_padded,
            d_out_shape,
            d_a_strides,
            d_b_strides
            );

        check_cuda(cudaGetLastError(), "cuda_subtract kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "cuda_subtract kernel execution failed");

        check_cuda(cudaFree(d_a_padded), "cuda_subtract cudaFree d_a_padded failed");
        check_cuda(cudaFree(d_b_padded), "cuda_subtract cudaFree d_b_padded failed");
        check_cuda(cudaFree(d_out_shape), "cuda_subtract cudaFree d_out_shape failed");
        check_cuda(cudaFree(d_a_strides), "cuda_subtract cudaFree d_a_strides failed");
        check_cuda(cudaFree(d_b_strides), "cuda_subtract cudaFree d_b_strides failed");
    }

    // explicit instantiations
    template void cuda_subtract<int, int, int>(
        const int*, const int*, int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<float, float, float>(
        const float*, const float*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<double, double, double>(
        const double*, const double*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<int, float, float>(
        const int*, const float*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<float, int, float>(
        const float*, const int*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<int, double, double>(
        const int*, const double*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<double, int, double>(
        const double*, const int*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<float, double, double>(
        const float*, const double*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

    template void cuda_subtract<double, float, double>(
        const double*, const float*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t);

}