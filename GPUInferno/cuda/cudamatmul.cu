
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include "cudaops.h"

namespace Inferno {
    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function name
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    __global__ void matmul_kernel_broadcast(
        const AT* aptr,
        const BT* bptr,
        RT* outptr,
        int a_rank,
        int b_rank,
        int batch_rank,
        const size_t* out_batch_shape,
        const size_t* a_batch_shape,
        const size_t* b_batch_shape,
        const size_t* a_strides,
        const size_t* b_strides,
        size_t M,
        size_t K,
        size_t N,
        size_t total_batches)
    {

        //every thread will get its position in the final matrix. 
        //this will be the M and N of where it is in the matric as 
        //well as the batch_idx  i.e {0,0} or {0,1,0} or {0,1,2} etc...
        const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total_out = total_batches * M * N;

        if (linear >= total_out) {
            return;
        }

        // Decode linear index into:
        // batch_linear, m, n
        size_t tmp = linear;

        const size_t n = tmp % N;
        tmp /= N;

        const size_t m = tmp % M;
        tmp /= M;

        size_t batch_linear = tmp;

        size_t batch_idx[MAX_DIMS];


        //figure out the batch_idx using whats left over from tmp from above
        for (int d = batch_rank - 1; d >= 0; --d) {
            batch_idx[d] = batch_linear % out_batch_shape[d];
            batch_linear /= out_batch_shape[d];
        }
        
        size_t a_batch_offset = 0;
        size_t b_batch_offset = 0;


        //loop through the batch indices, this will be batch_rank in total, which is the total rank minus the M and N ranks. so total - 2
        for (int d = 0; d < batch_rank; ++d) {
            const size_t a_idx = (a_batch_shape[d] == 1) ? 0 : batch_idx[d];
            const size_t b_idx = (b_batch_shape[d] == 1) ? 0 : batch_idx[d];

            a_batch_offset += a_idx * a_strides[d];
            b_batch_offset += b_idx * b_strides[d];
        }

        const size_t a_base = a_batch_offset;// *(M * K);
        const size_t b_base = b_batch_offset;// *(K * N);

        RT sum = static_cast<RT>(0);

        for (size_t k = 0; k < K; ++k) {
            size_t a_idx = m * a_strides[a_rank - 2] + k * a_strides[a_rank - 1];
            size_t b_idx = k * b_strides[b_rank - 2] + n * b_strides[b_rank - 1];

            size_t a_offset = a_base + a_idx;
            size_t b_offset = b_base + b_idx;

            //const size_t a_offset = a_base + m * K + k;
            //const size_t b_offset = b_base + k * N + n;

            sum += static_cast<RT>(aptr[a_offset]) * static_cast<RT>(bptr[b_offset]);
        }

        outptr[linear] = sum;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function name
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    void cuda_matmul(
        const AT* aptr,
        const BT* bptr,
        RT* outptr,
        const std::vector<size_t>& a_shape,
        const std::vector<size_t>& a_strides,
        const std::vector<size_t>& b_shape,
        const std::vector<size_t>& b_strides,
        const std::vector<size_t>& out_shape)
    {
        const size_t a_rank = a_shape.size();
        const size_t b_rank = b_shape.size();
        const size_t out_rank = out_shape.size();

        if (a_rank < 2 || b_rank < 2 || out_rank < 2) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "cuda_matmul requires rank >= 2 tensors");
            exit(1);
        }

        const size_t M = a_shape[a_rank - 2];
        const size_t K = a_shape[a_rank - 1];
        const size_t N = b_shape[b_rank - 1];

        // Batch dims only
        std::vector<size_t> a_batch_shape(a_shape.begin(), a_shape.end() - 2);
        std::vector<size_t> b_batch_shape(b_shape.begin(), b_shape.end() - 2);
        std::vector<size_t> out_batch_shape(out_shape.begin(), out_shape.end() - 2);

        const size_t batch_rank = out_batch_shape.size();

        if (batch_rank > MAX_DIMS) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "cuda_matmul: batch rank exceeds MAX_DIMS");
            exit(1);
        }
           

        // Total number of broadcasted batches
        size_t total_batches = std::accumulate(out_batch_shape.begin(), out_batch_shape.end(), 1, std::multiplies<size_t>());


        size_t* d_out_batch_shape = nullptr;
        size_t* d_a_batch_shape = nullptr;
        size_t* d_b_batch_shape = nullptr;
        size_t* d_a_strides = nullptr;
        size_t* d_b_strides = nullptr;

        //if (batch_rank > 0) {
            check_cuda(cudaMalloc(&d_out_batch_shape, batch_rank * sizeof(size_t)), "cuda_matmul cudaMalloc d_out_batch_shape failed");
            check_cuda(cudaMalloc(&d_a_batch_shape, batch_rank * sizeof(size_t)), "cuda_matmul cudaMalloc d_a_batch_padded failed");
            check_cuda(cudaMalloc(&d_b_batch_shape, batch_rank * sizeof(size_t)), "cuda_matmul cudaMalloc d_b_batch_padded failed");
            check_cuda(cudaMalloc(&d_a_strides, a_rank * sizeof(size_t)), "cuda_matmul cudaMalloc d_a_batch_strides failed");
            check_cuda(cudaMalloc(&d_b_strides, b_rank * sizeof(size_t)), "cuda_matmul cudaMalloc d_b_batch_strides failed");

            check_cuda(cudaMemcpy(d_out_batch_shape,out_batch_shape.data(),batch_rank * sizeof(size_t),cudaMemcpyHostToDevice),"cuda_matmul cudaMemcpy d_out_batch_shape failed");
            check_cuda(cudaMemcpy(d_a_batch_shape,a_batch_shape.data(),batch_rank * sizeof(size_t),cudaMemcpyHostToDevice),"cuda_matmul cudaMemcpy d_a_batch_padded failed");
            check_cuda(cudaMemcpy(d_b_batch_shape,b_batch_shape.data(),batch_rank * sizeof(size_t),cudaMemcpyHostToDevice),"cuda_matmul cudaMemcpy d_b_batch_padded failed");
            check_cuda(cudaMemcpy(d_a_strides,a_strides.data(),a_rank * sizeof(size_t),cudaMemcpyHostToDevice),"cuda_matmul cudaMemcpy d_a_batch_strides failed");
            check_cuda(cudaMemcpy(d_b_strides,b_strides.data(),b_rank * sizeof(size_t),cudaMemcpyHostToDevice),"cuda_matmul cudaMemcpy d_b_batch_strides failed");
        //}

        const size_t total_out = total_batches * M * N;

        constexpr int threads = 256;
        const int blocks = static_cast<int>((total_out + threads - 1) / threads);

        matmul_kernel_broadcast<AT, BT, RT> << <blocks, threads >> > (
            aptr,
            bptr,
            outptr,
            static_cast<int>(a_rank),
            static_cast<int>(b_rank),
            static_cast<int>(batch_rank),
            d_out_batch_shape,
            d_a_batch_shape,
            d_b_batch_shape,
            d_a_strides,
            d_b_strides,
            M,
            K,
            N,
            total_batches
            );

        check_cuda(cudaGetLastError(), "cuda_matmul kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "cuda_matmul kernel execution failed");

        //if (batch_rank > 0) {
            check_cuda(cudaFree(d_out_batch_shape), "cuda_matmul cudaFree d_out_batch_shape failed");
            check_cuda(cudaFree(d_a_batch_shape), "cuda_matmul cudaFree d_a_batch_shape failed");
            check_cuda(cudaFree(d_b_batch_shape), "cuda_matmul cudaFree d_b_batch_shape failed");
            check_cuda(cudaFree(d_a_strides), "cuda_matmul cudaFree d_a_strides failed");
            check_cuda(cudaFree(d_b_strides), "cuda_matmul cudaFree d_b_strides failed");
        //}
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function name
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template void cuda_matmul<int, int, int>(
        const int*,
        const int*,
        int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<float, float, float>(
        const float*,
        const float*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<double, double, double>(
        const double*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<int, float, float>(
        const int*,
        const float*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<float, int, float>(
        const float*,
        const int*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<int, double, double>(
        const int*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<double, int, double>(
        const double*,
        const int*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<float, double, double>(
        const float*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul<double, float, double>(
        const double*,
        const float*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

}


    

    /*template <typename AT, typename BT, typename RT>
    __global__ void matmul_kernel(const AT* aptr,const BT* bptr, RT* outptr,size_t M,size_t K,size_t N)
    {
        size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        size_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += aptr[row * K + k] * bptr[k * N + col];
            }
            outptr[row * N + col] = sum;
        }
    }

    constexpr int TILE = 32;


    template <typename AT, typename BT, typename RT>
    __global__ void matmul_kernel_tiled(
        const AT* __restrict__ A,
        const BT* __restrict__ B,
        RT* __restrict__ C,
        int M,
        int K,
        int N)
    {
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        const int row = blockIdx.y * TILE + threadIdx.y;
        const int col = blockIdx.x * TILE + threadIdx.x;

        float sum = 0.0f;

        const int num_tiles = (K + TILE - 1) / TILE;

        for (int t = 0; t < num_tiles; ++t)
        {
            const int a_col = t * TILE + threadIdx.x;
            const int b_row = t * TILE + threadIdx.y;

            if (row < M && a_col < K)
                As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
            else
                As[threadIdx.y][threadIdx.x] = 0.0f;

            if (b_row < K && col < N)
                Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();

#pragma unroll
            for (int k = 0; k < TILE; ++k)
            {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < M && col < N)
        {
            C[row * N + col] = sum;
        }
    }

    template <typename AT, typename BT, typename RT>
    //void cuda_matmul(const AT* aptr, const BT* bptr, RT* outptr, size_t M, size_t K, size_t N) {
    void cuda_matmul(const AT* aptr, const BT* bptr, RT* optr, std::vector<size_t> a_padded_shape, std::vector<size_t> b_padded_shape, std::vector<size_t> out_shape) {
        //dim3 block(16, 16);
        //dim3 grid((N + block.x - 1) / block.x,(M + block.y - 1) / block.y);

        //matmul_kernel<AT,BT,RT> << <grid, block >> > (aptr, bptr, outptr, M, K, N);

        //dim3 block(Inferno::TILE, Inferno::TILE);
        //dim3 grid((N + Inferno::TILE - 1) / Inferno::TILE, (M + Inferno::TILE - 1) / Inferno::TILE);

        //Inferno::matmul_kernel_tiled<AT,BT,RT> << <grid, block >> > (aptr, bptr, outptr, M, K, N);

    }

    template void cuda_matmul<int, int, int>(const int*, const int*, int*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);
    template void cuda_matmul<int, float, float>(const int*, const float*, float*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);
    template void cuda_matmul<float, int, float>(const float*, const int*, float*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);

    template void cuda_matmul<float, float, float>(const float*, const float*, float*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);
    template void cuda_matmul<float, double, double>(const float*, const double*, double*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);
    template void cuda_matmul<double, float, double>(const double*, const float*, double*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);

    template void cuda_matmul<double, double, double>(const double*, const double*, double*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);
    template void cuda_matmul<double, int, double>(const double*, const int*, double*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);
    template void cuda_matmul<int, double, double>(const int*, const double*, double*, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>);*/



