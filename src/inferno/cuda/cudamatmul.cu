
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include "cudaops.h"

namespace Inferno {

    constexpr int MATMUL_TILE = 16;
    constexpr int MATMUL_FAST_TILE = 16;
    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function matmul_kernel_broadcast
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
    //  Function cuda_matmul
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
        //check_cuda(cudaDeviceSynchronize(), "cuda_matmul kernel execution failed");

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
    //  Explicit instantiations
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











    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //  Function matmul_kernel_tiled_broadcast
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT, int TILE>
    __global__ void matmul_kernel_tiled_broadcast(
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
        __shared__ RT As[TILE][TILE];
        __shared__ RT Bs[TILE][TILE];

        const size_t batch = static_cast<size_t>(blockIdx.z);

        const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
        const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;

        if (batch >= total_batches) {
            return;
        }

        // Decode batch index into multi-d batch indices
        size_t batch_linear = batch;
        size_t batch_idx[MAX_DIMS];

        for (int d = batch_rank - 1; d >= 0; --d) {
            batch_idx[d] = batch_linear % out_batch_shape[d];
            batch_linear /= out_batch_shape[d];
        }

        size_t a_batch_offset = 0;
        size_t b_batch_offset = 0;

        for (int d = 0; d < batch_rank; ++d) {
            const size_t a_idx = (a_batch_shape[d] == 1) ? 0 : batch_idx[d];
            const size_t b_idx = (b_batch_shape[d] == 1) ? 0 : batch_idx[d];

            a_batch_offset += a_idx * a_strides[d];
            b_batch_offset += b_idx * b_strides[d];
        }

        RT sum = static_cast<RT>(0);

        const size_t a_row_stride = a_strides[a_rank - 2];
        const size_t a_col_stride = a_strides[a_rank - 1];
        const size_t b_row_stride = b_strides[b_rank - 2];
        const size_t b_col_stride = b_strides[b_rank - 1];

        const int num_tiles = static_cast<int>((K + TILE - 1) / TILE);

        for (int t = 0; t < num_tiles; ++t) {
            const size_t kA = static_cast<size_t>(t) * TILE + threadIdx.x;
            const size_t kB = static_cast<size_t>(t) * TILE + threadIdx.y;

            // Load A tile
            if (row < M && kA < K) {
                const size_t a_offset = a_batch_offset + row * a_row_stride + kA * a_col_stride;
                As[threadIdx.y][threadIdx.x] = static_cast<RT>(aptr[a_offset]);
            }
            else {
                As[threadIdx.y][threadIdx.x] = static_cast<RT>(0);
            }

            // Load B tile
            if (kB < K && col < N) {
                const size_t b_offset = b_batch_offset + kB * b_row_stride + col * b_col_stride;
                Bs[threadIdx.y][threadIdx.x] = static_cast<RT>(bptr[b_offset]);
            }
            else {
                Bs[threadIdx.y][threadIdx.x] = static_cast<RT>(0);
            }

            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < TILE; ++kk) {
                sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            const size_t out_linear = batch * (M * N) + row * N + col;
            outptr[out_linear] = sum;
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function can_use_tiled_fast_path
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    inline bool can_use_tiled_fast_path(
        const std::vector<size_t>& a_shape,
        const std::vector<size_t>& a_strides,
        const std::vector<size_t>& b_shape,
        const std::vector<size_t>& b_strides,
        const std::vector<size_t>& out_shape)
    {
        const size_t a_rank = a_shape.size();
        const size_t b_rank = b_shape.size();
        const size_t out_rank = out_shape.size();

        if (a_rank < 2 || b_rank < 2 || out_rank < 2)
            return false;

        const size_t batch_rank = out_rank - 2;
        if (batch_rank > MAX_DIMS)
            return false;

        // Require regular row-major matrix layout in the last 2 dims
        // A[..., M, K]
        // B[..., K, N]
        if (a_strides[a_rank - 1] != 1) return false;
        if (b_strides[b_rank - 1] != 1) return false;
        if (a_strides[a_rank - 2] != a_shape[a_rank - 1]) return false;
        if (b_strides[b_rank - 2] != b_shape[b_rank - 1]) return false;

        // Require output to be contiguous row-major too
        if (out_shape.size() >= 2) {
            // We assume the caller allocated out contiguous, which your Tensor ctor does.
            // So we do not need output strides here.
        }

        // Batch dims must be regular or broadcasted-by-1 only.
        // Since your padded shapes are already aligned, we can validate that
        // any batch dim is either equal to out dim or 1.
        for (size_t d = 0; d < batch_rank; ++d) {
            if (!(a_shape[d] == out_shape[d] || a_shape[d] == 1))
                return false;
            if (!(b_shape[d] == out_shape[d] || b_shape[d] == 1))
                return false;
        }

        return true;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_matmul_fast
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    void cuda_matmul_fast(
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
                "cuda_matmul_fast requires rank >= 2 tensors");
            exit(1);
        }

        const size_t M = a_shape[a_rank - 2];
        const size_t K = a_shape[a_rank - 1];
        const size_t N = b_shape[b_rank - 1];

        std::vector<size_t> a_batch_shape(a_shape.begin(), a_shape.end() - 2);
        std::vector<size_t> b_batch_shape(b_shape.begin(), b_shape.end() - 2);
        std::vector<size_t> out_batch_shape(out_shape.begin(), out_shape.end() - 2);

        const size_t batch_rank = out_batch_shape.size();

        if (batch_rank > MAX_DIMS) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "cuda_matmul_fast: batch rank exceeds MAX_DIMS");
            exit(1);
        }

        // If layout is irregular, use your original implementation.
        if (!can_use_tiled_fast_path(a_shape, a_strides, b_shape, b_strides, out_shape)) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG,
                "cuda_matmul_fast falling back to cuda_matmul");
            cuda_matmul<AT, BT, RT>(
                aptr, bptr, outptr,
                a_shape, a_strides,
                b_shape, b_strides,
                out_shape
            );
            return;
        }

        size_t total_batches = 1;
        if (!out_batch_shape.empty()) {
            total_batches = std::accumulate(
                out_batch_shape.begin(),
                out_batch_shape.end(),
                static_cast<size_t>(1),
                std::multiplies<size_t>()
            );
        }

        size_t* d_out_batch_shape = nullptr;
        size_t* d_a_batch_shape = nullptr;
        size_t* d_b_batch_shape = nullptr;
        size_t* d_a_strides = nullptr;
        size_t* d_b_strides = nullptr;

        check_cuda(cudaMalloc(&d_out_batch_shape, batch_rank * sizeof(size_t)),
            "cuda_matmul_fast cudaMalloc d_out_batch_shape failed");
        check_cuda(cudaMalloc(&d_a_batch_shape, batch_rank * sizeof(size_t)),
            "cuda_matmul_fast cudaMalloc d_a_batch_shape failed");
        check_cuda(cudaMalloc(&d_b_batch_shape, batch_rank * sizeof(size_t)),
            "cuda_matmul_fast cudaMalloc d_b_batch_shape failed");
        check_cuda(cudaMalloc(&d_a_strides, a_rank * sizeof(size_t)),
            "cuda_matmul_fast cudaMalloc d_a_strides failed");
        check_cuda(cudaMalloc(&d_b_strides, b_rank * sizeof(size_t)),
            "cuda_matmul_fast cudaMalloc d_b_strides failed");

        if (batch_rank > 0) {
            check_cuda(cudaMemcpy(d_out_batch_shape, out_batch_shape.data(), batch_rank * sizeof(size_t), cudaMemcpyHostToDevice),
                "cuda_matmul_fast cudaMemcpy d_out_batch_shape failed");
            check_cuda(cudaMemcpy(d_a_batch_shape, a_batch_shape.data(), batch_rank * sizeof(size_t), cudaMemcpyHostToDevice),
                "cuda_matmul_fast cudaMemcpy d_a_batch_shape failed");
            check_cuda(cudaMemcpy(d_b_batch_shape, b_batch_shape.data(), batch_rank * sizeof(size_t), cudaMemcpyHostToDevice),
                "cuda_matmul_fast cudaMemcpy d_b_batch_shape failed");
        }

        check_cuda(cudaMemcpy(d_a_strides, a_strides.data(), a_rank * sizeof(size_t), cudaMemcpyHostToDevice),
            "cuda_matmul_fast cudaMemcpy d_a_strides failed");
        check_cuda(cudaMemcpy(d_b_strides, b_strides.data(), b_rank * sizeof(size_t), cudaMemcpyHostToDevice),
            "cuda_matmul_fast cudaMemcpy d_b_strides failed");

        dim3 block(MATMUL_TILE, MATMUL_TILE);
        dim3 grid(
            static_cast<unsigned int>((N + MATMUL_TILE - 1) / MATMUL_TILE),
            static_cast<unsigned int>((M + MATMUL_TILE - 1) / MATMUL_TILE),
            static_cast<unsigned int>(total_batches)
        );

        matmul_kernel_tiled_broadcast<AT, BT, RT, MATMUL_TILE> << <grid, block >> > (
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

        check_cuda(cudaGetLastError(), "cuda_matmul_fast kernel launch failed");
        //check_cuda(cudaDeviceSynchronize(), "cuda_matmul_fast kernel execution failed");

        check_cuda(cudaFree(d_out_batch_shape), "cuda_matmul_fast cudaFree d_out_batch_shape failed");
        check_cuda(cudaFree(d_a_batch_shape), "cuda_matmul_fast cudaFree d_a_batch_shape failed");
        check_cuda(cudaFree(d_b_batch_shape), "cuda_matmul_fast cudaFree d_b_batch_shape failed");
        check_cuda(cudaFree(d_a_strides), "cuda_matmul_fast cudaFree d_a_strides failed");
        check_cuda(cudaFree(d_b_strides), "cuda_matmul_fast cudaFree d_b_strides failed");
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template void cuda_matmul_fast<int, int, int>(
        const int*,
        const int*,
        int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<float, float, float>(
        const float*,
        const float*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<double, double, double>(
        const double*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<int, float, float>(
        const int*,
        const float*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<float, int, float>(
        const float*,
        const int*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<int, double, double>(
        const int*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<double, int, double>(
        const double*,
        const int*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<float, double, double>(
        const float*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast<double, float, double>(
        const double*,
        const float*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function matmul_kernel_tiled_contiguous
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT, int TILE>
    __global__ void matmul_kernel_tiled_contiguous(
        const AT* aptr,
        const BT* bptr,
        RT* outptr,
        size_t M,
        size_t K,
        size_t N,
        size_t total_batches)
    {
        __shared__ RT As[TILE][TILE];
        __shared__ RT Bs[TILE][TILE];

        const size_t batch = static_cast<size_t>(blockIdx.z);
        const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
        const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;

        if (batch >= total_batches)
            return;

        const size_t a_batch_base = batch * M * K;
        const size_t b_batch_base = batch * K * N;
        const size_t o_batch_base = batch * M * N;

        RT sum = static_cast<RT>(0);

        const int num_tiles = static_cast<int>((K + TILE - 1) / TILE);

        for (int t = 0; t < num_tiles; ++t) {
            const size_t tiled_k_a = static_cast<size_t>(t) * TILE + threadIdx.x;
            const size_t tiled_k_b = static_cast<size_t>(t) * TILE + threadIdx.y;

            if (row < M && tiled_k_a < K) {
                As[threadIdx.y][threadIdx.x] =
                    static_cast<RT>(aptr[a_batch_base + row * K + tiled_k_a]);
            }
            else {
                As[threadIdx.y][threadIdx.x] = static_cast<RT>(0);
            }

            if (tiled_k_b < K && col < N) {
                Bs[threadIdx.y][threadIdx.x] =
                    static_cast<RT>(bptr[b_batch_base + tiled_k_b * N + col]);
            }
            else {
                Bs[threadIdx.y][threadIdx.x] = static_cast<RT>(0);
            }

            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < TILE; ++kk) {
                sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            outptr[o_batch_base + row * N + col] = sum;
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_matmul_fast
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename BT, typename RT>
    void cuda_matmul_fast2(
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
                "cuda_matmul_fast requires rank >= 2 tensors");
            exit(1);
        }

        const size_t M = a_shape[a_rank - 2];
        const size_t K = a_shape[a_rank - 1];
        const size_t N = b_shape[b_rank - 1];

        if (b_shape[b_rank - 2] != K) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                "cuda_matmul_fast incompatible dimensions for matrix multiplication");
            exit(1);
        }

        // For contiguous tensors, total batch count is the product of all batch dims.
        size_t total_batches = 1;
        for (size_t i = 0; i + 2 < out_shape.size(); ++i) {
            total_batches *= out_shape[i];
        }

        dim3 block(MATMUL_FAST_TILE, MATMUL_FAST_TILE);
        dim3 grid(
            static_cast<unsigned int>((N + MATMUL_FAST_TILE - 1) / MATMUL_FAST_TILE),
            static_cast<unsigned int>((M + MATMUL_FAST_TILE - 1) / MATMUL_FAST_TILE),
            static_cast<unsigned int>(total_batches)
        );

        matmul_kernel_tiled_contiguous<AT, BT, RT, MATMUL_FAST_TILE> << <grid, block >> > (
            aptr,
            bptr,
            outptr,
            M,
            K,
            N,
            total_batches
            );

        check_cuda(cudaGetLastError(), "cuda_matmul_fast kernel launch failed");
        ///check_cuda(cudaDeviceSynchronize(), "cuda_matmul_fast kernel execution failed");
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template void cuda_matmul_fast2<int, int, int>(
        const int*,
        const int*,
        int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<float, float, float>(
        const float*,
        const float*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<double, double, double>(
        const double*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<int, float, float>(
        const int*,
        const float*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<float, int, float>(
        const float*,
        const int*,
        float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<int, double, double>(
        const int*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<double, int, double>(
        const double*,
        const int*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<float, double, double>(
        const float*,
        const double*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);

    template void cuda_matmul_fast2<double, float, double>(
        const double*,
        const float*,
        double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        const std::vector<size_t>&);



    template <typename AT, typename BT, typename RT>
    void cublas_mm(const AT* aptr, const BT* bptr, RT* optr,
        size_t M, size_t K, size_t N)
    {
        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "Error: Could not create cublas handle");

        if constexpr (std::is_same_v<AT, float> &&
            std::is_same_v<BT, float> &&
            std::is_same_v<RT, float>) {
            const float alpha = 1.0f;
            const float beta = 0.0f;

            check_cublas(
                cublasSgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N),   // row-major trick
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    bptr, static_cast<int>(N),
                    aptr, static_cast<int>(K),
                    &beta,
                    optr, static_cast<int>(N)
                ),
                "cublasSgemm failed"
            );
        }
        else if constexpr (std::is_same_v<AT, double> &&
            std::is_same_v<BT, double> &&
            std::is_same_v<RT, double>) {
            const double alpha = 1.0;
            const double beta = 0.0;

            check_cublas(
                cublasDgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N),
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    bptr, static_cast<int>(N),
                    aptr, static_cast<int>(K),
                    &beta,
                    optr, static_cast<int>(N)
                ),
                "cublasDgemm failed"
            );
        }

        check_cublas(cublasDestroy(handle), "Error: Failed to destroy handle");
    }


    //template void cublas_mm<int, int, int>(const int*, const int*, int*, size_t, size_t, size_t);
    //template void cublas_mm<int, float, float>(const int*, const float*, float*, size_t, size_t, size_t);
    //template void cublas_mm<int, double, double>(const int*, const double*, double*, size_t, size_t, size_t);

    //template void cublas_mm<float, int, float>(const float*, const int*, float*, size_t, size_t, size_t);
    template void cublas_mm<float, float, float>(const float*, const float*, float*, size_t, size_t, size_t);
    template void cublas_mm<float, double, double>(const float*, const double*, double*, size_t, size_t, size_t);

    //template void cublas_mm<double, int, double>(const double*, const int*, double*, size_t, size_t, size_t);
    template void cublas_mm<double, float, double>(const double*, const float*, double*, size_t, size_t, size_t);
    template void cublas_mm<double, double, double>(const double*, const double*, double*, size_t, size_t, size_t);


    cublasHandle_t get_cublas_handle() {
        static cublasHandle_t handle = [] {
            cublasHandle_t h;
            check_cublas(cublasCreate(&h), "Error: Could not create cublas handle");
            return h;
        }();
        return handle;
    }

    size_t product(const std::vector<size_t>& v) {
        size_t out = 1;
        for (size_t x : v) out *= x;
        return out;
    }

    bool all_ones(const std::vector<size_t>& v) {
        for (size_t x : v) {
            if (x != 1) return false;
        }
        return true;
    }



    template <typename AT, typename BT, typename RT>
    void cublas_mm_row_major(const AT* aptr, const BT* bptr, RT* optr,
        size_t M, size_t K, size_t N)
    {
        auto handle = get_cublas_handle();

        if constexpr (std::is_same_v<AT, float> &&
            std::is_same_v<BT, float> &&
            std::is_same_v<RT, float>)
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;

            check_cublas(
                cublasSgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N),   // swapped for row-major
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    bptr, static_cast<int>(N),
                    aptr, static_cast<int>(K),
                    &beta,
                    optr, static_cast<int>(N)
                ),
                "cublasSgemm failed"
            );
        }
        else if constexpr (std::is_same_v<AT, double> &&
            std::is_same_v<BT, double> &&
            std::is_same_v<RT, double>)
        {
            const double alpha = 1.0;
            const double beta = 0.0;

            check_cublas(
                cublasDgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N),   // swapped for row-major
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    bptr, static_cast<int>(N),
                    aptr, static_cast<int>(K),
                    &beta,
                    optr, static_cast<int>(N)
                ),
                "cublasDgemm failed"
            );
        }

        //check_cuda(cudaDeviceSynchronize(), "cublas_mm_row_major sync failed");
    }

    template <typename AT, typename BT, typename RT>
    void cublas_mm_strided_batched_row_major(
        const AT* aptr,
        const BT* bptr,
        RT* optr,
        size_t M,
        size_t K,
        size_t N,
        long long strideA,
        long long strideB,
        long long strideC,
        int batch_count)
    {
        auto handle = get_cublas_handle();

        if constexpr (std::is_same_v<AT, float> &&
            std::is_same_v<BT, float> &&
            std::is_same_v<RT, float>)
        {
            const float alpha = 1.0f;
            const float beta = 0.0f;

            check_cublas(
                cublasSgemmStridedBatched(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N),   // swapped for row-major
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    bptr, static_cast<int>(N), strideB,
                    aptr, static_cast<int>(K), strideA,
                    &beta,
                    optr, static_cast<int>(N), strideC,
                    batch_count
                ),
                "cublasSgemmStridedBatched failed"
            );
        }
        else if constexpr (std::is_same_v<AT, double> &&
            std::is_same_v<BT, double> &&
            std::is_same_v<RT, double>)
        {
            const double alpha = 1.0;
            const double beta = 0.0;

            check_cublas(
                cublasDgemmStridedBatched(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N),   // swapped for row-major
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    bptr, static_cast<int>(N), strideB,
                    aptr, static_cast<int>(K), strideA,
                    &beta,
                    optr, static_cast<int>(N), strideC,
                    batch_count
                ),
                "cublasDgemmStridedBatched failed"
            );
        }

        //check_cuda(cudaDeviceSynchronize(), "cublas_mm_strided_batched_row_major sync failed");
    }

    bool can_use_strided_batched_fastpath(
        const std::vector<size_t>& a_batch_shape,
        const std::vector<size_t>& b_batch_shape,
        const std::vector<size_t>& out_batch_shape)
    {
        // Exact same batch shape: perfect fit for strided batched GEMM.
        if (a_batch_shape == b_batch_shape && a_batch_shape == out_batch_shape) {
            return true;
        }

        // One side is a single matrix reused across all batches.
        if (all_ones(a_batch_shape) && b_batch_shape == out_batch_shape) {
            return true;
        }

        if (all_ones(b_batch_shape) && a_batch_shape == out_batch_shape) {
            return true;
        }

        // Mixed broadcasting like [B,1] x [1,H] is not representable
        // with one constant stride per operand.
        return false;
    }


    template void cublas_mm_strided_batched_row_major<float, float, float>(const float*, const float*, float*, size_t, size_t, size_t, long long, long long, long long, int );
    template void cublas_mm_strided_batched_row_major<double, double, double>(const double*, const double*, double*, size_t, size_t, size_t, long long, long long, long long, int);

    template void cublas_mm_row_major<float,float,float>(const float*, const float*, float*, size_t, size_t, size_t);
    template void cublas_mm_row_major<double, double, double>(const double*, const double*, double*, size_t, size_t, size_t);
}


    

    