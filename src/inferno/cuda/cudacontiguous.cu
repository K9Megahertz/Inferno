#include "cudaops.h"

namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function contiguous_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T>
    __global__ void contiguous_kernel(
        const T* __restrict__ aptr,
        T* __restrict__ optr,
        const size_t* __restrict__ shape,
        const size_t* __restrict__ strides,
        size_t rank,
        size_t offset,
        size_t N)
    {
        size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
        if (linear >= N) return;

        // Convert contiguous output linear index -> multi-index -> source offset
        size_t tmp = linear;
        size_t src_index = offset;

        // Unravel from last dim to first dim
        for (int dim = static_cast<int>(rank) - 1; dim >= 0; --dim) {
            size_t idx = tmp % shape[dim];
            tmp /= shape[dim];
            src_index += idx * strides[dim];
        }

        optr[linear] = aptr[src_index];
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_contiguous_copy
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT>
    void cuda_contiguous_copy(
        const AT* aptr,
        AT* optr,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides,
        size_t offset,
        size_t N)
    {
        if (shape.size() != strides.size()) {            
            INFERNO_LOG_ERROR() << "cuda_contiguous_copy: shape and strides rank mismatch" << std::endl;
            exit(1);
        }

        if (N == 0) {
            return;
        }

        const size_t rank = shape.size();

        size_t* d_shape = nullptr;
        size_t* d_strides = nullptr;

        cudaError_t err;

        err = cudaMalloc(&d_shape, rank * sizeof(size_t));
        if (err != cudaSuccess) {            
            INFERNO_LOG_ERROR() << "cuda_contiguous_copy: cudaMalloc failed for d_shape" << std::endl;
            exit(1);
        }

        err = cudaMalloc(&d_strides, rank * sizeof(size_t));
        if (err != cudaSuccess) {            
            check_cuda(cudaFree(d_shape), "cuda_contiguous_copy failed to cudaFree");            
            INFERNO_LOG_ERROR() << "cuda_contiguous_copy: cudaMalloc failed for d_strides" << std::endl;
            exit(1);
        }

        err = cudaMemcpy(d_shape, shape.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            
            check_cuda(cudaFree(d_shape), "cuda_contiguous_copy failed to cudaFree");            
            check_cuda(cudaFree(d_strides), "cuda_contiguous_copy failed to cudaFree");            
            INFERNO_LOG_ERROR() << "cuda_contiguous_copy: cudaMemcpy failed for d_shape" << std::endl;
            exit(1);
        }

        err = cudaMemcpy(d_strides, strides.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {            
            check_cuda(cudaFree(d_shape), "cuda_contiguous_copy failed to cudaFree");            
            check_cuda(cudaFree(d_strides), "cuda_contiguous_copy failed to cudaFree");            
            INFERNO_LOG_ERROR() << "cuda_contiguous_copy: cudaMemcpy failed for d_strides" << std::endl;
        }

        constexpr int threads = 256;
        const int blocks = static_cast<int>((N + threads - 1) / threads);

        contiguous_kernel<AT> << <blocks, threads >> > (
            aptr,
            optr,
            d_shape,
            d_strides,
            rank,
            offset,
            N
            );

        err = cudaGetLastError();
        if (err != cudaSuccess) {            
            check_cuda(cudaFree(d_shape), "cuda_contiguous_copy failed to cudaFree");
            check_cuda(cudaFree(d_strides), "cuda_contiguous_copy failed to cudaFree");            
            INFERNO_LOG_ERROR() << "cuda_contiguous_copy: kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        /*err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {            
            check_cuda(cudaFree(d_shape), "cuda_contiguous_copy failed to cudaFree");
            check_cuda(cudaFree(d_strides), "cuda_contiguous_copy failed to cudaFree");
            INFERNO_LOG_ERROR() << "cuda_contiguous_copy: kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }*/
        
        check_cuda(cudaFree(d_shape), "cuda_contiguous_copy failed to cudaFree");        
        check_cuda(cudaFree(d_strides), "cuda_contiguous_copy failed to cudaFree");
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


    template void cuda_contiguous_copy<int>(
        const int*, int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t, size_t);

    template void cuda_contiguous_copy<float>(
        const float*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t, size_t);

    template void cuda_contiguous_copy<double>(
        const double*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t, size_t);

  
}