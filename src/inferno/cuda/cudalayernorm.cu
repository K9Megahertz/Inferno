#include "cudaops.h"
#include "cudamath.cuh"


namespace Inferno {  

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Kernel
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT>
    __global__ void layernorm_forward_kernel(
        const AT* __restrict__ iptr,
        AT* __restrict__ optr,
        const float* __restrict__ gptr,
        const float* __restrict__ bptr,
        float* __restrict__ meanptr,
        float* __restrict__ invstdptr,
        size_t dim,
        float eps
    ) {
        const size_t batch = static_cast<size_t>(blockIdx.x);
        const size_t tid = static_cast<size_t>(threadIdx.x);

        extern __shared__ float sdata[];
        float* s_sum = sdata;
        float* s_sqsum = sdata + blockDim.x;

        const size_t base = batch * dim;

        // ---------------------------------------------------------------------------------------------------
        // Step 1: compute mean
        // ---------------------------------------------------------------------------------------------------
        float local_sum = 0.0f;
        for (size_t i = tid; i < dim; i += blockDim.x) {
            local_sum += static_cast<float>(iptr[base + i]);
        }

        s_sum[tid] = local_sum;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_sum[tid] += s_sum[tid + stride];
            }
            __syncthreads();
        }

        const float mean = s_sum[0] / static_cast<float>(dim);

        // ---------------------------------------------------------------------------------------------------
        // Step 2: compute variance
        // ---------------------------------------------------------------------------------------------------
        float local_sqsum = 0.0f;
        for (size_t i = tid; i < dim; i += blockDim.x) {
            float x = static_cast<float>(iptr[base + i]);
            float d = x - mean;
            local_sqsum += d * d;
        }

        s_sqsum[tid] = local_sqsum;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_sqsum[tid] += s_sqsum[tid + stride];
            }
            __syncthreads();
        }

        const float var = s_sqsum[0] / static_cast<float>(dim);
        const float invstd = rsqrtf(var + eps);

        // save stats for backward
        if (tid == 0) {
            meanptr[batch] = mean;
            invstdptr[batch] = invstd;
        }

        // ---------------------------------------------------------------------------------------------------
        // Step 3: normalize + affine
        // ---------------------------------------------------------------------------------------------------
        for (size_t i = tid; i < dim; i += blockDim.x) {
            float x = static_cast<float>(iptr[base + i]);
            float xhat = (x - mean) * invstd;
            float y = xhat * gptr[i] + bptr[i];
            optr[base + i] = static_cast<AT>(y);
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_layer_normalization
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT>
    void cuda_layer_normalization(
        const AT* iptr,
        AT* optr,
        const float* gptr,
        const float* bptr,
        float* meanptr,
        float* invstdptr,
        size_t num_batches,
        size_t dim,
        float eps
    ) {
        if (num_batches == 0 || dim == 0)
            return;

        // Pick a reasonable thread count.
        // 256 is a good default for this kind of row-wise reduction.
        int threads = 256;

        // No need to launch more threads than useful for tiny dims.
        if (dim < static_cast<size_t>(threads)) {
            threads = 1;
            while (threads < static_cast<int>(dim) && threads < 256)
                threads <<= 1;
        }

        dim3 block(threads);
        dim3 grid(static_cast<unsigned int>(num_batches));

        // shared memory = two float arrays of blockDim.x
        size_t shmem_bytes = 2 * threads * sizeof(float);

        layernorm_forward_kernel<AT> << <grid, block, shmem_bytes >> > (
            iptr,
            optr,
            gptr,
            bptr,
            meanptr,
            invstdptr,
            dim,
            eps
            );

        check_cuda(cudaGetLastError(), "layernorm_forward_kernel launch failed");
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template void cuda_layer_normalization<float>(
        const float* iptr,
        float* optr,
        const float* gptr,
        const float* bptr,
        float* meanptr,
        float* invstdptr,
        size_t num_batches,
        size_t dim,
        float eps
    );

    template void cuda_layer_normalization<double>(
        const double* iptr,
        double* optr,
        const float* gptr,
        const float* bptr,
        float* meanptr,
        float* invstdptr,
        size_t num_batches,
        size_t dim,
        float eps
    );

    template void cuda_layer_normalization<int>(
        const int* iptr,
        int* optr,
        const float* gptr,
        const float* bptr,
        float* meanptr,
        float* invstdptr,
        size_t num_batches,
        size_t dim,
        float eps
    );



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Kernel: input gradient
    //
    //  Computes:
    //      dxhat_i = gout_i * gamma_i
    //      dx_i = (invstd / dim) * (dim * dxhat_i - sum(dxhat) - xhat_i * sum(dxhat * xhat))
    //
    //  One block handles one batch row.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename GT>
    __global__ void layernorm_input_grad_kernel(
        const AT* __restrict__ aptr,
        const AT* __restrict__ goutptr,
        const float* __restrict__ gptr,
        const float* __restrict__ meanptr,
        const float* __restrict__ invstdptr,
        GT* __restrict__ gaptr,
        size_t dim
    ) {
        const size_t b = static_cast<size_t>(blockIdx.x);
        const size_t tid = static_cast<size_t>(threadIdx.x);
        const size_t base = b * dim;

        const float mean = meanptr[b];
        const float invstd = invstdptr[b];

        extern __shared__ float sdata[];
        float* s_sum_dxhat = sdata;
        float* s_sum_dxhat_xhat = sdata + blockDim.x;

        float local_sum_dxhat = 0.0f;
        float local_sum_dxhat_xhat = 0.0f;

        // First pass: compute reductions
        for (size_t i = tid; i < dim; i += blockDim.x) {
            const float x = static_cast<float>(aptr[base + i]);
            const float gout = static_cast<float>(goutptr[base + i]);
            const float gamma = gptr[i];

            const float xhat = (x - mean) * invstd;
            const float dxhat = gout * gamma;

            local_sum_dxhat += dxhat;
            local_sum_dxhat_xhat += dxhat * xhat;
        }

        s_sum_dxhat[tid] = local_sum_dxhat;
        s_sum_dxhat_xhat[tid] = local_sum_dxhat_xhat;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_sum_dxhat[tid] += s_sum_dxhat[tid + stride];
                s_sum_dxhat_xhat[tid] += s_sum_dxhat_xhat[tid + stride];
            }
            __syncthreads();
        }

        const float sum_dxhat = s_sum_dxhat[0];
        const float sum_dxhat_xhat = s_sum_dxhat_xhat[0];
        const float dim_f = static_cast<float>(dim);

        // Second pass: compute input gradient
        for (size_t i = tid; i < dim; i += blockDim.x) {
            const float x = static_cast<float>(aptr[base + i]);
            const float gout = static_cast<float>(goutptr[base + i]);
            const float gamma = gptr[i];

            const float xhat = (x - mean) * invstd;
            const float dxhat = gout * gamma;

            const float gx =
                (invstd / dim_f) *
                (dim_f * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);

            gaptr[base + i] = static_cast<GT>(gx);
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Kernel: gamma and beta gradients
    //
    //  Computes:
    //      dBeta_i  = sum_b(gout[b, i])
    //      dGamma_i = sum_b(gout[b, i] * xhat[b, i])
    //
    //  One thread handles one feature index i.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT>
    __global__ void layernorm_param_grad_kernel(
        const AT* __restrict__ aptr,
        const AT* __restrict__ goutptr,
        const float* __restrict__ meanptr,
        const float* __restrict__ invstdptr,
        float* __restrict__ ggptr,
        float* __restrict__ gbptr,
        size_t num_batches,
        size_t dim
    ) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
        if (i >= dim)
            return;

        float ggamma = 0.0f;
        float gbeta = 0.0f;

        for (size_t b = 0; b < num_batches; ++b) {
            const size_t idx = b * dim + i;

            const float x = static_cast<float>(aptr[idx]);
            const float gout = static_cast<float>(goutptr[idx]);
            const float mean = meanptr[b];
            const float invstd = invstdptr[b];

            const float xhat = (x - mean) * invstd;

            gbeta += gout;
            ggamma += gout * xhat;
        }

        ggptr[i] = ggamma;
        gbptr[i] = gbeta;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_layernorm_backward
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename GT>
    void cuda_layernorm_backward(
        const AT* aptr,
        const AT* goutptr,
        const float* gptr,
        const float* meanptr,
        const float* invstdptr,
        GT* gaptr,
        float* ggptr,
        float* gbptr,
        size_t num_batches,
        size_t dim
    ) {
        if (num_batches == 0 || dim == 0)
            return;

        // -----------------------------------------------
        // Kernel 1: input gradient
        // -----------------------------------------------
        int threads_x = 256;
        if (dim < static_cast<size_t>(threads_x)) {
            threads_x = 1;
            while (threads_x < static_cast<int>(dim) && threads_x < 256)
                threads_x <<= 1;
        }

        dim3 block_x(threads_x);
        dim3 grid_x(static_cast<unsigned int>(num_batches));

        size_t shmem_bytes = 2 * threads_x * sizeof(float);

        layernorm_input_grad_kernel<AT, GT> << <grid_x, block_x, shmem_bytes >> > (
            aptr,
            goutptr,
            gptr,
            meanptr,
            invstdptr,
            gaptr,
            dim
            );

        check_cuda(cudaGetLastError(), "layernorm_input_grad_kernel launch failed");

        // -----------------------------------------------
        // Kernel 2: gamma and beta gradients
        // -----------------------------------------------
        const int threads_p = 256;
        const int blocks_p = static_cast<int>((dim + threads_p - 1) / threads_p);

        layernorm_param_grad_kernel<AT> << <blocks_p, threads_p >> > (
            aptr,
            goutptr,
            meanptr,
            invstdptr,
            ggptr,
            gbptr,
            num_batches,
            dim
            );

        check_cuda(cudaGetLastError(), "layernorm_param_grad_kernel launch failed");
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit instantiations
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template void cuda_layernorm_backward<float, float>(
        const float* aptr,
        const float* goutptr,
        const float* gptr,
        const float* meanptr,
        const float* invstdptr,
        float* gaptr,
        float* ggptr,
        float* gbptr,
        size_t num_batches,
        size_t dim
    );    

    template void cuda_layernorm_backward<double, double>(
        const double* aptr,
        const double* goutptr,
        const float* gptr,
        const float* meanptr,
        const float* invstdptr,
        double* gaptr,
        float* ggptr,
        float* gbptr,
        size_t num_batches,
        size_t dim
    );

    template void cuda_layernorm_backward<int, float>(
        const int* aptr,
        const int* goutptr,
        const float* gptr,
        const float* meanptr,
        const float* invstdptr,
        float* gaptr,
        float* ggptr,
        float* gbptr,
        size_t num_batches,
        size_t dim
    );



}






















//old shit



    /*//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_kernel_calc_mean
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename AT>
    __global__ void cuda_kernel_calc_mean(const AT* iptr, AT* optr, size_t num_batches, size_t dim) {


        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (idx < num_batches) {
            size_t base = idx * dim;
            AT mean = 0;
            for (size_t curr_pos = 0; curr_pos < dim; curr_pos++) {
                mean += iptr[base + curr_pos];
            }
            mean /= dim;

            optr[idx] = mean;
        }
        

    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_kernel_calc_stddev
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename AT>
    __global__ void cuda_kernel_calc_stddev(const AT* iptr, const AT* mptr, AT* optr, size_t num_batches, size_t dim) {


        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (idx < num_batches) {
            //get variance
            size_t base = idx * dim;            
            AT var = 0;
            for (size_t curr_pos = 0; curr_pos < dim; curr_pos++) {
                AT val = iptr[base + curr_pos];
                var += (val - mptr[idx]) * (val - mptr[idx]);
            }
            var /= dim;


            //get stddev
            AT stddev = std::sqrt(var);

        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_kernel_calc_full
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename AT>
    __global__ void cuda_kernel_calc_full(const AT* iptr, AT* optr, size_t N, size_t ndim) {


        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (idx < N) {
    
            size_t row = idx / ndim;
            size_t base = row * ndim;

            //get mean
            AT mean = 0;
            for (size_t curr_pos = 0; curr_pos < ndim; curr_pos++) {
                mean += iptr[base + curr_pos];
            }
            mean /= ndim;


            //get variance            
            AT var = 0;
            for (size_t curr_pos = 0; curr_pos < ndim; curr_pos++) {
                AT val = iptr[base + curr_pos];
                var += (val - mean) * (val - mean);
            }
            var /= ndim;

            //get stddev
            AT stddev = cuda_sqrt(var);

            optr[idx] = (iptr[idx] - mean) / stddev;

        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_layer_normalization
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    template <typename AT>
    void cuda_layer_normalization(const AT* iptr, AT* optr, float* gptr, float* bptr, size_t num_batches, size_t dim) {

        constexpr int threads = 256;
        int blocks = static_cast<int>((num_batches * dim + threads - 1) / threads);
        cuda_kernel_calc_full << <blocks, threads >> > (iptr, optr, num_batches * dim, dim);

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

    template void cuda_layer_normalization<int>(const int*, int*, float*, float*, size_t, size_t);
    template void cuda_layer_normalization<float>(const float*, float*, float*, float*, size_t, size_t);
    template void cuda_layer_normalization<double>(const double*, double*, float*, float*, size_t, size_t);*/

