#include "cudaops.h"

namespace Inferno {


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_cross_entropy_loss_fused_kernel
    //  My version
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename LT>
    __global__ void cuda_cross_entropy_loss_fused_kernel(const LT* logitsptr, const int* targetsptr, LT* optr, size_t rows, size_t vocab_size) {


        __shared__ LT ShWarpmaxes[8];
        __shared__ LT ShWarpSums[8];
        __shared__ LT ShMax[1];
        __shared__ LT ShFinalSum[1];


        int row = blockIdx.x;

        if (row >= rows)
            return;

        int target = targetsptr[row];
        if (target < 0 || target >= vocab_size)
            return;

        const LT* rowptr = logitsptr + row * vocab_size;

        int tid = threadIdx.x;

        int warp = tid / 32;
        int lane = tid % 32;

        LT local_max = -INFINITY;
        LT local_sum = LT(0);

        //each thread gets a strided max across the 60259 columns
        // e.g. thread 0 compares idx 0, 256, 512, 784, 1024, etc...
        // e.g. thread 1 compares idx 1, 257, 513, 785, 1025, etc...
        // etc...
        for (int k = tid; k < vocab_size; k += blockDim.x) {

            //need to update local_max
            //need to update local_sum
            LT val = rowptr[k];
            LT next_max = fmax(val, local_max);
            local_sum = local_sum * exp(local_max - next_max) + exp(val - next_max);
            local_max = next_max;

        }


        LT global_sum = local_sum;
        LT global_max = local_max;
        //each warp will compare the maxes that each of the 32 threads found
        //all 8 warps will do this
        for (int i = 16; i > 0; i >>= 1) {
            LT remote_max = __shfl_down_sync(0xffffffff, global_max, i);
            LT remote_sum = __shfl_down_sync(0xffffffff, global_sum, i);
            LT next_max = fmax(remote_max, global_max);
            global_sum = global_sum * exp(global_max - next_max) + remote_sum * exp(remote_max - next_max);
            global_max = next_max;
        }

        //save in shared memory
        if (lane == 0) {
            ShWarpmaxes[warp] = global_max;
            ShWarpSums[warp] = global_sum;
        }

        __syncthreads();


        if (warp == 0) {

            LT final_max = (tid < 8) ? ShWarpmaxes[tid] : -INFINITY;
            LT final_sum = (tid < 8) ? ShWarpSums[tid] : LT(0);
            if (tid < 8) {
                for (int i = 4; i > 0; i >>= 1) {
                    LT remote_max = __shfl_down_sync(0x000000ff, final_max, i);
                    LT remote_sum = __shfl_down_sync(0x000000ff, final_sum, i);
                    LT next_max = fmax(remote_max, final_max);
                    final_sum = final_sum * exp(final_max - next_max) + remote_sum * exp(remote_max - next_max);
                    final_max = next_max;
                }
            }
            if (tid == 0) {
                ShMax[0] = final_max;
                ShFinalSum[0] = final_sum;
            }
        }

        __syncthreads();


        if (tid == 0) {

            LT row_loss = LT(0);

            LT target_logit = rowptr[target];
            row_loss = -(target_logit - ShMax[0] - log(ShFinalSum[0]));

            LT normalized_loss = row_loss / static_cast<LT>(rows);
            atomicAdd(optr, normalized_loss);
        }


    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_cross_entropy_loss_backward_fused_kernel
    //  My implementation
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename LT>
    __global__ void cuda_cross_entropy_loss_backward_fused_kernel(const LT* logitsptr, const int* targetsptr, const LT* upstreamptr, LT* gradlogitsptr, size_t rows, size_t vocab_size) {


        __shared__ LT ShWarpmaxes[8];
        __shared__ LT ShWarpSums[8];
        __shared__ LT ShMax[1];
        __shared__ LT ShFinalSum[1];


        int row = blockIdx.x;

        if (row >= rows)
            return;

        int target = targetsptr[row];
        if (target < 0 || target >= vocab_size)
            return;

        const LT* rowptr = logitsptr + row * vocab_size;

        int tid = threadIdx.x;

        int warp = tid / 32;
        int lane = tid % 32;

        LT local_max = -INFINITY;
        LT local_sum = LT(0);

        //each thread gets a strided max across the 60259 columns
        // e.g. thread 0 compares idx 0, 256, 512, 784, 1024, etc...
        // e.g. thread 1 compares idx 1, 257, 513, 785, 1025, etc...
        // etc...
        for (int k = tid; k < vocab_size; k += blockDim.x) {

            //need to update local_max
            //need to update local_sum
            LT val = rowptr[k];
            LT next_max = fmax(val, local_max);
            local_sum = local_sum * exp(local_max - next_max) + exp(val - next_max);
            local_max = next_max;

        }


        LT global_sum = local_sum;
        LT global_max = local_max;
        //each warp will compare the maxes that each of the 32 threads found
        //all 8 warps will do this
        for (int i = 16; i > 0; i >>= 1) {
            LT remote_max = __shfl_down_sync(0xffffffff, global_max, i);
            LT remote_sum = __shfl_down_sync(0xffffffff, global_sum, i);
            LT next_max = fmax(remote_max, global_max);
            global_sum = global_sum * exp(global_max - next_max) + remote_sum * exp(remote_max - next_max);
            global_max = next_max;
        }

        //save in shared memory
        if (lane == 0) {
            ShWarpmaxes[warp] = global_max;
            ShWarpSums[warp] = global_sum;
        }

        __syncthreads();


        if (warp == 0) {

            LT final_max = (tid < 8) ? ShWarpmaxes[tid] : -INFINITY;
            LT final_sum = (tid < 8) ? ShWarpSums[tid] : LT(0);

            for (int i = 4; i > 0; i >>= 1) {
                LT remote_max = __shfl_down_sync(0xffffffff, final_max, i);
                LT remote_sum = __shfl_down_sync(0xffffffff, final_sum, i);
                LT next_max = fmax(remote_max, final_max);
                final_sum = final_sum * exp(final_max - next_max) + remote_sum * exp(remote_max - next_max);
                final_max = next_max;
            }

            if (tid == 0) {
                ShMax[0] = final_max;
                ShFinalSum[0] = final_sum;
            }
        }


        __syncthreads(); // Ensure ShMax and ShFinalSum are fully visible

        // Compute the master scale multiplier using the pointer value
        LT scale = upstreamptr[0] / static_cast<LT>(rows);

        // Set up your destination row writer pointer matching your logits layout
        LT* grad_rowptr = gradlogitsptr + row * vocab_size;


        for (int k = tid; k < vocab_size; k += blockDim.x) {

            //need to update local_max
            //need to update local_sum
            LT val = rowptr[k];
            LT prob = exp(val - ShMax[0]) / ShFinalSum[0];

            LT indicator = (k == target) ? LT(1) : LT(0);
            grad_rowptr[k] = (prob - indicator) * scale;

        }

        

    }





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cross_entropy_loss_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    __global__ void cross_entropy_loss_kernel(const LT* logits,const int* targets,LT* out,size_t rows,size_t vocab_size) {
        size_t r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows) return;

        const LT* row_ptr = logits + (r * vocab_size);
        int target_id = targets[r];

        if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
            return;
        }

        LT max_logit = row_ptr[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row_ptr[v] > max_logit) {
                max_logit = row_ptr[v];
            }
        }

        LT sum_exp = static_cast<LT>(0);
        for (size_t v = 0; v < vocab_size; v++) {
            sum_exp += exp(row_ptr[v] - max_logit);
        }

        LT log_sum_exp = log(sum_exp);
        LT target_logit = row_ptr[target_id];
        LT row_loss = -(target_logit - max_logit - log_sum_exp);

        atomicAdd(out, row_loss / static_cast<LT>(rows));
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_cross_entropy_loss
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    void cuda_cross_entropy_loss(const LT* logits, const int* targets, LT* out, size_t rows, size_t vocab_size) {

        cudaMemset(out, 0, sizeof(LT));

        const int threads = 256;
        //const int blocks = static_cast<int>((rows + threads - 1) / threads);
        const int blocks = static_cast<int>(rows);
        
        cuda_cross_entropy_loss_fused_kernel<LT> << <blocks, threads >> > (logits, targets, out, rows, vocab_size);

        check_cuda(cudaGetLastError(), "CUDA kernel launch error in cuda_cross_entropy_loss");
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

    
    template void cuda_cross_entropy_loss<float>(const float*, const int*, float*, size_t, size_t);
    template void cuda_cross_entropy_loss<double>(const double*, const int*, double*, size_t, size_t);








    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cross_entropy_loss_backward_kernel
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    __global__ void cross_entropy_loss_backward_kernel(const LT* logits,const int* targets,const LT* upstream,LT* grad_logits,size_t rows,size_t vocab_size) {
        size_t r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= rows) return;

        const LT* row_ptr = logits + (r * vocab_size);
        LT* grad_row = grad_logits + (r * vocab_size);
        int target_id = targets[r];

        if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
            return;
        }

        LT max_logit = row_ptr[0];
        for (size_t v = 1; v < vocab_size; v++) {
            if (row_ptr[v] > max_logit) {
                max_logit = row_ptr[v];
            }
        }

        LT sum_exp = static_cast<LT>(0);
        for (size_t v = 0; v < vocab_size; v++) {
            sum_exp += exp(row_ptr[v] - max_logit);
        }

        LT scale = upstream[0] / static_cast<LT>(rows);

        for (size_t v = 0; v < vocab_size; v++) {
            LT prob = exp(row_ptr[v] - max_logit) / sum_exp;
            grad_row[v] = prob * scale;
        }

        grad_row[target_id] -= scale;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cuda_cross_entropy_loss_backward
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename LT>
    void cuda_cross_entropy_loss_backward(const LT* logits, const int* targets, const LT* upstream, LT* grad_logits, size_t rows, size_t vocab_size) {
        const int threads = 256;
        const int blocks = static_cast<int>((rows + threads - 1) / threads);        

        cross_entropy_loss_backward_kernel<LT> << <blocks, threads >> > (logits, targets, upstream, grad_logits, rows, vocab_size);        

        check_cuda(cudaGetLastError(), "CUDA kernel launch error in cuda_cross_entropy_loss_backward");
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

    
    template void cuda_cross_entropy_loss_backward<float>(const float*, const int*, const float*, float*, size_t, size_t);
    template void cuda_cross_entropy_loss_backward<double>(const double*, const int*, const double*, double*, size_t, size_t);




    template<typename LT, int BLOCK_SIZE>
    __global__ void cross_entropy_loss_backward_kernel_fast(
        const LT* logits,
        const int* targets,
        const LT* upstream,
        LT* grad_logits,
        size_t rows,
        size_t vocab_size
    ) {
        size_t r = blockIdx.x;
        if (r >= rows) return;

        const LT* row_ptr = logits + r * vocab_size;
        LT* grad_row = grad_logits + r * vocab_size;

        int tid = threadIdx.x;
        int target_id = targets[r];

        if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
            return;
        }

        __shared__ LT sdata[BLOCK_SIZE];

        // 1. parallel max
        LT local_max = -INFINITY;

        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            LT x = row_ptr[v];
            if (x > local_max) local_max = x;
        }

        sdata[tid] = local_max;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                LT other = sdata[tid + stride];
                if (other > sdata[tid]) sdata[tid] = other;
            }
            __syncthreads();
        }

        LT max_logit = sdata[0];

        // 2. parallel sum exp
        LT local_sum = static_cast<LT>(0);

        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            //local_sum += exp(row_ptr[v] - max_logit);
            local_sum += __expf(static_cast<float>(row_ptr[v] - max_logit));
        }

        sdata[tid] = local_sum;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        LT sum_exp = sdata[0];
        LT scale = upstream[0] / static_cast<LT>(rows);

        // 3. parallel write grad
        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            //LT prob = exp(row_ptr[v] - max_logit) / sum_exp;
            LT prob = __expf(static_cast<float>(row_ptr[v] - max_logit)) / sum_exp;
            LT g = prob * scale;

            if (static_cast<int>(v) == target_id) {
                g -= scale;
            }

            grad_row[v] = g;
        }
    }

    // Drop-in replacement for cross_entropy_loss_backward_kernel_fast /
    // cuda_cross_entropy_loss_backward_fast.
    //
    // Changes vs. the original:
    //   1. exp() -> __expf()                         (fast approximate intrinsic)
    //   2. max-pass + sum-pass MERGED into a single online-softmax pass
    //        -> only 2 full reads of `logits` total (stats pass + write pass),
    //           down from 3 in the original (max pass, sum pass, write pass)
    //   3. Block reduction uses warp shuffle (__shfl_xor_sync) for the intra-warp
    //      part, with only ONE shared-memory round-trip + __syncthreads() to
    //      combine the per-warp partial results, instead of log2(BLOCK_SIZE)
    //      rounds of shared-memory tree reduction with a sync every round.
    //
    // Requires BLOCK_SIZE to be a multiple of 32 (warp size). Same as before,
    // one block per row.

    template <typename LT, int BLOCK_SIZE>
    __global__ void cross_entropy_loss_backward_kernel_fast_v2(
        const LT* logits,
        const int* targets,
        const LT* upstream,
        LT* grad_logits,
        size_t rows,
        size_t vocab_size)
    {
        static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
        constexpr int NUM_WARPS = BLOCK_SIZE / 32;

        size_t r = blockIdx.x;
        if (r >= rows) return;

        const LT* row_ptr = logits + r * vocab_size;
        LT* grad_row = grad_logits + r * vocab_size;

        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp = tid / 32;

        int target_id = targets[r];
        if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
            return;
        }

        // -------------------------------------------------------------------
        // Pass 1 (merged): online max + sum-of-exp in a single sweep over the
        // row. Each thread strides across the vocab, maintaining a running
        // (local_m, local_l) pair using the standard online-softmax update.
        // -------------------------------------------------------------------
        float local_m = -INFINITY;
        float local_l = 0.0f;

        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            float x = static_cast<float>(row_ptr[v]);
            float new_m = fmaxf(local_m, x);
            local_l = local_l * __expf(local_m - new_m) + __expf(x - new_m);
            local_m = new_m;
        }

        // ---- intra-warp reduction via shuffle (no shared memory, no sync) ----
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_m = __shfl_xor_sync(0xffffffff, local_m, offset);
            float other_l = __shfl_xor_sync(0xffffffff, local_l, offset);
            float new_m = fmaxf(local_m, other_m);
            float new_l = local_l * __expf(local_m - new_m) + other_l * __expf(other_m - new_m);
            local_m = new_m;
            local_l = new_l;
        }
        // every lane in the warp now holds this warp's combined (local_m, local_l)

        // ---- combine the NUM_WARPS per-warp results via shared memory ----
        __shared__ float warp_m[NUM_WARPS];
        __shared__ float warp_l[NUM_WARPS];

        if (lane == 0) {
            warp_m[warp] = local_m;
            warp_l[warp] = local_l;
        }
        __syncthreads();

        // single thread finishes the small (NUM_WARPS-way) reduction and
        // broadcasts via shared memory -- NUM_WARPS is small (e.g. 8 for
        // BLOCK_SIZE=256), so a simple serial loop here is cheap and avoids
        // another round of shuffles/syncs for a handful of elements.
        __shared__ float row_max_shared;
        __shared__ float row_sum_shared;

        if (tid == 0) {
            float m = warp_m[0];
            float l = warp_l[0];
#pragma unroll
            for (int i = 1; i < NUM_WARPS; i++) {
                float new_m = fmaxf(m, warp_m[i]);
                float new_l = l * __expf(m - new_m) + warp_l[i] * __expf(warp_m[i] - new_m);
                m = new_m;
                l = new_l;
            }
            row_max_shared = m;
            row_sum_shared = l;
        }
        __syncthreads();

        float max_logit = row_max_shared;
        float sum_exp = row_sum_shared;
        LT scale = upstream[0] / static_cast<LT>(rows);

        // -------------------------------------------------------------------
        // Pass 2: write gradients. This is the row's SECOND full read of
        // `logits` (down from the original's third pass) -- unavoidable,
        // since caching all `vocab_size` exponentiated values in shared
        // memory isn't viable at typical vocab sizes (e.g. 50257 floats
        // would be ~200KB, over the shared memory budget per block).
        // -------------------------------------------------------------------
        for (size_t v = tid; v < vocab_size; v += BLOCK_SIZE) {
            float x = static_cast<float>(row_ptr[v]);
            float prob = __expf(x - max_logit) / sum_exp;
            LT g = static_cast<LT>(prob) * scale;
            if (static_cast<int>(v) == target_id) {
                g -= scale;
            }
            grad_row[v] = g;
        }
    }

   

    template<typename LT>
    void cuda_cross_entropy_loss_backward_fast(
        const LT* logits,
        const int* targets,
        const LT* upstream,
        LT* grad_logits,
        size_t rows,
        size_t vocab_size
    ) {
        constexpr int threads = 256;
        int blocks = static_cast<int>(rows);

        //cross_entropy_loss_backward_kernel_fast<LT, threads> << <blocks, threads >> > (logits, targets, upstream, grad_logits, rows, vocab_size);
        cuda_cross_entropy_loss_backward_fused_kernel<LT> << <blocks, threads >> > (logits, targets, upstream, grad_logits, rows, vocab_size);

        check_cuda(cudaGetLastError(), "CUDA kernel launch error in cuda_cross_entropy_loss_backward");
    }


    template void cuda_cross_entropy_loss_backward_fast<float>(const float*, const int*, const float*, float*, size_t, size_t);
    template void cuda_cross_entropy_loss_backward_fast<double>(const double*, const int*, const double*, double*, size_t, size_t);

}