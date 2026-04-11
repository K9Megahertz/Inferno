#pragma once
#include "cudaops.h"


namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  masked_fill_kernel
    //
    //  Writes:
    //      out[idx] = (mask == true) ? value : input
    //
    //  Assumptions:
    //  - output shape is the same as input shape
    //  - mask is broadcastable to input shape
    //  - input and mask may be non-contiguous and may have offsets
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename MT>
    __global__ void masked_fill_kernel(
        const AT* iptr,
        const MT* mptr,
        AT* optr,
        const size_t* input_shape,
        const size_t* input_strides,
        size_t input_offset,
        int input_rank,
        const size_t* mask_shape,
        const size_t* mask_strides,
        size_t mask_offset,
        int mask_rank,
        size_t total_elements,
        AT value)
    {
        size_t linear_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (linear_idx >= total_elements)
            return;

        // Convert output linear index into multi-index in input/output shape
        // Since out.shape == input.shape, we unravel using input_shape.
        size_t rem = linear_idx;

        size_t input_storage_idx = input_offset;
        size_t mask_storage_idx = mask_offset;

        // Walk dimensions from last to first
        for (int d = input_rank - 1; d >= 0; --d) {
            size_t coord = rem % input_shape[d];
            rem /= input_shape[d];

            // input index is direct because out.shape == input.shape
            input_storage_idx += coord * input_strides[d];

            // broadcast mask against input shape
            int md = d - (input_rank - mask_rank);   // aligned mask dim
            if (md >= 0) {
                size_t mask_dim = mask_shape[md];

                // If mask dim == 1, it is broadcast, so coordinate stays 0.
                // Otherwise coordinate matches input coordinate.
                size_t mask_coord = (mask_dim == 1) ? 0 : coord;

                mask_storage_idx += mask_coord * mask_strides[md];
            }
        }

        MT mask_val = mptr[mask_storage_idx];
        optr[linear_idx] = static_cast<bool>(mask_val) ? value : iptr[input_storage_idx];
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  cuda_masked_fill
    //
    //  Host launcher
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename MT>
    void cuda_masked_fill(
        const AT* iptr,
        const MT* mptr,
        AT* optr,
        const std::vector<size_t>& input_shape,
        const std::vector<size_t>& input_strides,
        size_t input_offset,
        const std::vector<size_t>& mask_shape,
        const std::vector<size_t>& mask_strides,
        size_t mask_offset,
        size_t total_elements,
        AT value)
    {
        int input_rank = static_cast<int>(input_shape.size());
        int mask_rank = static_cast<int>(mask_shape.size());

        if (input_shape.size() != input_strides.size()) {
            throw std::runtime_error("cuda_masked_fill: input_shape and input_strides size mismatch");
        }

        if (mask_shape.size() != mask_strides.size()) {
            throw std::runtime_error("cuda_masked_fill: mask_shape and mask_strides size mismatch");
        }

        if (mask_rank > input_rank) {
            throw std::runtime_error("cuda_masked_fill: mask rank cannot exceed input rank");
        }

        // Optional broadcastability check
        for (int i = 0; i < mask_rank; ++i) {
            size_t in_dim = input_shape[input_rank - mask_rank + i];
            size_t mk_dim = mask_shape[i];

            if (mk_dim != 1 && mk_dim != in_dim) {
                throw std::runtime_error("cuda_masked_fill: mask is not broadcastable to input shape");
            }
        }

        size_t* d_input_shape = nullptr;
        size_t* d_input_strides = nullptr;
        size_t* d_mask_shape = nullptr;
        size_t* d_mask_strides = nullptr;

        cudaMalloc(&d_input_shape, input_rank * sizeof(size_t));
        cudaMalloc(&d_input_strides, input_rank * sizeof(size_t));
        cudaMalloc(&d_mask_shape, mask_rank * sizeof(size_t));
        cudaMalloc(&d_mask_strides, mask_rank * sizeof(size_t));

        cudaMemcpy(d_input_shape, input_shape.data(), input_rank * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_strides, input_strides.data(), input_rank * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask_shape, mask_shape.data(), mask_rank * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask_strides, mask_strides.data(), mask_rank * sizeof(size_t), cudaMemcpyHostToDevice);

        constexpr int threads = 256;
        int blocks = static_cast<int>((total_elements + threads - 1) / threads);

        masked_fill_kernel<AT, MT> << <blocks, threads >> > (
            iptr,
            mptr,
            optr,
            d_input_shape,
            d_input_strides,
            input_offset,
            input_rank,
            d_mask_shape,
            d_mask_strides,
            mask_offset,
            mask_rank,
            total_elements,
            value
            );

        cudaFree(d_input_shape);
        cudaFree(d_input_strides);
        cudaFree(d_mask_shape);
        cudaFree(d_mask_strides);
    }


    /*template void Inferno::cuda_masked_fill<float, bool>(
        const float*, const bool*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        float
    );*/

    template void Inferno::cuda_masked_fill<int, int>(
        const int*, const int*, int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        int
    );

    template void Inferno::cuda_masked_fill<int, float>(
        const int*, const float*, int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        int
    );


    template void Inferno::cuda_masked_fill<int, double>(
        const int*, const double*, int*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        int
    );

    template void Inferno::cuda_masked_fill<double, int>(
        const double*, const int*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        double
    );

    template void Inferno::cuda_masked_fill<float, int>(
        const float*, const int*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        float
    );


    template void Inferno::cuda_masked_fill<float, float>(
        const float*, const float*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        float
    );


    template void Inferno::cuda_masked_fill<double, double>(
        const double*, const double*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        double
    );

    template void Inferno::cuda_masked_fill<double, float>(
        const double*, const float*, double*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        double
    );

    template void Inferno::cuda_masked_fill<float, double>(
        const float*, const double*, float*,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        const std::vector<size_t>&,
        const std::vector<size_t>&,
        size_t,
        size_t,
        float
    );

} 
