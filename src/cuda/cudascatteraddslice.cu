#include "cudaops.h"
#include "gpu_atomic.cuh"


namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function scatter_add_slice_kernel
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
	__global__ void scatter_add_slice_kernel(
		T* optr,
		const T* gptr,
		const size_t* a_shape,
		const size_t* a_strides,
		size_t a_offset,
		const size_t* out_shape,
		const size_t* out_strides,
		size_t out_offset,
		size_t rank,
		size_t onumel,
		size_t axis,
		size_t start,
		size_t step)
	{
		size_t linearidx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (linearidx >= onumel)
			return;

		size_t tmp = linearidx;

		size_t outidx = out_offset;
		size_t aidx = a_offset;

		// unravel linearidx into out multi-index, then map to A
		for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
			size_t coord = tmp % out_shape[d];
			tmp /= out_shape[d];

			outidx += coord * out_strides[d];

			size_t acoord = (static_cast<size_t>(d) == axis)
				? (start + coord * step)
				: coord;

			aidx += acoord * a_strides[d];
		}

		gpu_atomic_add(&optr[aidx], gptr[outidx]);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cuda_scatter_add_slice
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
	void cuda_scatter_add_slice(
		T* optr,
		const T* gptr,
		const std::vector<size_t>& shape,
		const std::vector<size_t>& strides,
		size_t offset,
		const std::vector<size_t>& out_shape,
		const std::vector<size_t>& out_strides,
		size_t out_offset,
		size_t onumel,
		size_t axis,
		size_t start,
		size_t step)
	{
		if (shape.size() != out_shape.size()) {
			throw std::runtime_error("cuda_scatter_add_slice: rank mismatch");
		}

		const size_t rank = out_shape.size();

		size_t* d_shape = nullptr;
		size_t* d_strides = nullptr;
		size_t* d_out_shape = nullptr;
		size_t* d_out_strides = nullptr;

		check_cuda(cudaMalloc(&d_shape, rank * sizeof(size_t)),"Error in cuda_scatter_add_slice");	
		check_cuda(cudaMalloc(&d_strides, rank * sizeof(size_t)), "Error in cuda_scatter_add_slice");
		check_cuda(cudaMalloc(&d_out_shape, rank * sizeof(size_t)), "Error in cuda_scatter_add_slice");
		check_cuda(cudaMalloc(&d_out_strides, rank * sizeof(size_t)), "Error in cuda_scatter_add_slice");
		
		check_cuda(cudaMemcpy(d_shape, shape.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice), "Error in cuda_scatter_add_slice");
		check_cuda(cudaMemcpy(d_strides, strides.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice), "Error in cuda_scatter_add_slice");
		check_cuda(cudaMemcpy(d_out_shape, out_shape.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice), "Error in cuda_scatter_add_slice");
		check_cuda(cudaMemcpy(d_out_strides, out_strides.data(), rank * sizeof(size_t), cudaMemcpyHostToDevice), "Error in cuda_scatter_add_slice");
		

		
		const int threads = 256;
		const int blocks = static_cast<int>((onumel + threads - 1) / threads);

		scatter_add_slice_kernel<T> << <blocks, threads >> > (
			optr,
			gptr,
			d_shape,
			d_strides,
			offset,
			d_out_shape,
			d_out_strides,
			out_offset,
			rank,
			onumel,
			axis,
			start,
			step
			);


		cudaFree(d_shape);
		cudaFree(d_strides);
		cudaFree(d_out_shape);
		cudaFree(d_out_strides);
		return;
	
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


	template void cuda_scatter_add_slice<int>(int*, const int*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t,
		size_t,
		size_t,
		size_t,
		size_t);

	template void cuda_scatter_add_slice<float>(float*, const float*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t,
		size_t,
		size_t,
		size_t,
		size_t);

	template void cuda_scatter_add_slice<double>(double*, const double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t,
		size_t,
		size_t,
		size_t,
		size_t);
}