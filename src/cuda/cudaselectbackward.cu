#include "cudaops.h"


namespace Inferno {

	template<typename AT>
	__global__ void select_backward_kernel(
		const AT* gptr,
		AT* optr,
		const size_t* out_shape,
		const size_t* gstrides,
		const size_t* parent_strides,
		int ndim_out,
		int axis,
		size_t index,
		size_t goffset,
		size_t poffset,
		size_t N
	) {
		size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (linear >= N) return;

		size_t tmp = linear;

		size_t gidx = goffset;
		size_t pidx = poffset;

		int ndim_parent = ndim_out + 1;
		int out_d = ndim_out - 1;

		for (int parent_d = ndim_parent - 1; parent_d >= 0; --parent_d) {
			if (parent_d == axis) {
				pidx += index * parent_strides[parent_d];
			}
			else {
				size_t coord = tmp % out_shape[out_d];
				tmp /= out_shape[out_d];

				gidx += coord * gstrides[out_d];
				pidx += coord * parent_strides[parent_d];
				--out_d;
			}
		}

		optr[pidx] = static_cast<AT>(gptr[gidx]);
	}

	template<typename AT>
	void cuda_select_backward_strided(
		const AT* gptr,
		AT* optr,
		const std::vector<size_t>& out_shape,
		const std::vector<size_t>& gstrides,
		const std::vector<size_t>& parent_strides,
		size_t goffset,
		size_t poffset,
		int axis,
		size_t index
	) {
		const int ndim_out = static_cast<int>(out_shape.size());
		if (gstrides.size() != out_shape.size() || parent_strides.size() != out_shape.size() + 1) {
			throw std::runtime_error("cuda_select_backward_strided: shape/stride rank mismatch");
		}

		size_t N = 1;
		for (size_t s : out_shape) N *= s;

		size_t* d_out_shape = nullptr;
		size_t* d_gstrides = nullptr;
		size_t* d_parent_strides = nullptr;

		cudaMalloc(&d_out_shape, ndim_out * sizeof(size_t));
		cudaMalloc(&d_gstrides, ndim_out * sizeof(size_t));
		cudaMalloc(&d_parent_strides, (ndim_out + 1) * sizeof(size_t));

		cudaMemcpy(d_out_shape, out_shape.data(), ndim_out * sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_gstrides, gstrides.data(), ndim_out * sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_parent_strides, parent_strides.data(), (ndim_out + 1) * sizeof(size_t), cudaMemcpyHostToDevice);

		constexpr int threads = 256;
		int blocks = static_cast<int>((N + threads - 1) / threads);

		select_backward_kernel<AT> << <blocks, threads >> > (
			gptr, optr,
			d_out_shape, d_gstrides, d_parent_strides,
			ndim_out, axis, index,
			goffset, poffset, N
			);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::printf("cuda_select_backward_strided launch failed: %s\n", cudaGetErrorString(err));
		}

		cudaFree(d_out_shape);
		cudaFree(d_gstrides);
		cudaFree(d_parent_strides);
	}

	template void cuda_select_backward_strided<float>(
		const float*, float*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, int, size_t);

	template void cuda_select_backward_strided<double>(
		const double*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, int, size_t);

	
	template void cuda_select_backward_strided<int>(
		const int*, int*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, int, size_t);
}