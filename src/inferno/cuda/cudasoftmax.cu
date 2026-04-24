#include "cudaops.h"

namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function decode_group_base
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__device__ size_t decode_group_base(
		size_t group,
		const size_t* shape,
		const size_t* strides,
		int ndim,
		int axis,
		size_t offset
	) {
		size_t base = offset;
		size_t tmp = group;

		for (int d = ndim - 1; d >= 0; --d) {
			if (d == axis) continue;

			size_t coord = tmp % shape[d];
			tmp /= shape[d];

			base += coord * strides[d];
		}

		return base;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function softmax_strided_kernel
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename RT>
	__global__ void softmax_strided_kernel(
		const AT* aptr,
		RT* optr,
		const size_t* shape,
		const size_t* astrides,
		const size_t* ostrides,
		int ndim,
		int axis,
		size_t aoffset,
		size_t ooffset,
		size_t groups,
		size_t axis_size
	) {
		size_t group = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (group >= groups) return;

		size_t abase = decode_group_base(group, shape, astrides, ndim, axis, aoffset);
		size_t obase = decode_group_base(group, shape, ostrides, ndim, axis, ooffset);

		// 1) max
		AT max_val = aptr[abase];
		for (size_t k = 1; k < axis_size; ++k) {
			size_t aidx = abase + k * astrides[axis];
			if (aptr[aidx] > max_val) {
				max_val = aptr[aidx];
			}
		}

		// 2) exp + sum
		RT sum = static_cast<RT>(0);
		for (size_t k = 0; k < axis_size; ++k) {
			size_t aidx = abase + k * astrides[axis];
			size_t oidx = obase + k * ostrides[axis];

			RT e = static_cast<RT>(exp((double)(aptr[aidx] - max_val)));
			optr[oidx] = e;
			sum += e;
		}

		// 3) normalize
		for (size_t k = 0; k < axis_size; ++k) {
			size_t oidx = obase + k * ostrides[axis];
			optr[oidx] /= sum;
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cuda_softmax
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename RT>
	void cuda_softmax(
		const AT* aptr,
		RT* optr,
		const std::vector<size_t>& shape,
		const std::vector<size_t>& astrides,
		const std::vector<size_t>& ostrides,
		size_t aoffset,
		size_t ooffset,
		int axis
	) {
		int ndim = static_cast<int>(shape.size());
		if (axis < 0) axis += ndim;
		if (axis < 0 || axis >= ndim) {
			throw std::runtime_error("softmax: invalid axis");
		}

		if (astrides.size() != shape.size() || ostrides.size() != shape.size()) {
			throw std::runtime_error("softmax: shape/stride rank mismatch");
		}

		size_t axis_size = shape[axis];
		size_t groups = 1;
		for (int d = 0; d < ndim; ++d) {
			if (d == axis) continue;
			groups *= shape[d];
		}

		size_t* d_shape = nullptr;
		size_t* d_astrides = nullptr;
		size_t* d_ostrides = nullptr;

		cudaMalloc(&d_shape, ndim * sizeof(size_t));
		cudaMalloc(&d_astrides, ndim * sizeof(size_t));
		cudaMalloc(&d_ostrides, ndim * sizeof(size_t));

		cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_astrides, astrides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ostrides, ostrides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);

		constexpr int threads = 256;
		int blocks = static_cast<int>((groups + threads - 1) / threads);

		softmax_strided_kernel<AT,RT> << <blocks, threads >> > (
			aptr,
			optr,
			d_shape,
			d_astrides,
			d_ostrides,
			ndim,
			axis,
			aoffset,
			ooffset,
			groups,
			axis_size
			);

		check_cuda(cudaGetLastError(), "cuda_softmax_strided launch failed");		

		cudaFree(d_shape);
		cudaFree(d_astrides);
		cudaFree(d_ostrides);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Explicit Instantiations
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template void cuda_softmax<int, float>(
		const int*, float*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, int);

	template void cuda_softmax<float, float>(
		const float*, float*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, int);

	template void cuda_softmax<double, double>(
		const double*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, int);

	



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function softmax_backward_strided_kernel
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename GT, typename RT>
	__global__ void softmax_backward_strided_kernel(
		const AT* yptr,
		const GT* gptr,
		RT* optr,
		const size_t* shape,
		const size_t* ystrides,
		const size_t* gstrides,
		const size_t* ostrides,
		int ndim,
		int axis,
		size_t yoffset,
		size_t goffset,
		size_t ooffset,
		size_t groups,
		size_t axis_size
	) {
		size_t group = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (group >= groups)
			return;

		size_t tmp = group;

		size_t ybase = yoffset;
		size_t gbase = goffset;
		size_t obase = ooffset;

		// decode coordinates for all dims except axis
		for (int d = ndim - 1; d >= 0; --d) {
			if (d == axis) continue;

			size_t coord = tmp % shape[d];
			tmp /= shape[d];

			ybase += coord * ystrides[d];
			gbase += coord * gstrides[d];
			obase += coord * ostrides[d];
		}

		RT dot = static_cast<RT>(0);

		for (size_t k = 0; k < axis_size; ++k) {
			const size_t yidx = ybase + k * ystrides[axis];
			const size_t gidx = gbase + k * gstrides[axis];

			dot += static_cast<RT>(gptr[gidx]) * static_cast<RT>(yptr[yidx]);
		}

		for (size_t k = 0; k < axis_size; ++k) {
			const size_t yidx = ybase + k * ystrides[axis];
			const size_t gidx = gbase + k * gstrides[axis];
			const size_t oidx = obase + k * ostrides[axis];

			const RT y = static_cast<RT>(yptr[yidx]);
			const RT g = static_cast<RT>(gptr[gidx]);

			optr[oidx] = y * (g - dot);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cuda_sigmoid_backward
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename GT, typename RT>
	void cuda_softmax_backward(
		const AT* yptr,
		const GT* gptr,
		RT* optr,
		const std::vector<size_t>& shape,
		const std::vector<size_t>& ystrides,
		const std::vector<size_t>& gstrides,
		const std::vector<size_t>& ostrides,
		size_t yoffset,
		size_t goffset,
		size_t ooffset,
		int axis
	) {
		const int ndim = static_cast<int>(shape.size());

		if (axis < 0)
			axis += ndim;

		if (axis < 0 || axis >= ndim)
			throw std::runtime_error("softmax backward: invalid axis");

		if (ystrides.size() != shape.size() ||
			gstrides.size() != shape.size() ||
			ostrides.size() != shape.size()) {
			throw std::runtime_error("softmax backward: shape/stride rank mismatch");
		}

		size_t axis_size = shape[axis];
		size_t groups = 1;
		for (int d = 0; d < ndim; ++d) {
			if (d == axis) continue;
			groups *= shape[d];
		}

		size_t* d_shape = nullptr;
		size_t* d_ystrides = nullptr;
		size_t* d_gstrides = nullptr;
		size_t* d_ostrides = nullptr;

		cudaMalloc(&d_shape, ndim * sizeof(size_t));
		cudaMalloc(&d_ystrides, ndim * sizeof(size_t));
		cudaMalloc(&d_gstrides, ndim * sizeof(size_t));
		cudaMalloc(&d_ostrides, ndim * sizeof(size_t));

		cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ystrides, ystrides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_gstrides, gstrides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ostrides, ostrides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);

		constexpr int threads = 256;
		int blocks = static_cast<int>((groups + threads - 1) / threads);

		softmax_backward_strided_kernel<AT, GT, RT> << <blocks, threads >> > (
			yptr,
			gptr,
			optr,
			d_shape,
			d_ystrides,
			d_gstrides,
			d_ostrides,
			ndim,
			axis,
			yoffset,
			goffset,
			ooffset,
			groups,
			axis_size
			);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::printf("cuda_softmax_backward_strided launch failed: %s\n", cudaGetErrorString(err));
		}

		cudaFree(d_shape);
		cudaFree(d_ystrides);
		cudaFree(d_gstrides);
		cudaFree(d_ostrides);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Explicit Instantiations
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template void cuda_softmax_backward<float, float, float>(
		const float*, const float*, float*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);

	template void cuda_softmax_backward<double, double, double>(
		const double*, const double*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);

	template void cuda_softmax_backward<float, double, double>(
		const float*, const double*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);

	template void cuda_softmax_backward<double, float, double>(
		const double*, const float*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);



	

}