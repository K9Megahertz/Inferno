#include "cudaops.h"

namespace Inferno {

	template <typename T>
	__global__ void concat_kernel(
		const T** src_ptrs,
		T* optr,
		const size_t* src_shapes_flat,
		const size_t* src_strides_flat,
		const size_t* src_offsets,
		const size_t* axis_starts,
		const size_t* out_shape,
		const size_t* out_strides,
		size_t out_offset,
		size_t out_numel,
		size_t axis,
		size_t rank,
		size_t num_tensors)
	{
		size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
		if (linear >= out_numel) return;

		size_t tmp = linear;

		size_t out_idx_local[16];
		size_t src_idx_local[16];

		// adjust this if you want larger max rank support
		if (rank > 16) return;

		for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
			out_idx_local[d] = tmp % out_shape[d];
			tmp /= out_shape[d];
		}

		size_t out_axis_idx = out_idx_local[axis];

		size_t tensor_idx = 0;
		for (size_t i = 0; i < num_tensors; ++i) {
			size_t start = axis_starts[i];
			size_t end = start + src_shapes_flat[i * rank + axis];
			if (out_axis_idx >= start && out_axis_idx < end) {
				tensor_idx = i;
				break;
			}
		}

		for (size_t d = 0; d < rank; ++d) {
			src_idx_local[d] = out_idx_local[d];
		}
		src_idx_local[axis] -= axis_starts[tensor_idx];

		size_t dst_storage_idx = out_offset;
		for (size_t d = 0; d < rank; ++d) {
			dst_storage_idx += out_idx_local[d] * out_strides[d];
		}

		size_t src_storage_idx = src_offsets[tensor_idx];
		for (size_t d = 0; d < rank; ++d) {
			src_storage_idx += src_idx_local[d] * src_strides_flat[tensor_idx * rank + d];
		}

		optr[dst_storage_idx] = src_ptrs[tensor_idx][src_storage_idx];
	}

	template <typename T>
	void cuda_concat(
		const std::vector<const T*>& src_ptrs_host,
		T* optr,
		const std::vector<size_t>& src_shapes_flat_host,
		const std::vector<size_t>& src_strides_flat_host,
		const std::vector<size_t>& src_offsets_host,
		const std::vector<size_t>& axis_starts_host,
		const std::vector<size_t>& out_shape_host,
		const std::vector<size_t>& out_strides_host,
		size_t out_offset,
		size_t out_numel,
		size_t axis,
		size_t rank)
	{
		const size_t num_tensors = src_ptrs_host.size();

		const T** d_src_ptrs = nullptr;
		size_t* d_src_shapes_flat = nullptr;
		size_t* d_src_strides_flat = nullptr;
		size_t* d_src_offsets = nullptr;
		size_t* d_axis_starts = nullptr;
		size_t* d_out_shape = nullptr;
		size_t* d_out_strides = nullptr;

		cudaError_t err;

		check_cuda(cudaMalloc(&d_src_ptrs, num_tensors * sizeof(T*)),"what");
		

		err = cudaMalloc(&d_src_shapes_flat, src_shapes_flat_host.size() * sizeof(size_t));
		if (err != cudaSuccess) goto fail;

		err = cudaMalloc(&d_src_strides_flat, src_strides_flat_host.size() * sizeof(size_t));
		if (err != cudaSuccess) goto fail;

		err = cudaMalloc(&d_src_offsets, src_offsets_host.size() * sizeof(size_t));
		if (err != cudaSuccess) goto fail;

		err = cudaMalloc(&d_axis_starts, axis_starts_host.size() * sizeof(size_t));
		if (err != cudaSuccess) goto fail;

		err = cudaMalloc(&d_out_shape, out_shape_host.size() * sizeof(size_t));
		if (err != cudaSuccess) goto fail;

		err = cudaMalloc(&d_out_strides, out_strides_host.size() * sizeof(size_t));
		if (err != cudaSuccess) goto fail;

		err = cudaMemcpy(d_src_ptrs, src_ptrs_host.data(), num_tensors * sizeof(T*), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) goto fail;

		err = cudaMemcpy(d_src_shapes_flat, src_shapes_flat_host.data(),
			src_shapes_flat_host.size() * sizeof(size_t), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) goto fail;

		err = cudaMemcpy(d_src_strides_flat, src_strides_flat_host.data(),
			src_strides_flat_host.size() * sizeof(size_t), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) goto fail;

		err = cudaMemcpy(d_src_offsets, src_offsets_host.data(),
			src_offsets_host.size() * sizeof(size_t), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) goto fail;

		err = cudaMemcpy(d_axis_starts, axis_starts_host.data(),
			axis_starts_host.size() * sizeof(size_t), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) goto fail;

		err = cudaMemcpy(d_out_shape, out_shape_host.data(),
			out_shape_host.size() * sizeof(size_t), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) goto fail;

		err = cudaMemcpy(d_out_strides, out_strides_host.data(),
			out_strides_host.size() * sizeof(size_t), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) goto fail;

		{
			const int threads = 256;
			const int blocks = static_cast<int>((out_numel + threads - 1) / threads);

			concat_kernel<T> << <blocks, threads >> > (
				d_src_ptrs,
				optr,
				d_src_shapes_flat,
				d_src_strides_flat,
				d_src_offsets,
				d_axis_starts,
				d_out_shape,
				d_out_strides,
				out_offset,
				out_numel,
				axis,
				rank,
				num_tensors
				);

			err = cudaGetLastError();
			if (err != cudaSuccess) goto fail;
		}

		cudaFree(d_src_ptrs);
		cudaFree(d_src_shapes_flat);
		cudaFree(d_src_strides_flat);
		cudaFree(d_src_offsets);
		cudaFree(d_axis_starts);
		cudaFree(d_out_shape);
		cudaFree(d_out_strides);
		return;

	fail:
		Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "cuda_concat failed.");
		if (d_src_ptrs) cudaFree(d_src_ptrs);
		if (d_src_shapes_flat) cudaFree(d_src_shapes_flat);
		if (d_src_strides_flat) cudaFree(d_src_strides_flat);
		if (d_src_offsets) cudaFree(d_src_offsets);
		if (d_axis_starts) cudaFree(d_axis_starts);
		if (d_out_shape) cudaFree(d_out_shape);
		if (d_out_strides) cudaFree(d_out_strides);
		exit(1);
	}


	template void cuda_concat<int>(
		const std::vector<const int*>& src_ptrs_host,
		int* optr,
		const std::vector<size_t>& src_shapes_flat_host,
		const std::vector<size_t>& src_strides_flat_host,
		const std::vector<size_t>& src_offsets_host,
		const std::vector<size_t>& axis_starts_host,
		const std::vector<size_t>& out_shape_host,
		const std::vector<size_t>& out_strides_host,
		size_t out_offset,
		size_t out_numel,
		size_t axis,
		size_t rank);

	template void cuda_concat<float>(
		const std::vector<const float*>& src_ptrs_host,
		float* optr,
		const std::vector<size_t>& src_shapes_flat_host,
		const std::vector<size_t>& src_strides_flat_host,
		const std::vector<size_t>& src_offsets_host,
		const std::vector<size_t>& axis_starts_host,
		const std::vector<size_t>& out_shape_host,
		const std::vector<size_t>& out_strides_host,
		size_t out_offset,
		size_t out_numel,
		size_t axis,
		size_t rank);

	template void cuda_concat<double>(
		const std::vector<const double*>& src_ptrs_host,
		double* optr,
		const std::vector<size_t>& src_shapes_flat_host,
		const std::vector<size_t>& src_strides_flat_host,
		const std::vector<size_t>& src_offsets_host,
		const std::vector<size_t>& axis_starts_host,
		const std::vector<size_t>& out_shape_host,
		const std::vector<size_t>& out_strides_host,
		size_t out_offset,
		size_t out_numel,
		size_t axis,
		size_t rank);



}