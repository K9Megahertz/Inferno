#pragma once
#include<vector>
#include <cuda_runtime.h>
#include <cstddef>
#include <device_launch_parameters.h>
#include "../Util/logger.h"


namespace Inferno {

	
	
	template<typename AT, typename BT, typename RT>	
	void cuda_add(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& astrides, size_t aoffset, const std::vector<size_t>& bshape, const std::vector<size_t>& bstrides, size_t boffset, const std::vector<size_t>& out_shape, size_t out_numel);

	template<typename AT, typename BT, typename RT>
	void cuda_subtract(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& astrides, size_t aoffset, const std::vector<size_t>& bshape, const std::vector<size_t>& bstrides, size_t boffset, const std::vector<size_t>& out_shape, size_t out_numel);

	template<typename AT, typename BT, typename RT>	
	void cuda_multiply(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& astrides, size_t aoffset, const std::vector<size_t>& bshape, const std::vector<size_t>& bstrides, size_t boffset, const std::vector<size_t>& out_shape, size_t out_numel);

	template<typename AT, typename BT, typename RT>
	void cuda_divide(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& astrides, size_t aoffset, const std::vector<size_t>& bshape, const std::vector<size_t>& bstrides, size_t boffset, const std::vector<size_t>& out_shape, size_t out_numel);

	template<typename AT>
	void cuda_negate(const AT* aptr, AT* outptr, size_t N);

	template<typename AT>
	void cuda_fill(AT* aptr, const AT value, size_t N);

	template<typename AT, typename BT, typename RT>
	void cuda_matmul(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& a_shape, const std::vector<size_t>& a_strides, const std::vector<size_t>& b_shape, const std::vector<size_t>& b_strides, const std::vector<size_t>& out_shape);

	template<typename AT>
	void cuda_sum_to_shape(AT* dst_ptr, const AT* src_ptr, size_t src_numel, size_t src_rank, const std::vector<size_t>& src_shape, const std::vector<size_t>& temp_dst_strides, size_t out_numel);

	template<typename AT, typename BT, typename RT>
	void cuda_mse_loss(const AT* a, const BT* b, RT* out, size_t numel);

	template<typename AT, typename BT, typename RT>
	void cuda_mse_loss_backward(const AT* aptr, const BT* bptr, RT* gaptr, RT* gbptr, const RT* gout, size_t numel);

	template<typename AT, typename RT>
	void cuda_sigmoid(const AT* aptr, RT* outptr, size_t N);

	template<typename AT, typename GT, typename RT>
	void cuda_sigmoid_backward(const AT* yptr, const GT* gptr, RT* outptr, size_t n);

	template <typename AT, typename BT>
	void cuda_step_impl(AT* dptr, const BT* gptr, size_t N, float lr);

	template <typename AT, typename BT>
	void cuda_embedding(const BT* tptr, const AT* eptr, AT* optr, size_t num_batches, size_t seq_len, size_t embed_dim);

	template <typename AT, typename BT>
	void cuda_scatter_add_embedding(const BT* gptr, const AT* tptr, BT* eptr, size_t embed_dim, size_t numtokens);

	template <typename AT>
	void cuda_scatter_add_slice(AT* optr, const AT* gptr, 
		const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset, 
		const std::vector<size_t>& out_shape, const std::vector<size_t>& out_strides, size_t out_offset, 
		size_t onumel, size_t axis, size_t start, size_t step);

	template <typename AT>
	void cuda_layer_normalization(const AT* iptr, AT* optr, float* gptr, float* bptr, size_t num_batches, size_t dim);

	inline void check_cuda(cudaError_t err, const char* msg) {
		if (err != cudaSuccess) {			
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, std::string(msg) + ": " + cudaGetErrorString(err));
			exit(1);		
		}
	}

	


}