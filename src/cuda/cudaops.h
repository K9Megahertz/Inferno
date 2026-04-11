#pragma once
#include<vector>
#include <cuda_runtime.h>
#include <cstddef>
#include <device_launch_parameters.h>
#include <util/logger.h>

	

namespace Inferno {

	constexpr int MAX_DIMS = 12;

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


	template<typename AT, typename BT, typename RT>
	void cuda_matmul_fast(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& a_shape, const std::vector<size_t>& a_strides, const std::vector<size_t>& b_shape, const std::vector<size_t>& b_strides, const std::vector<size_t>& out_shape);

	template<typename AT, typename BT, typename RT>
	void cuda_matmul_fast2(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& a_shape, const std::vector<size_t>& a_strides, const std::vector<size_t>& b_shape, const std::vector<size_t>& b_strides, const std::vector<size_t>& out_shape);

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
	//void cuda_layer_normalization(const AT* iptr, AT* optr, float* gptr, float* bptr, size_t num_batches, size_t dim);
	void cuda_layer_normalization(const AT* iptr, AT* optr, const float* gptr, const float* bptr, float* meanptr, float* invstdptr, size_t num_batches, size_t dim, float eps);


	template<typename AT, typename GT>
	void cuda_layernorm_backward(const AT* aptr, const AT* goutptr, const float* gptr, const float* meanptr, const float* invstdptr,
		GT* gaptr, float* ggptr, float* gbptr, size_t num_batches, size_t dim);


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
		size_t rank);


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
	);

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
	);

	template<typename AT, typename RT>
	void cuda_gelu(const AT* aptr, RT* optr, size_t N, size_t off);

	template<typename AT, typename RT>
	void cuda_gelu_strided(const AT* aptr, RT* optr, const std::vector<size_t>& shape, const std::vector<size_t>& astrides, const std::vector<size_t>& ostrides, size_t aoffset, size_t ooffset);


	template<typename AT, typename GT, typename RT>
	void cuda_gelu_backward(const AT* aptr, const GT* gptr, RT* optr, size_t N, size_t off);


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
	);

	template<typename AT>
	void cuda_contiguous_copy(const AT* aptr, AT* optr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset, size_t N);

	template<typename AT>
	void cuda_triu(const AT* aptr, AT* optr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset, size_t out_numel, int diagonal);


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
		AT value);

	template <typename LT>
	void cuda_cross_entropy_loss(const LT* logits, const int* targets, LT* out, size_t rows, size_t vocab_size);
	

	template <typename LT>
	void cuda_cross_entropy_loss_backward(const LT* logits, const int* targets, const LT* upstream, LT* grad_logits, size_t rows, size_t vocab_size);


	inline void check_cuda(cudaError_t err, const char* msg) {
		if (err != cudaSuccess) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, std::string(msg) + ": " + cudaGetErrorString(err));
			exit(1);
		}
	}




}