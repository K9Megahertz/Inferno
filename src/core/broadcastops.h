#pragma once
#include <inferno/core/tensor.h>
#include "core/tensorimpl.h"


namespace Inferno {
	Tensor sum_to_shape(const Tensor& src, const std::vector<size_t>& target_shape);
	Tensor scatter_add_embedding(const Tensor &embeddings, const Tensor& token_ids, const Tensor& g_out);
	Tensor scatter_add_slice(Tensor& g_a, const Tensor& g_out, int axis, size_t start, size_t step);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_sum_to_shape
	//  
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename AT, typename BT>
	void cpu_sum_to_shape(AT* dst_ptr, const BT* src_ptr, size_t src_numel, size_t src_rank, std::vector<size_t> src_shape, std::vector<size_t> temp_dst_strides, size_t out_numel) {

		for (size_t i = 0; i < out_numel; ++i) {
			dst_ptr[i] = static_cast<AT>(0);
		}

		std::vector<size_t> src_idx(src_rank, 0);

		for (size_t linear = 0; linear < src_numel; ++linear) {

			size_t tmp = linear;

			for (int d = static_cast<int>(src_rank) - 1; d >= 0; --d) {
				src_idx[d] = tmp % src_shape[d];
				tmp /= src_shape[d];
			}

			size_t dst_offset = 0;
			for (size_t d = 0; d < src_rank; ++d) {
				dst_offset += src_idx[d] * temp_dst_strides[d];
			}

			dst_ptr[dst_offset] += src_ptr[linear];
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_scatter_add_embedding
	//  
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename AT, typename BT>
	void cpu_scatter_add_embedding(const BT* gptr, const AT* tptr, BT* optr, size_t embed_dim, size_t numtokens) {


		for (size_t t = 0; t < numtokens; t++) {

			size_t tokenid = tptr[t];
			size_t obaseidx = tokenid * embed_dim;
			size_t gbaseidx= t * embed_dim;			
			for (size_t e = 0; e < embed_dim; e++) {
				optr[obaseidx++] += gptr[gbaseidx++];
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_scatter_add_slice
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
	void cpu_scatter_add_slice(
		T* optr,
		const T* gptr,
		std::vector<size_t>& shape,
		std::vector<size_t>& strides,
		size_t offset,
		std::vector<size_t>& out_shape,
		std::vector<size_t>& out_strides,
		size_t out_offset,
		size_t onumel,
		int axis,
		size_t start,
		size_t step) {

		
		std::vector<size_t> a_idx(shape.size());
		std::vector<size_t> out_idx(out_shape.size());		

		for (size_t linearidx = 0; linearidx < onumel; linearidx++) {

			size_t tmp = linearidx;
			size_t rank = out_shape.size();
			for (int d = rank - 1; d >= 0; d--) {
				out_idx[d] = tmp % out_shape[d];
				tmp /= out_shape[d];
			}

			//map to A
			a_idx = out_idx;
			a_idx[axis] = start + out_idx[axis] * step;

			//compute storage offsets
			size_t outidx = out_offset;
			size_t aidx = offset;

			for (int d = 0; d < rank; ++d) {
				outidx += out_idx[d] * out_strides[d];
				aidx += a_idx[d] * strides[d];
			}

			//scatter
			optr[aidx] += gptr[outidx];

		}
	}
}