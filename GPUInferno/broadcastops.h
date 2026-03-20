#pragma once
#include "tensor.h"


namespace Inferno {
	Tensor sum_to_shape(const Tensor& src, const std::vector<size_t>& target_shape);

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


}