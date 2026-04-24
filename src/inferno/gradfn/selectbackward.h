#pragma once

#include "node.h"
#include <inferno/core/tensor.h>


namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class SelectBackward 
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class SelectBackward : public Node {
	public:
		SelectBackward(const Tensor& A, int axis, size_t index);

		void backward() override;
		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;

	private:
		Tensor m_A;
		int m_axis;
		size_t m_index;
	};


	template<typename AT>
	void cpu_select_backward_strided(
		const AT* gptr,
		AT* optr,
		const std::vector<size_t>& out_shape,      // shape of g_out
		const std::vector<size_t>& gstrides,       // strides of g_out
		const std::vector<size_t>& parent_strides, // strides of g_A
		size_t goffset,
		size_t poffset,
		int axis,
		size_t index
	) {
		const size_t ndim_out = out_shape.size();
		const size_t ndim_parent = ndim_out + 1;

		if (gstrides.size() != ndim_out || parent_strides.size() != ndim_parent) {
			throw std::runtime_error("cpu_select_backward_strided: shape/stride rank mismatch");
		}

		size_t N = 1;
		for (size_t s : out_shape) N *= s;

		for (size_t linear = 0; linear < N; ++linear) {
			size_t tmp = linear;

			size_t gidx = goffset;
			size_t pidx = poffset;

			int out_d = static_cast<int>(ndim_out) - 1;

			for (int parent_d = static_cast<int>(ndim_parent) - 1; parent_d >= 0; --parent_d) {
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
	}


}