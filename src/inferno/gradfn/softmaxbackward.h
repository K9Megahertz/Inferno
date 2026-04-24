#pragma once
#include "node.h"
#include <inferno/core/tensor.h>
#include "inferno/core/broadcastops.h"



namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class SoftmaxBackward 
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class SoftmaxBackward : public Node {

	public:

		SoftmaxBackward(const Tensor& A, const Tensor& out, int axis);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_A;
		Tensor m_out;
		int m_axis;


	};


	static int normalize_softmax_axis(int axis, int ndim) {
		if (axis < 0)
			axis += ndim;

		if (axis < 0 || axis >= ndim)
			throw std::runtime_error("softmax backward: invalid axis");

		return axis;
	}

	static size_t softmax_groups_excluding_axis(const std::vector<size_t>& shape, int axis) {
		size_t groups = 1;
		for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
			if (d == axis) continue;
			groups *= shape[d];
		}
		return groups;
	}


	template<typename AT, typename GT, typename RT>
	void cpu_softmax_backward(
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
		//axis = normalize_softmax_axis(axis, ndim);

		if (ystrides.size() != shape.size() ||
			gstrides.size() != shape.size() ||
			ostrides.size() != shape.size()) {
			throw std::runtime_error("softmax backward: shape/stride rank mismatch");
		}

		const size_t axis_size = shape[axis];
		const size_t groups = softmax_groups_excluding_axis(shape, axis);

		for (size_t group = 0; group < groups; ++group) {
			size_t tmp = group;

			size_t ybase = yoffset;
			size_t gbase = goffset;
			size_t obase = ooffset;

			// decode group index across all dims except axis
			for (int d = ndim - 1; d >= 0; --d) {
				if (d == axis) continue;

				const size_t coord = tmp % shape[d];
				tmp /= shape[d];

				ybase += coord * ystrides[d];
				gbase += coord * gstrides[d];
				obase += coord * ostrides[d];
			}

			RT dot = static_cast<RT>(0);

			// dot = sum(g * y) along axis
			for (size_t k = 0; k < axis_size; ++k) {
				const size_t yidx = ybase + k * ystrides[axis];
				const size_t gidx = gbase + k * gstrides[axis];

				dot += static_cast<RT>(gptr[gidx]) * static_cast<RT>(yptr[yidx]);
			}

			// dx = y * (g - dot)
			for (size_t k = 0; k < axis_size; ++k) {
				const size_t yidx = ybase + k * ystrides[axis];
				const size_t gidx = gbase + k * gstrides[axis];
				const size_t oidx = obase + k * ostrides[axis];

				const RT y = static_cast<RT>(yptr[yidx]);
				const RT g = static_cast<RT>(gptr[gidx]);

				optr[oidx] = y * (g - dot);
			}
		}
	}

	template void cpu_softmax_backward<float, float, float>(
		const float*, const float*, float*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);

	template void cpu_softmax_backward<double, double, double>(
		const double*, const double*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);

	template void cpu_softmax_backward<float, double, double>(
		const float*, const double*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);

	template void cpu_softmax_backward<double, float, double>(
		const double*, const float*, double*,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		const std::vector<size_t>&,
		size_t, size_t, size_t, int
	);




}