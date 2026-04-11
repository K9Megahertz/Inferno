#include <vector>

namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function num_groups_excluding_axis
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static size_t num_groups_excluding_axis(const std::vector<size_t>& shape, int axis) {

		size_t groups = 1;
		for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
			if (d == axis) continue;
			groups *= shape[d];
		}
		return groups;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_softmax
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	template<typename AT, typename RT>
	void cpu_softmax(
		const AT* aptr,
		RT* optr,
		const std::vector<size_t>& shape,
		const std::vector<size_t>& astrides,
		const std::vector<size_t>& ostrides,
		size_t aoffset,
		size_t ooffset,
		int axis
	) {
		const int ndim = static_cast<int>(shape.size());

		if (astrides.size() != shape.size() || ostrides.size() != shape.size()) {
			throw std::runtime_error("cpu_softmax: shape/stride rank mismatch");
			exit(1);
		}

		const size_t axis_size = shape[axis];
		const size_t groups = num_groups_excluding_axis(shape, axis);

		for (size_t group = 0; group < groups; ++group) {
			size_t tmp = group;

			size_t abase = aoffset;
			size_t obase = ooffset;

			// Decode this logical group into coordinates for all dims except axis
			for (int d = ndim - 1; d >= 0; --d) {
				if (d == axis) continue;

				const size_t coord = tmp % shape[d];
				tmp /= shape[d];

				abase += coord * astrides[d];
				obase += coord * ostrides[d];
			}

			// 1) max along axis
			RT max_val = std::numeric_limits<RT>::lowest();
			for (size_t k = 0; k < axis_size; ++k) {
				size_t aidx = abase + k * astrides[axis];
				if (aptr[aidx] > max_val) {
					max_val = aptr[aidx];
				}
			}

			// 2) exp-shift + sum
			RT sum = static_cast<RT>(0);
			for (size_t k = 0; k < axis_size; ++k) {
				size_t aidx = abase + k * astrides[axis];
				size_t oidx = obase + k * ostrides[axis];

				RT e = static_cast<RT>(std::exp(static_cast<double>(aptr[aidx] - max_val)));
				optr[oidx] = e;
				sum += e;
			}

			// 3) normalize
			for (size_t k = 0; k < axis_size; ++k) {
				size_t oidx = obase + k * ostrides[axis];
				optr[oidx] /= sum;
			}
		}
	 }

}