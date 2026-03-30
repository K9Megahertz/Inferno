#pragma once

#include "node.h"
#include "../tensor.h"

namespace Inferno {

	class ReshapeBackward : public Node {
	public:
		explicit ReshapeBackward(const Tensor& A);

		void backward() override;
		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;

	private:
		Tensor m_A;
	};

}