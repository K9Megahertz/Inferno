#pragma once
#include "node.h"
#include "../tensor.h"
#include "../ops.h"


namespace Inferno {


	class AccumulateGrad : public Node {

	public:

		//AccumulateGrad(const Tensor& A);
		explicit AccumulateGrad(std::weak_ptr<TensorImpl> leaf)	: m_leaf(std::move(leaf)) {
		
		}

		void backward() override;


	private:
		
		void get_inputs(std::vector<Tensor>& out) const override { out.clear(); }
		void release() override;

		std::weak_ptr<TensorImpl> m_leaf;
	};


}