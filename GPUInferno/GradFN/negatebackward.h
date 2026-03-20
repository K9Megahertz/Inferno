#pragma once
#include "node.h"
#include "../tensor.h"
#include "../broadcastops.h"



namespace Inferno {


	class NegateBackward : public Node {

	public:

		NegateBackward(const Tensor& A);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_A;
		

	};





}