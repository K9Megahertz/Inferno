#pragma once
#include "node.h"
#include "../tensor.h"
#include "../broadcastops.h"



namespace Inferno {


	class AddBackward : public Node {

	public:

		AddBackward(const Tensor& A, const Tensor& B);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;
		

		Tensor m_A;
		Tensor m_B;


	};





}