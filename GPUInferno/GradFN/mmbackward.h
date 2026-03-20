#pragma once
#include "node.h"
#include "../tensor.h"
#include "../broadcastops.h"
#include "../ops.h"


namespace Inferno {


	class MMBackward : public Node {

	public:

		MMBackward(const Tensor& A, const Tensor& B);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_A;
		Tensor m_B;


	};





}