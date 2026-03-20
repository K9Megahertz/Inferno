#pragma once
#include "node.h"
#include "../tensor.h"
#include "../ops.h"


namespace Inferno {


	class TransposeBackward : public Node {

	public:

		TransposeBackward(const Tensor& A, int dima, int dimb);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;

		Tensor m_A;
		int m_dima;
		int m_dimb;


	};





}