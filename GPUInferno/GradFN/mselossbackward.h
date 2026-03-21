#pragma once
#include "node.h"
#include "../tensor.h"
#include "../broadcastops.h"



namespace Inferno {


	class MSELossBackward : public Node {

	public:

		MSELossBackward(const Tensor& A, const Tensor& B);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_A;
		Tensor m_B;


	};

	template <typename AT, typename BT, typename RT>
	void cpu_mse_loss_backward(const AT* aptr, const BT* bptr, RT* gaptr, RT* gbptr, const RT* gout, size_t N) {
		RT upstream = gout[0];
		RT scale = upstream * static_cast<RT>(2) / static_cast<RT>(N);

		for (size_t i = 0; i < N; ++i) {
			RT diff = static_cast<RT>(aptr[i]) - static_cast<RT>(bptr[i]);
			gaptr[i] = scale * diff;
			gbptr[i] = -scale * diff;
		}
	}





}