#pragma once
#include "node.h"
#include "../tensor.h"




namespace Inferno {


	class SigmoidBackward : public Node {

	public:

		SigmoidBackward(const Tensor& A, const Tensor& out);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_A;		
		Tensor m_out;


	};

	template<typename AT, typename RT>
	void cpu_sigmoid(const AT* aptr, RT* outptr, size_t N) {
		for (size_t i = 0; i < N; ++i) {
			RT x = static_cast<RT>(aptr[i]);
			outptr[i] = static_cast<RT>(1) / (static_cast<RT>(1) + std::exp(-x));
		}
	}

	template<typename AT, typename GT, typename RT>
	void cpu_sigmoid_backward(
		const AT* yptr,      // saved sigmoid output
		const GT* gptr,      // upstream gradient
		RT* outptr,          // gradient wrt input
		size_t n)
	{
		for (size_t i = 0; i < n; ++i)
		{
			RT y = static_cast<RT>(yptr[i]);
			RT g = static_cast<RT>(gptr[i]);

			outptr[i] = g * y * (static_cast<RT>(1) - y);
		}
	}




}