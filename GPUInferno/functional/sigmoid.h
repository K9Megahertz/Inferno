#pragma once
#include "../util/random.h"
#include "../layers/module.h"
#include "../GradFN/sigmoidbackward.h"



namespace  Inferno {


	class Sigmoid : public Module {

	public:

		Sigmoid(Device device = Device::cpu(), DType dtype = DType::Float32) {}
		Tensor forward(const Tensor& A);

	private:
		


	};



}

