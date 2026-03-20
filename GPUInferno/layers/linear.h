#pragma once
#include "../util/random.h"
#include "module.h"



namespace  Inferno {


	class Linear : public Module {

	public:

		Linear(size_t in_features, size_t out_features, Device device = Device::cpu(), DType dtype = DType::Float32 );
		Tensor forward(const Tensor& input);

	private:

		Tensor m_weights;
		Tensor m_biases;
		size_t m_in_features;
		size_t m_out_features;


	};



}

