#pragma once
#include <vector>
#include "tensor.h"



namespace Inferno {	


	Tensor add(const Tensor& A, const Tensor& B);	
	Tensor subtract(const Tensor& A, const Tensor& B);
	Tensor multiply(const Tensor& A, const Tensor& B);
	Tensor divide(const Tensor& A, const Tensor& B);
	Tensor matmul(const Tensor& A, const Tensor& B);	
	Tensor concat(const std::vector<Tensor>& tensors, int axis = 0);
	Tensor select(const Tensor& A, int axis, size_t index);
	Tensor triu(const Tensor& A, int diagonal);	
	Tensor masked_fill(const Tensor& input, const Tensor& mask, float value);


	
}


