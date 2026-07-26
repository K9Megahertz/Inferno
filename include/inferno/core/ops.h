#pragma once
#include <vector>
#include "tensor.h"



namespace Inferno {	


	Tensor add(const Tensor& A, const Tensor& B);	
	Tensor subtract(const Tensor& A, const Tensor& B);
	Tensor multiply(const Tensor& A, const Tensor& B);
	Tensor divide(const Tensor& A, const Tensor& B);
	Tensor matmul(const Tensor& A, const Tensor& B, std::string label = "unlabeled matmul", bool transA = false, bool transB = false);
	Tensor matmul_nt(const Tensor& A, const Tensor& B, std::string label);
	Tensor matmul_tn(const Tensor& A, const Tensor& B, std::string label);
	Tensor concat(const std::vector<Tensor>& tensors, int axis = 0);
	Tensor select(const Tensor& A, int axis, size_t index);
	Tensor triu(const Tensor& A, int diagonal);	
	Tensor masked_fill(const Tensor& input, const Tensor& mask, float value);
	Tensor flash_attention_simple_forward(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal);
	Tensor flash_attention_bigdaddy_forward(const Tensor& qkv, size_t num_heads, bool causal);
	Tensor flash_attention_bigdaddy_forward_my_version_check (const Tensor& qkv, size_t num_heads, bool causal);


	
}


