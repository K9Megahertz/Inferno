#pragma once
#include<vector>
#include <cuda_runtime.h>
#include <cstddef>
#include <device_launch_parameters.h>
#include "../Util/logger.h"


namespace Inferno {

	template<typename AT, typename BT, typename RT>	
	void cuda_add(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& bshape, const std::vector<size_t>& out_shape, size_t out_numel);

	template<typename AT, typename BT, typename RT>
	void cuda_subtract(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& bshape, const std::vector<size_t>& out_shape, size_t out_numel);

	template<typename AT, typename BT, typename RT>
	void cuda_multiply(const AT* aptr, const BT* bptr, RT* outptr, size_t N);	

	template<typename AT, typename BT, typename RT>
	void cuda_divide(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& ashape, const std::vector<size_t>& bshape, const std::vector<size_t>& out_shape, size_t out_numel);

	template<typename AT>
	void cuda_negate(const AT* aptr, AT* outptr, size_t N);

	template<typename AT>
	void cuda_fill(AT* aptr, const AT value, size_t N);

	template<typename AT, typename BT, typename RT>
	void cuda_matmul(const AT* aptr, const BT* bptr, RT* outptr, const std::vector<size_t>& a_shape, const std::vector<size_t>& a_strides, const std::vector<size_t>& b_shape, const std::vector<size_t>& b_strides, const std::vector<size_t>& out_shape);

	template<typename AT>
	void cuda_sum_to_shape(AT* dst_ptr, const AT* src_ptr, size_t src_numel, size_t src_rank, const std::vector<size_t>& src_shape, const std::vector<size_t>& temp_dst_strides, size_t out_numel);


	inline void check_cuda(cudaError_t err, const char* msg) {
		if (err != cudaSuccess) {			
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, std::string(msg) + ": " + cudaGetErrorString(err));
			exit(1);		
		}
	}

	


}