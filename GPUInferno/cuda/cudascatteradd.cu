#include "cudaops.h"
#include "gpu_atomic.cuh"



namespace Inferno {




	template <typename AT, typename BT>
	__global__ void cuda_kernel_scatter_add(const BT* gptr, const AT* tptr, BT* eptr, size_t embed_dim, size_t numtokens) {


		size_t N = embed_dim * numtokens;
		
		size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

		if (idx < N) {

			size_t tokenidx = idx / embed_dim;
			size_t offs = idx % embed_dim;
			size_t tokenid = tptr[tokenidx];

			size_t eidx = tokenid * embed_dim + offs;
			size_t gidx = tokenidx * embed_dim + offs;						

			gpu_atomic_add(&eptr[eidx], gptr[gidx]);
			 

		}

	}



	template <typename AT, typename BT>
	void cuda_scatter_add(const BT* gptr, const AT* tptr, BT* eptr, size_t embed_dim, size_t numtokens) {


		constexpr int threads = 256;
		int blocks = static_cast<int>(((embed_dim * numtokens) + threads - 1) / threads);


		//cuda_kernel_scatter_add<AT, BT> << <blocks, threads >> > (gptr, tptr, eptr, embed_dim, numtokens);
	}


	template void cuda_scatter_add<int, int>(const int*, const int*, int*, size_t, size_t);
	template void cuda_scatter_add<int, float>(const float*, const int*, float*, size_t, size_t);
	template void cuda_scatter_add<int, double>(const double*, const int*, double*, size_t, size_t);

	template void cuda_scatter_add<float, int>(const int*, const float*, int*, size_t, size_t);
	template void cuda_scatter_add<float, float>(const float*, const float*, float*, size_t, size_t);
	template void cuda_scatter_add<float, double>(const double*, const float*, double*, size_t, size_t);

	template void cuda_scatter_add<double, int>(const int*, const double*, int*, size_t, size_t);
	template void cuda_scatter_add<double, float>(const float*, const double*, float*, size_t, size_t);
	template void cuda_scatter_add<double, double>(const double*, const double*, double*, size_t, size_t);


}