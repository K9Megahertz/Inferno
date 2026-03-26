#include "cudaops.h"


namespace Inferno {

	template <typename AT, typename BT>
	__global__ void  cuda_kernel_embedding(const BT* tptr, const AT* eptr, AT* optr, size_t num_batches, size_t seq_len, size_t embed_dim) {

		const size_t N = num_batches * seq_len * embed_dim;

		const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

		if (linear < N) {

			const size_t tokenidx = linear / embed_dim;
			const size_t tokenid = tptr[tokenidx];
			const size_t offs = linear % embed_dim;
			const size_t eidx = tokenid * embed_dim + offs;

			optr[linear] = eptr[eidx];
		}

		//what Visual Studio Intellicode came up with

		//if (linear < N) {

			//const size_t batch_idx = linear / (seq_len * embed_dim);
			//const size_t seq_idx = (linear / embed_dim) % seq_len;
			//const size_t embed_idx = linear % embed_dim;
			//const size_t token_idx = tptr[batch_idx * seq_len + seq_idx];
			//optr[linear] = eptr[token_idx * embed_dim + embed_idx];

		//}

	}



	template <typename AT, typename BT>
	void cuda_embedding(const BT* tptr, const AT* eptr, AT* optr, size_t num_batches, size_t seq_len, size_t embed_dim) {


		constexpr int threads = 256;
		int blocks = static_cast<int>(((num_batches * seq_len * embed_dim) + threads - 1) / threads);


		cuda_kernel_embedding<AT, BT> << <blocks, threads >> > (tptr, eptr, optr, num_batches, seq_len, embed_dim);
	}


	template void cuda_embedding<int, int>(const int*, const int*, int*, size_t, size_t, size_t);
	template void cuda_embedding<float, int>(const int*, const float*, float*, size_t, size_t, size_t);
	template void cuda_embedding<double, int>(const int*, const double*, double*, size_t, size_t, size_t);

	template void cuda_embedding<int, float>(const float*, const int*, int*, size_t, size_t, size_t);
	template void cuda_embedding<float, float>(const float*, const float*, float*, size_t, size_t, size_t);
	template void cuda_embedding<double, float>(const float*, const double*, double*, size_t, size_t, size_t);

	template void cuda_embedding<int, double>(const double*, const int*, int*, size_t, size_t, size_t);
	template void cuda_embedding<float, double>(const double*, const float*, float*, size_t, size_t, size_t);
	template void cuda_embedding<double, double>(const double*, const double*, double*, size_t, size_t, size_t);



}