#pragma once
#include <vector>
#include "tensor.h"



namespace Inferno {	


	Tensor add(Tensor& A, Tensor& B);
	Tensor add_nograd(const Tensor& A, const Tensor& B);
	Tensor subtract(Tensor& A, Tensor& B);
	Tensor multiply(Tensor& A, Tensor& B);
	Tensor matmul(const Tensor& A, const Tensor& B);
	Tensor matmul_impl(const Tensor& A, const Tensor& B);
	Tensor transpose_impl(const Tensor& A, int dima, int dimb);
	Tensor make_view(const Tensor& base, const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, size_t new_storage_offset, const std::string& name);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_add
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename BT, typename RT>
	void cpu_add(const AT* aptr,const BT* bptr,RT* optr,const std::vector<size_t>& ashape,const std::vector<size_t>& bshape,const std::vector<size_t>& out_shape,size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded(out_rank, 1);
		std::vector<size_t> b_padded(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded[a_rank_offset + i] = ashape[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded[b_rank_offset + i] = bshape[i];
		}

		// Build contiguous padded strides for A and B
		std::vector<size_t> a_strides(out_rank, 1);
		std::vector<size_t> b_strides(out_rank, 1);

		if (out_rank > 0) {
			for (int d = static_cast<int>(out_rank) - 2; d >= 0; --d) {
				a_strides[d] = a_strides[d + 1] * a_padded[d + 1];
				b_strides[d] = b_strides[d + 1] * b_padded[d + 1];
			}
		}

		std::vector<size_t> out_idx(out_rank, 0);

		for (size_t linear = 0; linear < out_numel; ++linear) {

			// Convert output linear index -> output multi-index
			size_t tmp = linear;
			for (int d = static_cast<int>(out_rank) - 1; d >= 0; --d) {
				out_idx[d] = tmp % out_shape[d];
				tmp /= out_shape[d];
			}

			// Compute A and B offsets under broadcasting
			size_t a_offset = 0;
			size_t b_offset = 0;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_strides[d];
				b_offset += b_idx * b_strides[d];
			}

			optr[linear] = static_cast<RT>(aptr[a_offset]) + static_cast<RT>(bptr[b_offset]);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_subtract
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename BT, typename RT>
	void cpu_subtract(const AT* aptr, const BT* bptr, RT* optr, const std::vector<size_t>& ashape, const std::vector<size_t>& bshape, const std::vector<size_t>& out_shape, size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded(out_rank, 1);
		std::vector<size_t> b_padded(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded[a_rank_offset + i] = ashape[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded[b_rank_offset + i] = bshape[i];
		}

		// Build contiguous padded strides for A and B
		std::vector<size_t> a_strides(out_rank, 1);
		std::vector<size_t> b_strides(out_rank, 1);

		if (out_rank > 0) {
			for (int d = static_cast<int>(out_rank) - 2; d >= 0; --d) {
				a_strides[d] = a_strides[d + 1] * a_padded[d + 1];
				b_strides[d] = b_strides[d + 1] * b_padded[d + 1];
			}
		}

		std::vector<size_t> out_idx(out_rank, 0);

		for (size_t linear = 0; linear < out_numel; ++linear) {

			// Convert output linear index -> output multi-index
			size_t tmp = linear;
			for (int d = static_cast<int>(out_rank) - 1; d >= 0; --d) {
				out_idx[d] = tmp % out_shape[d];
				tmp /= out_shape[d];
			}

			// Compute A and B offsets under broadcasting
			size_t a_offset = 0;
			size_t b_offset = 0;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_strides[d];
				b_offset += b_idx * b_strides[d];
			}

			optr[linear] = static_cast<RT>(aptr[a_offset]) - static_cast<RT>(bptr[b_offset]);
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_multiply
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename BT, typename RT>
	void cpu_multiply(const AT* aptr, const BT* bptr, RT* optr, const std::vector<size_t>& ashape, const std::vector<size_t>& bshape, const std::vector<size_t>& out_shape, size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded(out_rank, 1);
		std::vector<size_t> b_padded(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded[a_rank_offset + i] = ashape[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded[b_rank_offset + i] = bshape[i];
		}

		// Build contiguous padded strides for A and B
		std::vector<size_t> a_strides(out_rank, 1);
		std::vector<size_t> b_strides(out_rank, 1);

		if (out_rank > 0) {
			for (int d = static_cast<int>(out_rank) - 2; d >= 0; --d) {
				a_strides[d] = a_strides[d + 1] * a_padded[d + 1];
				b_strides[d] = b_strides[d + 1] * b_padded[d + 1];
			}
		}

		std::vector<size_t> out_idx(out_rank, 0);

		for (size_t linear = 0; linear < out_numel; ++linear) {

			// Convert output linear index -> output multi-index
			size_t tmp = linear;
			for (int d = static_cast<int>(out_rank) - 1; d >= 0; --d) {
				out_idx[d] = tmp % out_shape[d];
				tmp /= out_shape[d];
			}

			// Compute A and B offsets under broadcasting
			size_t a_offset = 0;
			size_t b_offset = 0;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_strides[d];
				b_offset += b_idx * b_strides[d];
			}

			optr[linear] = static_cast<RT>(aptr[a_offset]) * static_cast<RT>(bptr[b_offset]);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_add
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT, typename BT, typename RT>
	void cpu_divide(const AT* aptr, const BT* bptr, RT* optr, const std::vector<size_t>& ashape, const std::vector<size_t>& bshape, const std::vector<size_t>& out_shape, size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded(out_rank, 1);
		std::vector<size_t> b_padded(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded[a_rank_offset + i] = ashape[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded[b_rank_offset + i] = bshape[i];
		}

		// Build contiguous padded strides for A and B
		std::vector<size_t> a_strides(out_rank, 1);
		std::vector<size_t> b_strides(out_rank, 1);

		if (out_rank > 0) {
			for (int d = static_cast<int>(out_rank) - 2; d >= 0; --d) {
				a_strides[d] = a_strides[d + 1] * a_padded[d + 1];
				b_strides[d] = b_strides[d + 1] * b_padded[d + 1];
			}
		}

		std::vector<size_t> out_idx(out_rank, 0);

		for (size_t linear = 0; linear < out_numel; ++linear) {

			// Convert output linear index -> output multi-index
			size_t tmp = linear;
			for (int d = static_cast<int>(out_rank) - 1; d >= 0; --d) {
				out_idx[d] = tmp % out_shape[d];
				tmp /= out_shape[d];
			}

			// Compute A and B offsets under broadcasting
			size_t a_offset = 0;
			size_t b_offset = 0;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_strides[d];
				b_offset += b_idx * b_strides[d];
			}

			optr[linear] = static_cast<RT>(aptr[a_offset]) / static_cast<RT>(bptr[b_offset]);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_negate
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT>
	void cpu_negate(const AT* aptr, AT* optr, size_t numel)
	{
		for (size_t i = 0; i < numel; i++) {
			optr[i] = -aptr[i];		
		}
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_matmul
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename AT, typename BT, typename RT>	
	//void cpu_matmul(const AT *aptr, const BT *bptr, RT *optr, size_t M, size_t N, size_t K) {
	void cpu_matmul(const AT* aptr, const BT* bptr, RT* optr, std::vector<size_t>& a_shape, std::vector<size_t>& a_strides, std::vector<size_t>& b_shape, std::vector<size_t>& b_strides, std::vector<size_t>& out_shape) {


		//At this point every thing has been appropriately padded, shapes, strides, etc...	

		size_t a_rank = a_shape.size();
		size_t b_rank = b_shape.size();
		size_t out_rank = out_shape.size();

		if (a_rank < 2 || b_rank < 2 || out_rank < 2) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
				"cpu_matmul requires rank >= 2 tensors");
			exit(1);
		}

		size_t M = a_shape[a_rank - 2];
		size_t N = b_shape[b_rank - 1];
		size_t K = a_shape[a_rank - 1];
		

		// Batch portions only, strip off all the indices but the last two (which are used for the matrix)
		std::vector<size_t> a_batch_shape(a_shape.begin(), a_shape.end() - 2);
		std::vector<size_t> b_batch_shape(b_shape.begin(), b_shape.end() - 2);
		std::vector<size_t> out_batch_shape(out_shape.begin(), out_shape.end() - 2);

		size_t batch_rank = out_batch_shape.size();

		// Size of one matrix per batch
		size_t a_matrix_size = M * K;
		size_t b_matrix_size = K * N;
		size_t out_matrix_size = M * N;

		// Total number of broadcasted batches
		size_t total_batches = std::accumulate(out_batch_shape.begin(), out_batch_shape.end(), 1, std::multiplies<size_t>());
		
		//this is to store the batch index of the current batch we are on  i.e. {0,0} or {0,1,2} or {1,1} etc...
		std::vector<size_t> batch_idx(batch_rank, 0);

		for (size_t batch_linear = 0; batch_linear < total_batches; batch_linear++) {

			// Convert batch linear index -> batch multi-index a.k.a what batch are we on?
			size_t tmp = batch_linear;
			for (int d = static_cast<int>(batch_rank) - 1; d >= 0; --d) {
				batch_idx[d] = tmp % out_batch_shape[d];
				tmp /= out_batch_shape[d];
			}

			// Map broadcasted batch index back to A and B batch offsets
			size_t a_batch_offset = 0;
			size_t b_batch_offset = 0;


			//convert from the output batch indices to the A and B shape indices
			for (size_t d = 0; d < batch_rank; ++d) {
				 size_t a_idx = (a_batch_shape[d] == 1) ? 0 : batch_idx[d];
				 size_t b_idx = (b_batch_shape[d] == 1) ? 0 : batch_idx[d];

				a_batch_offset += a_idx * a_strides[d];
				b_batch_offset += b_idx * b_strides[d];
			}

			// Convert batch offsets into base offsets for the actual matrices
			size_t a_base = a_batch_offset;// *a_matrix_size;
			size_t b_base = b_batch_offset;// *b_matrix_size;
			 size_t o_base = batch_linear * out_matrix_size;

			// Standard matrix multiply for this batch
			for (size_t m = 0; m < M; ++m) {
				for (size_t n = 0; n < N; ++n) {
					RT sum = static_cast<RT>(0);

					for (size_t k = 0; k < K; k++) {

						size_t a_idx = m * a_strides[a_rank - 2] + k * a_strides[a_rank - 1];
						size_t b_idx = k * b_strides[b_rank - 2] + n * b_strides[b_rank - 1];

						size_t a_offset = a_base + a_idx;
						size_t b_offset = b_base + b_idx;

						RT aval = static_cast<RT>(aptr[a_offset]);
						RT bval = static_cast<RT>(bptr[b_offset]);

						sum += static_cast<RT>(aptr[a_offset]) * static_cast<RT>(bptr[b_offset]);
					}			

					optr[o_base + m * N + n] = sum;
				}
			}
		}		
	}
}


