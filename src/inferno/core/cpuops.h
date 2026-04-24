#include <vector>

namespace Inferno {

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
	void cpu_add(const AT* aptr,
		const BT* bptr,
		RT* optr,
		const std::vector<size_t>& ashape,
		const std::vector<size_t>& astrides,
		size_t aoffset,
		const std::vector<size_t>& bshape,
		const std::vector<size_t>& bstrides,
		size_t boffset,
		const std::vector<size_t>& out_shape,
		size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded_shape(out_rank, 1);
		std::vector<size_t> b_padded_shape(out_rank, 1);

		std::vector<size_t> a_padded_strides(out_rank, 1);
		std::vector<size_t> b_padded_strides(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded_shape[a_rank_offset + i] = ashape[i];
			a_padded_strides[a_rank_offset + i] = astrides[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded_shape[b_rank_offset + i] = bshape[i];
			b_padded_strides[b_rank_offset + i] = bstrides[i];
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
			size_t a_offset = aoffset;
			size_t b_offset = boffset;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded_shape[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded_shape[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_padded_strides[d];
				b_offset += b_idx * b_padded_strides[d];
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
	void cpu_subtract(const AT* aptr,
		const BT* bptr,
		RT* optr,
		const std::vector<size_t>& ashape,
		const std::vector<size_t>& astrides,
		size_t aoffset,
		const std::vector<size_t>& bshape,
		const std::vector<size_t>& bstrides,
		size_t boffset,
		const std::vector<size_t>& out_shape,
		size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded_shape(out_rank, 1);
		std::vector<size_t> b_padded_shape(out_rank, 1);

		std::vector<size_t> a_padded_strides(out_rank, 1);
		std::vector<size_t> b_padded_strides(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded_shape[a_rank_offset + i] = ashape[i];
			a_padded_strides[a_rank_offset + i] = astrides[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded_shape[b_rank_offset + i] = bshape[i];
			b_padded_strides[b_rank_offset + i] = bstrides[i];
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
			size_t a_offset = aoffset;
			size_t b_offset = boffset;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded_shape[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded_shape[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_padded_strides[d];
				b_offset += b_idx * b_padded_strides[d];
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
	void cpu_multiply(const AT* aptr,
		const BT* bptr,
		RT* optr,
		const std::vector<size_t>& ashape,
		const std::vector<size_t>& astrides,
		size_t aoffset,
		const std::vector<size_t>& bshape,
		const std::vector<size_t>& bstrides,
		size_t boffset,
		const std::vector<size_t>& out_shape,
		size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded_shape(out_rank, 1);
		std::vector<size_t> b_padded_shape(out_rank, 1);

		std::vector<size_t> a_padded_strides(out_rank, 1);
		std::vector<size_t> b_padded_strides(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded_shape[a_rank_offset + i] = ashape[i];
			a_padded_strides[a_rank_offset + i] = astrides[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded_shape[b_rank_offset + i] = bshape[i];
			b_padded_strides[b_rank_offset + i] = bstrides[i];
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
			size_t a_offset = aoffset;
			size_t b_offset = boffset;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded_shape[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded_shape[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_padded_strides[d];
				b_offset += b_idx * b_padded_strides[d];
			}

			optr[linear] = static_cast<RT>(aptr[a_offset]) * static_cast<RT>(bptr[b_offset]);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_divide
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	template<typename AT, typename BT, typename RT>
	void cpu_divide(const AT* aptr,
		const BT* bptr,
		RT* optr,
		const std::vector<size_t>& ashape,
		const std::vector<size_t>& astrides,
		size_t aoffset,
		const std::vector<size_t>& bshape,
		const std::vector<size_t>& bstrides,
		size_t boffset,
		const std::vector<size_t>& out_shape,
		size_t out_numel)
	{
		const size_t out_rank = out_shape.size();

		// Left-pad A and B shapes to output rank
		std::vector<size_t> a_padded_shape(out_rank, 1);
		std::vector<size_t> b_padded_shape(out_rank, 1);

		std::vector<size_t> a_padded_strides(out_rank, 1);
		std::vector<size_t> b_padded_strides(out_rank, 1);

		size_t a_rank_offset = out_rank - ashape.size();
		size_t b_rank_offset = out_rank - bshape.size();

		for (size_t i = 0; i < ashape.size(); ++i) {
			a_padded_shape[a_rank_offset + i] = ashape[i];
			a_padded_strides[a_rank_offset + i] = astrides[i];
		}

		for (size_t i = 0; i < bshape.size(); ++i) {
			b_padded_shape[b_rank_offset + i] = bshape[i];
			b_padded_strides[b_rank_offset + i] = bstrides[i];
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
			size_t a_offset = aoffset;
			size_t b_offset = boffset;

			for (size_t d = 0; d < out_rank; ++d) {
				size_t a_idx = (a_padded_shape[d] == 1) ? 0 : out_idx[d];
				size_t b_idx = (b_padded_shape[d] == 1) ? 0 : out_idx[d];

				a_offset += a_idx * a_padded_strides[d];
				b_offset += b_idx * b_padded_strides[d];
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
	void cpu_matmul(
		const AT* aptr,
		const BT* bptr,
		RT* optr,
		std::vector<size_t>& a_shape,
		std::vector<size_t>& a_strides,
		std::vector<size_t>& b_shape,
		std::vector<size_t>& b_strides,
		std::vector<size_t>& out_shape) {


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
		size_t total_batches = std::accumulate(out_batch_shape.begin(), out_batch_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());

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



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_mse_loss
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	template<typename AT, typename BT, typename RT>
	void cpu_mse_loss(const AT* a, const BT* b, RT* out, size_t numel) {
		RT sum = static_cast<RT>(0);

		for (size_t i = 0; i < numel; ++i) {
			RT diff = static_cast<RT>(a[i]) - static_cast<RT>(b[i]);
			sum += diff * diff;
		}

		out[0] = sum / static_cast<RT>(numel);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_concat
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
	void cpu_concat(
		const std::vector<const T*>& src_ptrs,
		T* optr,
		const std::vector<size_t>& src_shapes_flat,
		const std::vector<size_t>& src_strides_flat,
		const std::vector<size_t>& src_offsets,
		const std::vector<size_t>& axis_starts,
		const std::vector<size_t>& out_shape,
		const std::vector<size_t>& out_strides,
		size_t out_offset,
		size_t out_numel,
		size_t axis,
		size_t ndim)
	{
		std::vector<size_t> out_idx(ndim, 0);
		std::vector<size_t> src_idx(ndim, 0);

		for (size_t linear = 0; linear < out_numel; ++linear) {

			size_t tmp = linear;
			for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
				out_idx[d] = tmp % out_shape[d];
				tmp /= out_shape[d];
			}

			size_t out_axis_idx = out_idx[axis];

			size_t tensor_idx = 0;
			for (size_t i = 0; i < axis_starts.size(); ++i) {
				size_t start = axis_starts[i];
				size_t end = start + src_shapes_flat[i * ndim + axis];
				if (out_axis_idx >= start && out_axis_idx < end) {
					tensor_idx = i;
					break;
				}
			}

			src_idx = out_idx;
			src_idx[axis] -= axis_starts[tensor_idx];

			size_t dst_storage_idx = out_offset;
			for (size_t d = 0; d < ndim; ++d) {
				dst_storage_idx += out_idx[d] * out_strides[d];
			}

			size_t src_storage_idx = src_offsets[tensor_idx];
			for (size_t d = 0; d < ndim; ++d) {
				src_storage_idx += src_idx[d] * src_strides_flat[tensor_idx * ndim + d];
			}

			optr[dst_storage_idx] = src_ptrs[tensor_idx][src_storage_idx];
		}
	}





	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function unravel_index
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static std::vector<size_t> unravel_index(size_t linear, const std::vector<size_t>& shape) {
		std::vector<size_t> idx(shape.size(), 0);

		for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
			idx[i] = linear % shape[i];
			linear /= shape[i];
		}

		return idx;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function offset_from_index
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	static size_t offset_from_index(
		const std::vector<size_t>& idx,
		const std::vector<size_t>& strides,
		size_t base_offset)
	{
		size_t off = base_offset;
		for (size_t i = 0; i < idx.size(); ++i) {
			off += idx[i] * strides[i];
		}
		return off;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_contiguous_copy
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT>
	void cpu_contiguous_copy(const AT* aptr, AT* optr, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset, size_t N) {

		for (size_t linear = 0; linear < N; ++linear) {
			std::vector<size_t> idx = unravel_index(linear, shape);

			size_t src_off = offset_from_index(idx, strides, offset);

			// dst is contiguous, so linear index matches physical index
			optr[linear] = aptr[src_off];
		}
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_triu
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename AT>
	void cpu_triu(
		const AT* aptr,
		AT* optr,
		const std::vector<size_t>& shape,
		const std::vector<size_t>& strides,
		size_t offset,
		size_t out_numel,
		int diagonal)
	{
		const size_t rank = shape.size();

		std::vector<size_t> idx(rank, 0);

		for (size_t linear = 0; linear < out_numel; ++linear) {

			// unravel linear index → multi-index
			size_t tmp = linear;
			for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
				idx[d] = tmp % shape[d];
				tmp /= shape[d];
			}

			size_t row = idx[rank - 2];
			size_t col = idx[rank - 1];

			// compute storage index using strides + offset
			size_t storage_idx = offset;
			for (size_t d = 0; d < rank; ++d) {
				storage_idx += idx[d] * strides[d];
			}

			if ((int)col - (int)row >= diagonal) {
				optr[linear] = aptr[storage_idx];
			}
			else {
				optr[linear] = static_cast<AT>(0);
			}
		}
	}


	template<typename AT, typename MT>
	void cpu_masked_fill(
		const AT* iptr,
		const MT* mptr,
		AT* optr,
		const std::vector<size_t>& input_shape,
		const std::vector<size_t>& input_strides,
		size_t input_offset,
		const std::vector<size_t>& mask_shape,
		const std::vector<size_t>& mask_strides,
		size_t mask_offset,
		size_t out_numel,
		AT fill_value)
	{
		const size_t out_rank = input_shape.size();

		std::vector<size_t> padded_mask_shape(out_rank, 1);
		std::vector<size_t> padded_mask_strides(out_rank, 0);

		size_t mask_rank_offset = out_rank - mask_shape.size();
		for (size_t i = 0; i < mask_shape.size(); ++i) {
			padded_mask_shape[mask_rank_offset + i] = mask_shape[i];
			padded_mask_strides[mask_rank_offset + i] = mask_strides[i];
		}

		std::vector<size_t> out_idx(out_rank, 0);

		for (size_t linear = 0; linear < out_numel; ++linear) {
			size_t tmp = linear;
			for (int d = static_cast<int>(out_rank) - 1; d >= 0; --d) {
				out_idx[d] = tmp % input_shape[d];
				tmp /= input_shape[d];
			}

			size_t in_storage_idx = input_offset;
			size_t mask_storage_idx = mask_offset;

			for (size_t d = 0; d < out_rank; ++d) {
				in_storage_idx += out_idx[d] * input_strides[d];

				size_t mask_idx = (padded_mask_shape[d] == 1) ? 0 : out_idx[d];
				mask_storage_idx += mask_idx * padded_mask_strides[d];
			}

			optr[linear] = (mptr[mask_storage_idx] != 0) ? fill_value : iptr[in_storage_idx];
		}
	}

	template<typename LT>
	void cpu_cross_entropy_loss(
		const LT* logits,
		const int* targets,
		LT* out,
		size_t rows,
		size_t vocab_size
	) {
		LT total_loss = static_cast<LT>(0);

		for (size_t r = 0; r < rows; r++) {
			const LT* row_ptr = logits + (r * vocab_size);
			int target_id = targets[r];

			if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
					"Target index out of bounds in cpu_cross_entropy_loss");
				exit(1);
			}

			// numerically stable log-sum-exp
			LT max_logit = row_ptr[0];
			for (size_t v = 1; v < vocab_size; v++) {
				if (row_ptr[v] > max_logit) {
					max_logit = row_ptr[v];
				}
			}

			LT sum_exp = static_cast<LT>(0);
			for (size_t v = 0; v < vocab_size; v++) {
				sum_exp += std::exp(row_ptr[v] - max_logit);
			}

			LT log_sum_exp = std::log(sum_exp);
			LT target_logit = row_ptr[target_id];

			// -(target_logit - max_logit - log(sum_exp))
			LT row_loss = -(target_logit - max_logit - log_sum_exp);
			total_loss += row_loss;
		}

		out[0] = total_loss / static_cast<LT>(rows);
	}

}