#include "broadcastops.h"

namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function sum_to_shape
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor sum_to_shape(const Tensor& src, const std::vector<size_t>& target_shape) {

		return dispatchOne(src.dtype(), [&](auto TA) {

			using T = typename decltype(TA)::type;

			const auto& src_shape = GetImpl(src)->shape();
			size_t src_rank = src_shape.size();
			size_t dst_rank = target_shape.size();


			if (dst_rank > src_rank) {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "sum_to_shape: target rank larger than source rank");
				exit(1);
			}

			// Left-pad target shape with 1s so ranks match
			std::vector<size_t> padded_target(src_rank, 1);
			size_t rank_offset = src_rank - dst_rank;

			for (size_t i = 0; i < dst_rank; ++i) {
				padded_target[rank_offset + i] = target_shape[i];
			}

			// Destination logical padded shape
			std::vector<size_t> padded_dst_shape = padded_target;

			// Standard contiguous strides for padded destination shape
			std::vector<size_t> padded_dst_strides(src_rank, 1);
			if (src_rank > 0) {
				padded_dst_strides[src_rank - 1] = 1;
				for (int d = static_cast<int>(src_rank) - 2; d >= 0; --d) {
					padded_dst_strides[d] =
						padded_dst_strides[d + 1] * padded_dst_shape[d + 1];
				}
			}

			// Temporary strides:
			// if this dim is being reduced (target dim == 1 while source dim > 1),
			// set stride to 0 so all indices along that dim collapse to same dst slot.
			std::vector<size_t> temp_dst_strides = padded_dst_strides;

			for (size_t d = 0; d < src_rank; ++d) {
				if (padded_target[d] == 1 && src_shape[d] > 1) {
					temp_dst_strides[d] = 0;
				}
				else if (padded_target[d] != src_shape[d] && padded_target[d] != 1) {
					Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "sum_to_shape: incompatible shapes");
					exit(1);
				}
			}

			Tensor out(src.dtype(), target_shape, "sum_to_shape", src.device());

			const T* src_ptr = GetImpl(src)->data_as_ptr<T>();
			T* dst_ptr = GetImpl(out)->data_as_ptr<T>();

			size_t out_numel = GetImpl(out)->numel();
			size_t src_numel = GetImpl(src)->numel();

			switch (src.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_sum_to_shape(dst_ptr, src_ptr, src_numel, src_rank, src_shape, temp_dst_strides, out_numel);

				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_sum_to_shape(dst_ptr, src_ptr, src_numel, src_rank, src_shape, temp_dst_strides, out_numel);
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			return out;

		});
	}



	Tensor scatter_add(const Tensor& embeddings, const Tensor& token_ids, const Tensor& g_out) {


		if (token_ids.device() != embeddings.device() || g_out.device() != embeddings.device()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "scatter_add_embedding: device mismatch.");
			exit(1);
		}



		const size_t vocab_size = embeddings.shape()[0];
		const size_t embed_dim = embeddings.shape()[1];


		if (g_out.shape().back() != embed_dim) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "scatter_add_embedding: g_out last dim must match embedding dim.");
			exit(1);
		}

		// Number of token positions
		const size_t numtokens = token_ids.numel();

		// g_out should have one embedding vector per token position
		if (g_out.numel() != numtokens * embed_dim) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "scatter_add_embedding: g_out shape does not match token_ids.shape + [embed_dim].");
			exit(1);
		}


		return dispatchTwo(token_ids.dtype(), embeddings.dtype(), [&](auto TagA, auto TagB) {
			using AT = typename decltype(TagA)::type;  //type for tokens
			using BT = typename decltype(TagB)::type;  //type for embeddings and g_out


			Inferno::Tensor out(embeddings.dtype(), embeddings.shape(), "EmbeddingBackward", embeddings.device());


			auto tptr = GetImpl(token_ids)->data_as_ptr<AT>();			
			auto gptr = GetImpl(g_out)->data_as_ptr<BT>();
			auto optr = GetImpl(out)->data_as_ptr<BT>();


			switch (embeddings.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_scatter_add(gptr, tptr, optr, embed_dim, numtokens);

				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_scatter_add(gptr, tptr, optr, embed_dim, numtokens);
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			return out;
			
			
		});


	}


}