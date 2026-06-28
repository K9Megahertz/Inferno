#include "cudaops.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <type_traits>

namespace Inferno {

	template<typename T>
	__device__ inline T flash_exp(T x);

	template<>
	__device__ inline float flash_exp<float>(float x) {
		return expf(x);
	}

	template<>
	__device__ inline double flash_exp<double>(double x) {
		return exp(x);
	}

	template<typename T>
	__device__ inline T flash_sqrt(T x);

	template<>
	__device__ inline float flash_sqrt<float>(float x) {
		return sqrtf(x);
	}

	template<>
	__device__ inline double flash_sqrt<double>(double x) {
		return sqrt(x);
	}


	template<typename A>
	__device__ inline A warp_sum(A v) {
		for (int offset = 16; offset > 0; offset >>= 1) {
			v += __shfl_down_sync(0xffffffff, v, offset);
		}
		return v;
	}

	template<typename A>
	__device__ inline A neg_inf();

	template<>
	__device__ inline float neg_inf<float>() {
		return -FLT_MAX;
	}

	template<>
	__device__ inline double neg_inf<double>() {
		return -DBL_MAX;
	}

	template<typename A>
	__device__ inline A my_exp(A x);

	template<>
	__device__ inline float my_exp<float>(float x) {
		return expf(x);
	}

	template<>
	__device__ inline double my_exp<double>(double x) {
		return exp(x);
	}

	template<typename T, typename A>
	__device__ inline void atomic_add_t(T* ptr, A value) {
		atomicAdd(ptr, static_cast<T>(value));
	}

	template<typename T, typename A, int BLOCK_M>
	__global__ void flash_attention_warp_forward_kernel(
		const T* __restrict__ qptr,
		const T* __restrict__ kptr,
		const T* __restrict__ vptr,
		T* __restrict__ optr,
		size_t B,
		size_t H,
		size_t Tseq,
		size_t D,
		A scale,
		bool causal
	) {
		int lane = threadIdx.x;        // 0..31
		int row_in_block = threadIdx.y;

		size_t q_row = blockIdx.x * BLOCK_M + row_in_block;
		size_t bh = blockIdx.y;

		if (q_row >= Tseq) {
			return;
		}

		size_t b = bh / H;
		size_t h = bh % H;

		size_t base = ((b * H + h) * Tseq) * D;

		const T* q_row_ptr = qptr + base + q_row * D;

		extern __shared__ unsigned char smem[];
		A* scores_all = reinterpret_cast<A*>(smem);

		// Each query row gets:
		//   Tseq score slots
		//   1 extra slot for row_sum
		A* scores = scores_all + row_in_block * (Tseq + 1);
		A* row_sum_ptr = scores + Tseq;

		////////////////////////////////////////////////////////////////////////
		// 1. Compute scores[j] = dot(Q[row], K[j]) * scale
		//
		// One warp handles one query row.
		// The 32 lanes split the D dimension.
		////////////////////////////////////////////////////////////////////////

		A row_max = neg_inf<A>();

		for (size_t j = 0; j < Tseq; j++) {
			const T* k_row_ptr = kptr + base + j * D;

			A partial = static_cast<A>(0);

			for (size_t d = lane; d < D; d += 32) {
				partial += static_cast<A>(q_row_ptr[d]) * static_cast<A>(k_row_ptr[d]);
			}

			A dot = warp_sum(partial);

			if (lane == 0) {
				dot *= scale;

				if (causal && j > q_row) {
					dot = neg_inf<A>();
				}

				scores[j] = dot;

				if (dot > row_max) {
					row_max = dot;
				}
			}
		}

		////////////////////////////////////////////////////////////////////////
		// 2. Softmax
		////////////////////////////////////////////////////////////////////////

		A row_sum = static_cast<A>(0);

		if (lane == 0) {
			for (size_t j = 0; j < Tseq; j++) {
				A e = my_exp(scores[j] - row_max);
				scores[j] = e;
				row_sum += e;
			}

			*row_sum_ptr = row_sum;
		}

		__syncwarp();

		row_sum = *row_sum_ptr;

		////////////////////////////////////////////////////////////////////////
		// 3. O[row, d] = sum_j softmax[j] * V[j, d]
		//
		// The 32 lanes split the D output dimension.
		////////////////////////////////////////////////////////////////////////

		for (size_t d = lane; d < D; d += 32) {
			A out = static_cast<A>(0);

			for (size_t j = 0; j < Tseq; j++) {
				A p = scores[j] / row_sum;
				out += p * static_cast<A>(vptr[base + j * D + d]);
			}

			optr[base + q_row * D + d] = static_cast<T>(out);
		}
	}

	template<typename T>
	void cuda_flash_attention_simple_forward(
		const T* qptr,
		const T* kptr,
		const T* vptr,
		T* optr,
		size_t B,
		size_t H,
		size_t Tseq,
		size_t D,
		bool causal
	) {
		if constexpr (std::is_same<T, float>::value) {

			constexpr int BLOCK_M = 8;

			dim3 threads(32, BLOCK_M, 1);

			dim3 blocks(
				static_cast<unsigned int>((Tseq + BLOCK_M - 1) / BLOCK_M),
				static_cast<unsigned int>(B * H),
				1
			);

			float scale = 1.0f / std::sqrt(static_cast<float>(D));

			size_t shared_bytes = BLOCK_M * (Tseq + 1) * sizeof(float);

			flash_attention_warp_forward_kernel<float, float, BLOCK_M>
				<< <blocks, threads, shared_bytes >> > (
					qptr,
					kptr,
					vptr,
					optr,
					B,
					H,
					Tseq,
					D,
					scale,
					causal
					);
		}
		else if constexpr (std::is_same<T, double>::value) {

			constexpr int BLOCK_M = 4;

			dim3 threads(32, BLOCK_M, 1);

			dim3 blocks(
				static_cast<unsigned int>((Tseq + BLOCK_M - 1) / BLOCK_M),
				static_cast<unsigned int>(B * H),
				1
			);

			double scale = 1.0 / std::sqrt(static_cast<double>(D));

			size_t shared_bytes = BLOCK_M * (Tseq + 1) * sizeof(double);

			flash_attention_warp_forward_kernel<double, double, BLOCK_M>
				<< <blocks, threads, shared_bytes >> > (
					qptr,
					kptr,
					vptr,
					optr,
					B,
					H,
					Tseq,
					D,
					scale,
					causal
					);
		}
		else {
			INFERNO_LOG_ERROR() << "cuda_flash_attention_simple_forward only supports float and double" << std::endl;
			exit(1);
		}

		check_cuda(cudaGetLastError(), "cuda_flash_attention_simple_forward kernel launch failed");

		//check_cuda(cudaDeviceSynchronize(), "cuda_flash_attention_simple_forward kernel execution failed");
	}

	template void cuda_flash_attention_simple_forward<float>(
		const float*,
		const float*,
		const float*,
		float*,
		size_t,
		size_t,
		size_t,
		size_t,
		bool
	);

	template void cuda_flash_attention_simple_forward<double>(
		const double*,
		const double*,
		const double*,
		double*,
		size_t,
		size_t,
		size_t,
		size_t,
		bool
	);






	template<typename T, typename A, int BLOCK_M, int MAX_D_PER_LANE>
	__global__ void flash_attention_streaming_forward_kernel(
		const T* __restrict__ qptr,
		const T* __restrict__ kptr,
		const T* __restrict__ vptr,
		T* __restrict__ optr,
		size_t B,
		size_t H,
		size_t Tseq,
		size_t D,
		A scale,
		bool causal
	) {
		int lane = threadIdx.x;       // 0..31
		int row_in_block = threadIdx.y;

		size_t q_row = blockIdx.x * BLOCK_M + row_in_block;
		size_t bh = blockIdx.y;

		if (q_row >= Tseq) {
			return;
		}

		size_t b = bh / H;
		size_t h = bh % H;

		size_t base = ((b * H + h) * Tseq) * D;

		const T* q_row_ptr = qptr + base + q_row * D;

		A m = neg_inf<A>();
		A l = static_cast<A>(0);

		A out_accum[MAX_D_PER_LANE];

#pragma unroll
		for (int i = 0; i < MAX_D_PER_LANE; i++) {
			out_accum[i] = static_cast<A>(0);
		}

		////////////////////////////////////////////////////////////////////////
		// Online softmax:
		//
		// Instead of storing scores[T], we stream through K/V once:
		//
		// m_new = max(m_old, score)
		// alpha = exp(m_old - m_new)
		// p     = exp(score - m_new)
		//
		// l = l * alpha + p
		// O = O * alpha + p * V
		////////////////////////////////////////////////////////////////////////

		for (size_t j = 0; j < Tseq; j++) {
			const T* k_row_ptr = kptr + base + j * D;

			A partial = static_cast<A>(0);

			for (size_t d = lane; d < D; d += 32) {
				partial += static_cast<A>(q_row_ptr[d]) * static_cast<A>(k_row_ptr[d]);
			}

			A dot = warp_sum(partial);

			if (lane == 0) {
				dot *= scale;

				if (causal && j > q_row) {
					dot = neg_inf<A>();
				}
			}

			dot = __shfl_sync(0xffffffff, dot, 0);

			A m_new = dot > m ? dot : m;

			A alpha;

			if (m == neg_inf<A>()) {
				alpha = static_cast<A>(0);
			}
			else {
				alpha = my_exp(m - m_new);
			}

			A p = my_exp(dot - m_new);

			for (int i = 0; i < MAX_D_PER_LANE; i++) {
				size_t d = static_cast<size_t>(lane) + static_cast<size_t>(i) * 32;

				if (d < D) {
					A v = static_cast<A>(vptr[base + j * D + d]);
					out_accum[i] = out_accum[i] * alpha + p * v;
				}
			}

			l = l * alpha + p;
			m = m_new;
		}

		////////////////////////////////////////////////////////////////////////
		// Normalize final output
		////////////////////////////////////////////////////////////////////////

		for (int i = 0; i < MAX_D_PER_LANE; i++) {
			size_t d = static_cast<size_t>(lane) + static_cast<size_t>(i) * 32;

			if (d < D) {
				A out = out_accum[i] / l;
				optr[base + q_row * D + d] = static_cast<T>(out);
			}
		}
	}


	template<typename T>
	void cuda_flash_attention_simple_forward2(
		const T* qptr,
		const T* kptr,
		const T* vptr,
		T* optr,
		size_t B,
		size_t H,
		size_t Tseq,
		size_t D,
		bool causal
	) {
		if (D > 128) {
			INFERNO_LOG_ERROR() << "cuda_flash_attention_simple_forward currently supports D <= 128" << std::endl;
			exit(1);
		}

		if constexpr (std::is_same<T, float>::value) {

			constexpr int BLOCK_M = 8;
			constexpr int MAX_D_PER_LANE = 4;

			dim3 threads(32, BLOCK_M, 1);

			dim3 blocks(
				static_cast<unsigned int>((Tseq + BLOCK_M - 1) / BLOCK_M),
				static_cast<unsigned int>(B * H),
				1
			);

			float scale = 1.0f / std::sqrt(static_cast<float>(D));

			flash_attention_streaming_forward_kernel<float, float, BLOCK_M, MAX_D_PER_LANE>
				<< <blocks, threads >> > (
					qptr,
					kptr,
					vptr,
					optr,
					B,
					H,
					Tseq,
					D,
					scale,
					causal
					);
		}
		else if constexpr (std::is_same<T, double>::value) {

			constexpr int BLOCK_M = 4;
			constexpr int MAX_D_PER_LANE = 4;

			dim3 threads(32, BLOCK_M, 1);

			dim3 blocks(
				static_cast<unsigned int>((Tseq + BLOCK_M - 1) / BLOCK_M),
				static_cast<unsigned int>(B * H),
				1
			);

			double scale = 1.0 / std::sqrt(static_cast<double>(D));

			flash_attention_streaming_forward_kernel<double, double, BLOCK_M, MAX_D_PER_LANE>
				<< <blocks, threads >> > (
					qptr,
					kptr,
					vptr,
					optr,
					B,
					H,
					Tseq,
					D,
					scale,
					causal
					);
		}
		else {
			INFERNO_LOG_ERROR() << "cuda_flash_attention_simple_forward only supports float and double" << std::endl;
			exit(1);
		}

		check_cuda(cudaGetLastError(), "cuda_flash_attention_simple_forward kernel launch failed");

		//check_cuda(cudaDeviceSynchronize(), "cuda_flash_attention_simple_forward kernel execution failed");
	}


	template void cuda_flash_attention_simple_forward2<float>(
		const float*,
		const float*,
		const float*,
		float*,
		size_t,
		size_t,
		size_t,
		size_t,
		bool
	);

	template void cuda_flash_attention_simple_forward2<double>(
		const double*,
		const double*,
		const double*,
		double*,
		size_t,
		size_t,
		size_t,
		size_t,
		bool
	);







	//Backward

	template<typename T, typename A, int BLOCK_M, int MAX_D_PER_LANE>
	__global__ void flash_attention_backward_kernel(
		const T* __restrict__ qptr,
		const T* __restrict__ kptr,
		const T* __restrict__ vptr,
		const T* __restrict__ optr,
		const T* __restrict__ goptr,
		T* __restrict__ dqptr,
		T* __restrict__ dkptr,
		T* __restrict__ dvptr,
		size_t B,
		size_t H,
		size_t Tseq,
		size_t D,
		A scale,
		bool causal
	) {
		int lane = threadIdx.x;
		int row_in_block = threadIdx.y;

		size_t q_row = blockIdx.x * BLOCK_M + row_in_block;
		size_t bh = blockIdx.y;

		if (q_row >= Tseq) {
			return;
		}

		size_t b = bh / H;
		size_t h = bh % H;

		size_t base = ((b * H + h) * Tseq) * D;

		const T* q_row_ptr = qptr + base + q_row * D;
		const T* o_row_ptr = optr + base + q_row * D;
		const T* go_row_ptr = goptr + base + q_row * D;

		////////////////////////////////////////////////////////////////////////
		// 1. Recompute softmax stats m and l for this query row
		////////////////////////////////////////////////////////////////////////

		A m = neg_inf<A>();
		A l = static_cast<A>(0);

		for (size_t j = 0; j < Tseq; j++) {
			const T* k_row_ptr = kptr + base + j * D;

			A partial = static_cast<A>(0);

			for (size_t d = lane; d < D; d += 32) {
				partial += static_cast<A>(q_row_ptr[d]) * static_cast<A>(k_row_ptr[d]);
			}

			A dot = warp_sum(partial);

			if (lane == 0) {
				dot *= scale;

				if (causal && j > q_row) {
					dot = neg_inf<A>();
				}

				A m_new = dot > m ? dot : m;

				A alpha;
				if (m == neg_inf<A>()) {
					alpha = static_cast<A>(0);
				}
				else {
					alpha = my_exp(m - m_new);
				}

				A p = my_exp(dot - m_new);

				l = l * alpha + p;
				m = m_new;
			}

			m = __shfl_sync(0xffffffff, m, 0);
			l = __shfl_sync(0xffffffff, l, 0);
		}

		////////////////////////////////////////////////////////////////////////
		// 2. Compute delta = sum_d dO[d] * O[d]
		//
		// This is used by softmax backward:
		// dS_j = P_j * (dP_j - delta)
		////////////////////////////////////////////////////////////////////////

		A delta_partial = static_cast<A>(0);

		for (size_t d = lane; d < D; d += 32) {
			delta_partial +=
				static_cast<A>(go_row_ptr[d]) *
				static_cast<A>(o_row_ptr[d]);
		}

		A delta = warp_sum(delta_partial);
		delta = __shfl_sync(0xffffffff, delta, 0);

		////////////////////////////////////////////////////////////////////////
		// 3. Stream over keys again and accumulate dQ, dK, dV
		////////////////////////////////////////////////////////////////////////

		A dq_accum[MAX_D_PER_LANE];

#pragma unroll
		for (int i = 0; i < MAX_D_PER_LANE; i++) {
			dq_accum[i] = static_cast<A>(0);
		}

		for (size_t j = 0; j < Tseq; j++) {
			const T* k_row_ptr = kptr + base + j * D;
			const T* v_row_ptr = vptr + base + j * D;

			//////////////////////////////////////////////////////////////////////
			// Recompute score and probability P_j
			//////////////////////////////////////////////////////////////////////

			A partial_score = static_cast<A>(0);

			for (size_t d = lane; d < D; d += 32) {
				partial_score +=
					static_cast<A>(q_row_ptr[d]) *
					static_cast<A>(k_row_ptr[d]);
			}

			A dot = warp_sum(partial_score);

			if (lane == 0) {
				dot *= scale;

				if (causal && j > q_row) {
					dot = neg_inf<A>();
				}
			}

			dot = __shfl_sync(0xffffffff, dot, 0);

			A p = my_exp(dot - m) / l;

			//////////////////////////////////////////////////////////////////////
			// dP_j = sum_d dO[d] * V[j,d]
			//////////////////////////////////////////////////////////////////////

			A dp_partial = static_cast<A>(0);

			for (size_t d = lane; d < D; d += 32) {
				dp_partial +=
					static_cast<A>(go_row_ptr[d]) *
					static_cast<A>(v_row_ptr[d]);
			}

			A dp = warp_sum(dp_partial);
			dp = __shfl_sync(0xffffffff, dp, 0);

			//////////////////////////////////////////////////////////////////////
			// dS_j = P_j * (dP_j - delta)
			// score included scale, so dQ/dK get another scale factor
			//////////////////////////////////////////////////////////////////////

			A ds = p * (dp - delta);
			A ds_scaled = ds * scale;

			//////////////////////////////////////////////////////////////////////
			// dQ += dS_j * K_j
			// dK_j += dS_j * Q
			// dV_j += P_j * dO
			//////////////////////////////////////////////////////////////////////

			for (int i = 0; i < MAX_D_PER_LANE; i++) {
				size_t d = static_cast<size_t>(lane) + static_cast<size_t>(i) * 32;

				if (d < D) {
					A qv = static_cast<A>(q_row_ptr[d]);
					A kv = static_cast<A>(k_row_ptr[d]);
					A gov = static_cast<A>(go_row_ptr[d]);

					dq_accum[i] += ds_scaled * kv;

					atomic_add_t<T, A>(
						dkptr + base + j * D + d,
						ds_scaled * qv
					);

					atomic_add_t<T, A>(
						dvptr + base + j * D + d,
						p * gov
					);
				}
			}
		}

		////////////////////////////////////////////////////////////////////////
		// 4. Store dQ
		////////////////////////////////////////////////////////////////////////

		for (int i = 0; i < MAX_D_PER_LANE; i++) {
			size_t d = static_cast<size_t>(lane) + static_cast<size_t>(i) * 32;

			if (d < D) {
				dqptr[base + q_row * D + d] = static_cast<T>(dq_accum[i]);
			}
		}
	}


	template<typename T>
	void cuda_flash_attention_simple_backward(
		const T* qptr,
		const T* kptr,
		const T* vptr,
		const T* optr,
		const T* goptr,
		T* dqptr,
		T* dkptr,
		T* dvptr,
		size_t B,
		size_t H,
		size_t Tseq,
		size_t D,
		bool causal
	) {
		if (D > 128) {
			INFERNO_LOG_ERROR() << "cuda_flash_attention_simple_backward2 currently supports D <= 128" << std::endl;
			exit(1);
		}

		size_t numel = B * H * Tseq * D;

		check_cuda(cudaMemset(dqptr, 0, numel * sizeof(T)), "cudaMemset dq failed");
		check_cuda(cudaMemset(dkptr, 0, numel * sizeof(T)), "cudaMemset dk failed");
		check_cuda(cudaMemset(dvptr, 0, numel * sizeof(T)), "cudaMemset dv failed");

		if constexpr (std::is_same<T, float>::value) {
			constexpr int BLOCK_M = 8;
			constexpr int MAX_D_PER_LANE = 4;

			dim3 threads(32, BLOCK_M, 1);

			dim3 blocks(
				static_cast<unsigned int>((Tseq + BLOCK_M - 1) / BLOCK_M),
				static_cast<unsigned int>(B * H),
				1
			);

			float scale = 1.0f / std::sqrt(static_cast<float>(D));

			flash_attention_backward_kernel<float, float, BLOCK_M, MAX_D_PER_LANE>
				<< <blocks, threads >> > (
					qptr,
					kptr,
					vptr,
					optr,
					goptr,
					dqptr,
					dkptr,
					dvptr,
					B,
					H,
					Tseq,
					D,
					scale,
					causal
					);
		}
		else if constexpr (std::is_same<T, double>::value) {
			constexpr int BLOCK_M = 4;
			constexpr int MAX_D_PER_LANE = 4;

			dim3 threads(32, BLOCK_M, 1);

			dim3 blocks(
				static_cast<unsigned int>((Tseq + BLOCK_M - 1) / BLOCK_M),
				static_cast<unsigned int>(B * H),
				1
			);

			double scale = 1.0 / std::sqrt(static_cast<double>(D));

			flash_attention_backward_kernel<double, double, BLOCK_M, MAX_D_PER_LANE>
				<< <blocks, threads >> > (
					qptr,
					kptr,
					vptr,
					optr,
					goptr,
					dqptr,
					dkptr,
					dvptr,
					B,
					H,
					Tseq,
					D,
					scale,
					causal
					);
		}
		else {
			INFERNO_LOG_ERROR() << "cuda_flash_attention_simple_backward2 only supports float and double" << std::endl;
			exit(1);
		}

		check_cuda(cudaGetLastError(), "flash_attention_backward_kernel launch failed");
		//check_cuda(cudaDeviceSynchronize(), "flash_attention_backward_kernel execution failed");
	}


	template void cuda_flash_attention_simple_backward<float>(
		const float*,
		const float*,
		const float*,
		const float*,
		const float*,
		float*,
		float*,
		float*,
		size_t,
		size_t,
		size_t,
		size_t,
		bool
	);

	template void cuda_flash_attention_simple_backward<double>(
		const double*,
		const double*,
		const double*,
		const double*,
		const double*,
		double*,
		double*,
		double*,
		size_t,
		size_t,
		size_t,
		size_t,
		bool
	);




	template <typename T>
	__global__ void flash_attention_bigdaddy_forward_kernel(
		const T* __restrict__ qkv,
		T* __restrict__ out,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

		size_t total = B * Tseq * C;
		if (idx >= total) return;

		size_t d_model = idx % C;
		size_t t = (idx / C) % Tseq;
		size_t b = idx / (Tseq * C);

		size_t h = d_model / D;
		size_t d = d_model % D;

		T scale = static_cast<T>(rsqrtf(static_cast<float>(D)));

		// Compute max score for stable softmax
		T max_score = -INFINITY;

		size_t j_end = causal ? t + 1 : Tseq;

		for (size_t j = 0; j < j_end; j++) {
			T score = static_cast<T>(0);

			for (size_t kd = 0; kd < D; kd++) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + kd;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + kd;

				score += qkv[q_index] * qkv[k_index];
			}

			score *= scale;

			if (score > max_score) {
				max_score = score;
			}
		}

		// Compute denominator
		T denom = static_cast<T>(0);

		for (size_t j = 0; j < j_end; j++) {
			T score = static_cast<T>(0);

			for (size_t kd = 0; kd < D; kd++) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + kd;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + kd;

				score += qkv[q_index] * qkv[k_index];
			}

			score *= scale;
			denom += expf(static_cast<float>(score - max_score));
		}

		// Compute output element
		T result = static_cast<T>(0);

		for (size_t j = 0; j < j_end; j++) {
			T score = static_cast<T>(0);

			for (size_t kd = 0; kd < D; kd++) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + kd;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + kd;

				score += qkv[q_index] * qkv[k_index];
			}

			score *= scale;

			T p = expf(static_cast<float>(score - max_score)) / denom;

			size_t v_index =
				b * Tseq * 3 * C +
				j * 3 * C +
				2 * C +
				h * D + d;

			result += p * qkv[v_index];
		}

		size_t out_index =
			b * Tseq * C +
			t * C +
			d_model;

		out[out_index] = result;
	}

	template <typename T>
	void cuda_flash_attention_bigdaddy_forward(
		const T* qkv,
		T* out,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		size_t total = B * Tseq * C;

		int threads = 256;
		int blocks = static_cast<int>((total + threads - 1) / threads);

		flash_attention_bigdaddy_forward_kernel<T> << <blocks, threads >> > (
			qkv,
			out,
			B,
			Tseq,
			C,
			H,
			D,
			causal
			);

		check_cuda(cudaGetLastError(), "flash_attention_bigdaddy_forward_kernel launch failed");
	}


	template void cuda_flash_attention_bigdaddy_forward<float>(const float*, float*, size_t, size_t, size_t, size_t, size_t, bool);
	template void cuda_flash_attention_bigdaddy_forward<double>(const double*, double*, size_t, size_t, size_t, size_t, size_t, bool);




	template <typename T>
	__global__ void flash_attention_bigdaddy_backward_kernel(
		const T* __restrict__ qkv,
		const T* __restrict__ out,
		const T* __restrict__ gout,
		T* __restrict__ dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		size_t row = blockIdx.x * blockDim.x + threadIdx.x;

		size_t total_rows = B * H * Tseq;
		if (row >= total_rows) return;

		size_t t = row % Tseq;
		size_t h = (row / Tseq) % H;
		size_t b = row / (H * Tseq);

		T scale = static_cast<T>(rsqrtf(static_cast<float>(D)));

		size_t j_end = causal ? t + 1 : Tseq;

		// ------------------------------------------------------------
		// 1. Recompute max score for this row
		// ------------------------------------------------------------
		T max_score = -INFINITY;

		for (size_t j = 0; j < j_end; j++) {
			T score = static_cast<T>(0);

			for (size_t d = 0; d < D; d++) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				score += qkv[q_index] * qkv[k_index];
			}

			score *= scale;

			if (score > max_score) {
				max_score = score;
			}
		}

		// ------------------------------------------------------------
		// 2. Recompute softmax denominator
		// ------------------------------------------------------------
		T denom = static_cast<T>(0);

		for (size_t j = 0; j < j_end; j++) {
			T score = static_cast<T>(0);

			for (size_t d = 0; d < D; d++) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				score += qkv[q_index] * qkv[k_index];
			}

			score *= scale;
			denom += expf(static_cast<float>(score - max_score));
		}

		// ------------------------------------------------------------
		// 3. Compute dot(dO_i, O_i)
		//
		// For softmax backward:
		// dS_ij = P_ij * (dP_ij - sum_k P_ik * dP_ik)
		//
		// Since O_i = sum_j P_ij V_j,
		// sum_j P_ij * dP_ij = dot(dO_i, O_i)
		// ------------------------------------------------------------
		T do_dot_o = static_cast<T>(0);

		for (size_t d = 0; d < D; d++) {
			size_t out_index =
				b * Tseq * C +
				t * C +
				h * D + d;

			do_dot_o += gout[out_index] * out[out_index];
		}

		// ------------------------------------------------------------
		// 4. For each attended token j:
		//
		// dV_j += P_ij * dO_i
		// dS_ij = P_ij * dot(dO_i, V_j - O_i)
		// dQ_i += dS_ij * K_j * scale
		// dK_j += dS_ij * Q_i * scale
		// ------------------------------------------------------------
		for (size_t j = 0; j < j_end; j++) {
			T score = static_cast<T>(0);

			for (size_t d = 0; d < D; d++) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				score += qkv[q_index] * qkv[k_index];
			}

			score *= scale;

			T p = expf(static_cast<float>(score - max_score)) / denom;

			T dscore = static_cast<T>(0);

			for (size_t d = 0; d < D; d++) {
				size_t out_index =
					b * Tseq * C +
					t * C +
					h * D + d;

				size_t v_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					2 * C +
					h * D + d;

				dscore += gout[out_index] * qkv[v_index];
			}

			dscore = p * (dscore - do_dot_o);

			for (size_t d = 0; d < D; d++) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				size_t v_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					2 * C +
					h * D + d;

				size_t out_index =
					b * Tseq * C +
					t * C +
					h * D + d;

				// dV_j += P_ij * dO_i
				atomicAdd(&dqkv[v_index], p * gout[out_index]);

				// dQ_i += dS_ij * K_j * scale
				atomicAdd(&dqkv[q_index], dscore * qkv[k_index] * scale);

				// dK_j += dS_ij * Q_i * scale
				atomicAdd(&dqkv[k_index], dscore * qkv[q_index] * scale);
			}
		}
	}



	template <typename T>
	void cuda_flash_attention_bigdaddy_backward(
		const T* qkv,
		const T* out,
		const T* gout,
		T* dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		size_t dqkv_numel = B * Tseq * 3 * C;

		check_cuda(
			cudaMemset(dqkv, 0, dqkv_numel * sizeof(T)),
			"cuda_flash_attention_bigdaddy_backward cudaMemset failed"
		);

		size_t total_rows = B * H * Tseq;

		int threads = 128;
		int blocks = static_cast<int>((total_rows + threads - 1) / threads);

		flash_attention_bigdaddy_backward_kernel<T> << <blocks, threads >> > (
			qkv,
			out,
			gout,
			dqkv,
			B,
			Tseq,
			C,
			H,
			D,
			causal
			);

		check_cuda(
			cudaGetLastError(),
			"flash_attention_bigdaddy_backward_kernel launch failed"
		);
	}


	template void cuda_flash_attention_bigdaddy_backward<float>(
		const float* qkv,
		const float* out,
		const float* gout,
		float* dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	);

	template void cuda_flash_attention_bigdaddy_backward<double>(
		const double* qkv,
		const double* out,
		const double* gout,
		double* dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	);


	template <typename T, int TILE_N, int DMAX>
	__global__ void flash_attention_bigdaddy_backward_tiled_kernel(
		const T* __restrict__ qkv,
		const T* __restrict__ out,
		const T* __restrict__ gout,
		T* __restrict__ dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		extern __shared__ unsigned char smem_raw[];

		T* shK = reinterpret_cast<T*>(smem_raw);
		T* shV = shK + TILE_N * DMAX;

		size_t row = blockIdx.x;

		size_t t = row % Tseq;
		size_t h = (row / Tseq) % H;
		size_t b = row / (H * Tseq);

		size_t tid = threadIdx.x;

		T scale = static_cast<T>(rsqrtf(static_cast<float>(D)));

		size_t j_limit = causal ? t + 1 : Tseq;

		T dq_local[DMAX];

		for (int d = 0; d < DMAX; d++) {
			dq_local[d] = static_cast<T>(0);
		}

		// ------------------------------------------------------------
		// 1. Compute max score
		// ------------------------------------------------------------
		T thread_max = -INFINITY;

		for (size_t tile_start = 0; tile_start < j_limit; tile_start += TILE_N) {
			size_t tile_size = min(static_cast<size_t>(TILE_N), j_limit - tile_start);

			for (size_t x = tid; x < tile_size * D; x += blockDim.x) {
				size_t j_local = x / D;
				size_t d = x % D;
				size_t j = tile_start + j_local;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				shK[j_local * DMAX + d] = qkv[k_index];
			}

			__syncthreads();

			for (size_t j_local = tid; j_local < tile_size; j_local += blockDim.x) {
				T score = static_cast<T>(0);

				for (size_t d = 0; d < D; d++) {
					size_t q_index =
						b * Tseq * 3 * C +
						t * 3 * C +
						h * D + d;

					score += qkv[q_index] * shK[j_local * DMAX + d];
				}

				score *= scale;

				if (score > thread_max) {
					thread_max = score;
				}
			}

			__syncthreads();
		}

		__shared__ T reduce_buf[256];

		reduce_buf[tid] = thread_max;
		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride) {
				T other = reduce_buf[tid + stride];
				if (other > reduce_buf[tid]) {
					reduce_buf[tid] = other;
				}
			}
			__syncthreads();
		}

		T row_max = reduce_buf[0];

		// ------------------------------------------------------------
		// 2. Compute softmax denominator
		// ------------------------------------------------------------
		T thread_sum = static_cast<T>(0);

		for (size_t tile_start = 0; tile_start < j_limit; tile_start += TILE_N) {
			size_t tile_size = min(static_cast<size_t>(TILE_N), j_limit - tile_start);

			for (size_t x = tid; x < tile_size * D; x += blockDim.x) {
				size_t j_local = x / D;
				size_t d = x % D;
				size_t j = tile_start + j_local;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				shK[j_local * DMAX + d] = qkv[k_index];
			}

			__syncthreads();

			for (size_t j_local = tid; j_local < tile_size; j_local += blockDim.x) {
				T score = static_cast<T>(0);

				for (size_t d = 0; d < D; d++) {
					size_t q_index =
						b * Tseq * 3 * C +
						t * 3 * C +
						h * D + d;

					score += qkv[q_index] * shK[j_local * DMAX + d];
				}

				score *= scale;
				thread_sum += expf(static_cast<float>(score - row_max));
			}

			__syncthreads();
		}

		reduce_buf[tid] = thread_sum;
		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride) {
				reduce_buf[tid] += reduce_buf[tid + stride];
			}
			__syncthreads();
		}

		T denom = reduce_buf[0];

		// ------------------------------------------------------------
		// 3. Compute dot(dO_i, O_i)
		// ------------------------------------------------------------
		T thread_do_dot_o = static_cast<T>(0);

		for (size_t d = tid; d < D; d += blockDim.x) {
			size_t out_index =
				b * Tseq * C +
				t * C +
				h * D + d;

			thread_do_dot_o += gout[out_index] * out[out_index];
		}

		reduce_buf[tid] = thread_do_dot_o;
		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride) {
				reduce_buf[tid] += reduce_buf[tid + stride];
			}
			__syncthreads();
		}

		T do_dot_o = reduce_buf[0];

		// ------------------------------------------------------------
		// 4. Main backward loop over K/V tiles
		// ------------------------------------------------------------
		for (size_t tile_start = 0; tile_start < j_limit; tile_start += TILE_N) {
			size_t tile_size = min(static_cast<size_t>(TILE_N), j_limit - tile_start);

			for (size_t x = tid; x < tile_size * D; x += blockDim.x) {
				size_t j_local = x / D;
				size_t d = x % D;
				size_t j = tile_start + j_local;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				size_t v_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					2 * C +
					h * D + d;

				shK[j_local * DMAX + d] = qkv[k_index];
				shV[j_local * DMAX + d] = qkv[v_index];
			}

			__syncthreads();

			for (size_t j_local = 0; j_local < tile_size; j_local++) {
				size_t j = tile_start + j_local;

				T score = static_cast<T>(0);

				for (size_t d = 0; d < D; d++) {
					size_t q_index =
						b * Tseq * 3 * C +
						t * 3 * C +
						h * D + d;

					score += qkv[q_index] * shK[j_local * DMAX + d];
				}

				score *= scale;

				T p = expf(static_cast<float>(score - row_max)) / denom;

				T thread_dscore_part = static_cast<T>(0);

				for (size_t d = tid; d < D; d += blockDim.x) {
					size_t out_index =
						b * Tseq * C +
						t * C +
						h * D + d;

					thread_dscore_part += gout[out_index] * shV[j_local * DMAX + d];
				}

				reduce_buf[tid] = thread_dscore_part;
				__syncthreads();

				for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
					if (tid < stride) {
						reduce_buf[tid] += reduce_buf[tid + stride];
					}
					__syncthreads();
				}

				T dscore = p * (reduce_buf[0] - do_dot_o);

				for (size_t d = tid; d < D; d += blockDim.x) {
					size_t q_index =
						b * Tseq * 3 * C +
						t * 3 * C +
						h * D + d;

					size_t k_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						C +
						h * D + d;

					size_t v_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						2 * C +
						h * D + d;

					size_t out_index =
						b * Tseq * C +
						t * C +
						h * D + d;

					T go = gout[out_index];

					// dQ local
					dq_local[d] += dscore * shK[j_local * DMAX + d] * scale;

					// dK accumulates from many query rows
					atomicAdd(&dqkv[k_index], dscore * qkv[q_index] * scale);

					// dV accumulates from many query rows
					atomicAdd(&dqkv[v_index], p * go);
				}

				__syncthreads();
			}

			__syncthreads();
		}

		// ------------------------------------------------------------
		// 5. Write dQ once
		// ------------------------------------------------------------
		for (size_t d = tid; d < D; d += blockDim.x) {
			size_t q_index =
				b * Tseq * 3 * C +
				t * 3 * C +
				h * D + d;

			dqkv[q_index] = dq_local[d];
		}
	}


	template <typename T>
	void cuda_flash_attention_bigdaddy_backward_tiled(
		const T* qkv,
		const T* out,
		const T* gout,
		T* dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		size_t dqkv_numel = B * Tseq * 3 * C;

		check_cuda(
			cudaMemset(dqkv, 0, dqkv_numel * sizeof(T)),
			"cuda_flash_attention_bigdaddy_backward_tiled cudaMemset failed"
		);

		constexpr int TILE_N = 32;
		constexpr int DMAX = 128;

		if (D > DMAX) {
			std::cerr << "D exceeds DMAX in flash_attention_bigdaddy_backward_tiled" << std::endl;
			std::exit(1);
		}

		int threads = 128;

		size_t total_rows = B * H * Tseq;
		int blocks = static_cast<int>(total_rows);

		size_t shared_bytes =
			2 * TILE_N * DMAX * sizeof(T); // K tile + V tile

		flash_attention_bigdaddy_backward_tiled_kernel<T, TILE_N, DMAX>
			<< <blocks, threads, shared_bytes >> > (
				qkv,
				out,
				gout,
				dqkv,
				B,
				Tseq,
				C,
				H,
				D,
				causal
				);

		check_cuda(
			cudaGetLastError(),
			"flash_attention_bigdaddy_backward_tiled_kernel launch failed"
		);
	}

	
	template void cuda_flash_attention_bigdaddy_backward_tiled<float>(
		const float* qkv,
		const float* out,
		const float* gout,
		float* dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	);

	template void cuda_flash_attention_bigdaddy_backward_tiled<double>(
		const double* qkv,
		const double* out,
		const double* gout,
		double* dqkv,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	);




	template <typename T, int TILE_N, int DMAX>
	__global__ void flash_attention_bigdaddy_forward_tiled_kernel(
		const T* __restrict__ qkv,
		T* __restrict__ out,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		extern __shared__ unsigned char smem_raw[];

		T* shK = reinterpret_cast<T*>(smem_raw);
		T* shV = shK + TILE_N * DMAX;
		T* shScores = shV + TILE_N * DMAX;
		T* reduce = shScores + TILE_N;

		const size_t row = blockIdx.x;

		const size_t t = row % Tseq;
		const size_t h = (row / Tseq) % H;
		const size_t b = row / (H * Tseq);

		const size_t tid = threadIdx.x;

		const T scale = static_cast<T>(rsqrtf(static_cast<float>(D)));
		const size_t j_limit = causal ? t + 1 : Tseq;

		// ------------------------------------------------------------
		// 1. Compute row max
		// ------------------------------------------------------------
		T row_max = -INFINITY;

		for (size_t tile_start = 0; tile_start < j_limit; tile_start += TILE_N) {
			const size_t tile_size =
				min(static_cast<size_t>(TILE_N), j_limit - tile_start);

			// Load K tile
			for (size_t x = tid; x < tile_size * D; x += blockDim.x) {
				size_t j_local = x / D;
				size_t d = x % D;
				size_t j = tile_start + j_local;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				shK[j_local * DMAX + d] = qkv[k_index];
			}

			__syncthreads();

			for (size_t j_local = 0; j_local < tile_size; j_local++) {
				T partial = static_cast<T>(0);

				for (size_t d = tid; d < D; d += blockDim.x) {
					size_t q_index =
						b * Tseq * 3 * C +
						t * 3 * C +
						h * D + d;

					partial += qkv[q_index] * shK[j_local * DMAX + d];
				}

				reduce[tid] = partial;
				__syncthreads();

				for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
					if (tid < stride) {
						reduce[tid] += reduce[tid + stride];
					}
					__syncthreads();
				}

				if (tid == 0) {
					T score = reduce[0] * scale;
					shScores[j_local] = score;

					if (score > row_max) {
						row_max = score;
					}
				}

				__syncthreads();
			}
		}

		// Broadcast row_max from thread 0
		reduce[0] = row_max;
		__syncthreads();
		row_max = reduce[0];

		// ------------------------------------------------------------
		// 2. Compute denominator
		// ------------------------------------------------------------
		T denom = static_cast<T>(0);

		for (size_t tile_start = 0; tile_start < j_limit; tile_start += TILE_N) {
			const size_t tile_size =
				min(static_cast<size_t>(TILE_N), j_limit - tile_start);

			// Load K tile
			for (size_t x = tid; x < tile_size * D; x += blockDim.x) {
				size_t j_local = x / D;
				size_t d = x % D;
				size_t j = tile_start + j_local;

				size_t k_index =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				shK[j_local * DMAX + d] = qkv[k_index];
			}

			__syncthreads();

			for (size_t j_local = 0; j_local < tile_size; j_local++) {
				T partial = static_cast<T>(0);

				for (size_t d = tid; d < D; d += blockDim.x) {
					size_t q_index =
						b * Tseq * 3 * C +
						t * 3 * C +
						h * D + d;

					partial += qkv[q_index] * shK[j_local * DMAX + d];
				}

				reduce[tid] = partial;
				__syncthreads();

				for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
					if (tid < stride) {
						reduce[tid] += reduce[tid + stride];
					}
					__syncthreads();
				}

				if (tid == 0) {
					T score = reduce[0] * scale;
					shScores[j_local] = score;
				}

				__syncthreads();
			}

			T local_denom = static_cast<T>(0);

			for (size_t j_local = tid; j_local < tile_size; j_local += blockDim.x) {
				local_denom += expf(static_cast<float>(shScores[j_local] - row_max));
			}

			reduce[tid] = local_denom;
			__syncthreads();

			for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
				if (tid < stride) {
					reduce[tid] += reduce[tid + stride];
				}
				__syncthreads();
			}

			if (tid == 0) {
				denom += reduce[0];
			}

			__syncthreads();
		}

		reduce[0] = denom;
		__syncthreads();
		denom = reduce[0];

		// ------------------------------------------------------------
		// 3. Compute output
		// ------------------------------------------------------------
		for (size_t d = tid; d < D; d += blockDim.x) {
			T result = static_cast<T>(0);

			for (size_t tile_start = 0; tile_start < j_limit; tile_start += TILE_N) {
				const size_t tile_size =
					min(static_cast<size_t>(TILE_N), j_limit - tile_start);

				// Load K and V tile
				for (size_t x = threadIdx.x; x < tile_size * D; x += blockDim.x) {
					size_t j_local = x / D;
					size_t dd = x % D;
					size_t j = tile_start + j_local;

					size_t k_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						C +
						h * D + dd;

					size_t v_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						2 * C +
						h * D + dd;

					shK[j_local * DMAX + dd] = qkv[k_index];
					shV[j_local * DMAX + dd] = qkv[v_index];
				}

				__syncthreads();

				// Recompute scores for this tile
				for (size_t j_local = 0; j_local < tile_size; j_local++) {
					T partial = static_cast<T>(0);

					for (size_t kd = threadIdx.x; kd < D; kd += blockDim.x) {
						size_t q_index =
							b * Tseq * 3 * C +
							t * 3 * C +
							h * D + kd;

						partial += qkv[q_index] * shK[j_local * DMAX + kd];
					}

					reduce[threadIdx.x] = partial;
					__syncthreads();

					for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
						if (threadIdx.x < stride) {
							reduce[threadIdx.x] += reduce[threadIdx.x + stride];
						}
						__syncthreads();
					}

					if (threadIdx.x == 0) {
						shScores[j_local] = reduce[0] * scale;
					}

					__syncthreads();
				}

				for (size_t j_local = 0; j_local < tile_size; j_local++) {
					T p = expf(static_cast<float>(shScores[j_local] - row_max)) / denom;
					result += p * shV[j_local * DMAX + d];
				}

				__syncthreads();
			}

			size_t out_index =
				b * Tseq * C +
				t * C +
				h * D + d;

			out[out_index] = result;
		}
	}


	template <typename T>
	void cuda_flash_attention_bigdaddy_forward_tiled(
		const T* qkv,
		T* out,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		constexpr int TILE_N = 32;
		constexpr int DMAX = 128;

		if (D > DMAX) {
			std::cerr << "flash_attention_bigdaddy_forward_tiled: D exceeds DMAX" << std::endl;
			std::exit(1);
		}

		int threads = 128;

		size_t total_rows = B * H * Tseq;
		int blocks = static_cast<int>(total_rows);

		size_t shared_bytes =
			(TILE_N * DMAX) * sizeof(T) +  // shK
			(TILE_N * DMAX) * sizeof(T) +  // shV
			TILE_N * sizeof(T) +           // shScores
			threads * sizeof(T);           // reduce

		flash_attention_bigdaddy_forward_tiled_kernel<T, TILE_N, DMAX>
			<< <blocks, threads, shared_bytes >> > (
				qkv,
				out,
				B,
				Tseq,
				C,
				H,
				D,
				causal
				);

		check_cuda(
			cudaGetLastError(),
			"flash_attention_bigdaddy_forward_tiled_kernel launch failed"
		);
	}


	template void cuda_flash_attention_bigdaddy_forward_tiled<float>(
		const float*, float*, size_t, size_t, size_t, size_t, size_t, bool
	);

	template void cuda_flash_attention_bigdaddy_forward_tiled<double>(
		const double*, double*, size_t, size_t, size_t, size_t, size_t, bool
	);






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function current production 
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T, int BLOCK_M, int TILE_N, int DMAX>
	__global__ void flash_attention_bigdaddy_forward_block_kernel2(
		const T* __restrict__ qkv,
		T* __restrict__ out,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		__shared__ T shQ[BLOCK_M][DMAX];
		__shared__ T shK[TILE_N][DMAX];
		__shared__ T shV[TILE_N][DMAX];
		__shared__ T shScore[BLOCK_M][TILE_N];

		__shared__ T shAcc[BLOCK_M][DMAX];
		__shared__ T shM[BLOCK_M];
		__shared__ T shL[BLOCK_M];
		__shared__ T shAlpha[BLOCK_M];

		int tid = threadIdx.x;

		size_t q_block_start = blockIdx.x * BLOCK_M;
		size_t h = blockIdx.y;
		size_t b = blockIdx.z;

		T scale = (T)(1.0f / sqrtf((float)D));

		for (int idx = tid; idx < BLOCK_M * DMAX; idx += blockDim.x) {
			int qi = idx / DMAX;
			int d = idx % DMAX;

			size_t t = q_block_start + qi;

			if (t < Tseq) {
				size_t q_index =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				shQ[qi][d] = qkv[q_index];
			}
			else {
				shQ[qi][d] = 0.0f;
			}

			shAcc[qi][d] = 0.0f;
		}

		for (int qi = tid; qi < BLOCK_M; qi += blockDim.x) {
			shM[qi] = -INFINITY;
			shL[qi] = 0.0f;
			shAlpha[qi] = 1.0f;
		}

		__syncthreads();

		size_t kv_limit = Tseq;

		if (causal) {
			size_t max_q_exclusive = q_block_start + BLOCK_M;
			kv_limit = min(Tseq, max_q_exclusive);
		}

		for (size_t tile_start = 0; tile_start < kv_limit; tile_start += TILE_N) {
			size_t tile_size = min((size_t)TILE_N, kv_limit - tile_start);

			for (int idx = tid; idx < TILE_N * DMAX; idx += blockDim.x) {
				int j_local = idx / DMAX;
				int d = idx % DMAX;

				if ((size_t)j_local < tile_size) {
					size_t j = tile_start + j_local;

					size_t k_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						C +
						h * D + d;

					size_t v_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						2 * C +
						h * D + d;

					shK[j_local][d] = qkv[k_index];
					shV[j_local][d] = qkv[v_index];
				}
				else {
					shK[j_local][d] = 0.0f;
					shV[j_local][d] = 0.0f;
				}
			}

			__syncthreads();

			for (int idx = tid; idx < BLOCK_M * TILE_N; idx += blockDim.x) {
				int qi = idx / TILE_N;
				int j_local = idx % TILE_N;

				size_t t = q_block_start + qi;
				size_t j = tile_start + j_local;

				T score = -INFINITY;

				if (t < Tseq && (size_t)j_local < tile_size) {
					bool valid = true;

					if (causal && j > t) {
						valid = false;
					}

					if (valid) {
						T sum = 0.0f;

#pragma unroll
						for (int d = 0; d < DMAX; d++) {
							sum += shQ[qi][d] * shK[j_local][d];
						}

						score = sum * scale;
					}
				}

				shScore[qi][j_local] = score;
			}

			__syncthreads();

			for (int qi = tid; qi < BLOCK_M; qi += blockDim.x) {
				size_t t = q_block_start + qi;

				if (t < Tseq) {
					T old_m = shM[qi];
					T old_l = shL[qi];

					T tile_max = -INFINITY;

#pragma unroll
					for (int j_local = 0; j_local < TILE_N; j_local++) {
						if ((size_t)j_local < tile_size) {
							tile_max = fmaxf(tile_max, shScore[qi][j_local]);
						}
					}

					T new_m = max(old_m, tile_max);
					T alpha = exp(old_m - new_m);

					T tile_l = 0.0f;

#pragma unroll
					for (int j_local = 0; j_local < TILE_N; j_local++) {
						if ((size_t)j_local < tile_size) {
							tile_l += __expf(shScore[qi][j_local] - new_m);
						}
					}

					shM[qi] = new_m;
					shL[qi] = old_l * alpha + tile_l;
					shAlpha[qi] = alpha;
				}
			}

			__syncthreads();

			for (int idx = tid; idx < BLOCK_M * DMAX; idx += blockDim.x) {
				int qi = idx / DMAX;
				int d = idx % DMAX;

				size_t t = q_block_start + qi;

				if (t < Tseq) {
					T new_m = shM[qi];
					T alpha = shAlpha[qi];

					T tile_acc = 0.0f;

#pragma unroll
					for (int j_local = 0; j_local < TILE_N; j_local++) {
						if ((size_t)j_local < tile_size) {
							T p = (T)__expf((float)(shScore[qi][j_local] - new_m));
							tile_acc += p * shV[j_local][d];
						}
					}

					shAcc[qi][d] = shAcc[qi][d] * alpha + tile_acc;
				}
			}

			__syncthreads();
		}

		for (int idx = tid; idx < BLOCK_M * DMAX; idx += blockDim.x) {
			int qi = idx / DMAX;
			int d = idx % DMAX;

			size_t t = q_block_start + qi;

			if (t < Tseq) {
				size_t out_index =
					b * Tseq * C +
					t * C +
					h * D + d;

				T l = shL[qi];

				if (l <= 0.0f || !isfinite(l)) {
					out[out_index] = 0.0f;
				}
				else {
					out[out_index] = shAcc[qi][d] / l;
				}
			}
		}
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
	void cuda_flash_block(
		const T* qkvptr,
		T* optr,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		constexpr int BLOCK_M = 8;
		constexpr int TILE_N = 32;
		constexpr int DMAX = 64;
		constexpr int THREADS = 256;

		if (D != DMAX) {
			std::cerr << "cuda_flash_block currently expects D == 64" << std::endl;
			std::exit(1);
		}

		dim3 grid(
			(unsigned int)((Tseq + BLOCK_M - 1) / BLOCK_M),
			(unsigned int)H,
			(unsigned int)B
		);

		flash_attention_bigdaddy_forward_block_kernel2<T, BLOCK_M, TILE_N, DMAX>
			<< <grid, THREADS >> > (
				qkvptr,
				optr,
				B,
				Tseq,
				C,
				H,
				D,
				causal
				);

		cudaError_t err = cudaGetLastError();

		if (err != cudaSuccess) {
			std::cerr << "flash_attention_bigdaddy_forward_block_kernel launch failed: "
				<< cudaGetErrorString(err) << std::endl;
			std::exit(1);
		}

		//cudaDeviceSynchronize();
	}

	template void cuda_flash_block<float>(const float*, float*, size_t, size_t, size_t, size_t, size_t, bool);
	template void cuda_flash_block<double>(const double*, double*, size_t, size_t, size_t, size_t, size_t, bool);




	template<typename T, int BLOCK_M, int TILE_N, int DMAX>
		__global__ void flash_backward_fused_kernel(
			const T* __restrict__ qkv,
			const T* __restrict__ dout,
			T* __restrict__ dqkv,
			size_t B,
			size_t Tseq,
			size_t C,
			size_t H,
			size_t D,
			bool causal
		) {
		constexpr int WARPS = BLOCK_M;
		constexpr int THREADS = WARPS * 32;

		static_assert(DMAX == 64, "DMAX must be 64");

		__shared__ T shK[TILE_N][DMAX];
		__shared__ T shV[TILE_N][DMAX];

		__shared__ T shDK[TILE_N][DMAX];
		__shared__ T shDV[TILE_N][DMAX];

		int tid = threadIdx.x;
		int warp = tid / 32;
		int lane = tid % 32;

		size_t q_start = blockIdx.x * BLOCK_M;

		size_t bh = blockIdx.y;
		size_t b = bh / H;
		size_t h = bh % H;

		size_t t = q_start + warp;

		if (warp >= BLOCK_M || t >= Tseq) {
			return;
		}

		float scale = rsqrtf((float)D);

		T q_reg[2];
		T do_reg[2];
		T dq_reg[2] = { (T)0, (T)0 };

		// Load Q[t] and dO[t].
		for (int i = 0; i < 2; i++) {
			int d = lane + i * 32;

			size_t q_index =
				b * Tseq * 3 * C +
				t * 3 * C +
				h * D + d;

			size_t do_index =
				b * Tseq * C +
				t * C +
				h * D + d;

			q_reg[i] = qkv[q_index];
			do_reg[i] = dout[do_index];
		}

		// =====================================================================
		// Pass 1: recompute row softmax normalizer:
		//
		//   row_m = max_j score_j
		//   row_l = sum_j exp(score_j - row_m)
		//
		// where score_j = dot(Q_t, K_j) / sqrt(D)
		// =====================================================================

		float row_m = -INFINITY;
		float row_l = 0.0f;

		for (size_t k_start = 0; k_start < Tseq; k_start += TILE_N) {
			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int kj = idx / DMAX;
				int d = idx % DMAX;

				size_t j = k_start + kj;

				if (j < Tseq) {
					size_t k_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						C +
						h * D + d;

					shK[kj][d] = qkv[k_index];
				}
				else {
					shK[kj][d] = (T)0;
				}
			}

			__syncthreads();

			for (int kj = 0; kj < TILE_N; kj++) {
				size_t j = k_start + kj;

				if (j >= Tseq) break;
				if (causal && j > t) continue;

				float score = 0.0f;

				for (int i = 0; i < 2; i++) {
					int d = lane + i * 32;
					score += (float)q_reg[i] * (float)shK[kj][d];
				}

#pragma unroll
				for (int offset = 16; offset > 0; offset >>= 1) {
					score += __shfl_down_sync(0xffffffff, score, offset);
				}
				score = __shfl_sync(0xffffffff, score, 0);

				score *= scale;

				float old_m = row_m;
				row_m = fmaxf(row_m, score);

				float alpha = __expf(old_m - row_m);
				row_l = row_l * alpha + __expf(score - row_m);
			}

			__syncthreads();
		}

		// =====================================================================
		// Pass 2: compute full delta for this row:
		//
		//   dp_j    = dot(dO_t, V_j)
		//   p_j     = softmax(score_j)
		//   delta   = sum_j p_j * dp_j
		//
		// This must be complete before computing dS_j.
		// =====================================================================

		float delta = 0.0f;

		for (size_t k_start = 0; k_start < Tseq; k_start += TILE_N) {
			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int kj = idx / DMAX;
				int d = idx % DMAX;

				size_t j = k_start + kj;

				if (j < Tseq) {
					size_t k_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						C +
						h * D + d;

					size_t v_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						2 * C +
						h * D + d;

					shK[kj][d] = qkv[k_index];
					shV[kj][d] = qkv[v_index];
				}
				else {
					shK[kj][d] = (T)0;
					shV[kj][d] = (T)0;
				}
			}

			__syncthreads();

			for (int kj = 0; kj < TILE_N; kj++) {
				size_t j = k_start + kj;

				if (j >= Tseq) break;
				if (causal && j > t) continue;

				float score = 0.0f;
				float dp = 0.0f;

				for (int i = 0; i < 2; i++) {
					int d = lane + i * 32;

					score += (float)q_reg[i] * (float)shK[kj][d];
					dp += (float)do_reg[i] * (float)shV[kj][d];
				}

#pragma unroll
				for (int offset = 16; offset > 0; offset >>= 1) {
					score += __shfl_down_sync(0xffffffff, score, offset);
					dp += __shfl_down_sync(0xffffffff, dp, offset);
				}

				score = __shfl_sync(0xffffffff, score, 0);
				dp = __shfl_sync(0xffffffff, dp, 0);

				score *= scale;

				float p = __expf(score - row_m) / row_l;

				delta += p * dp;
			}

			__syncthreads();
		}

		// =====================================================================
		// Pass 3: compute dQ, dK, dV using full delta:
		//
		//   dP_j = dot(dO_t, V_j)
		//   dS_j = p_j * (dP_j - delta) * scale
		//
		//   dQ_t += dS_j * K_j
		//   dK_j += dS_j * Q_t
		//   dV_j += p_j  * dO_t
		//
		// scale belongs in dS because S = QK^T * scale.
		// =====================================================================

		for (size_t k_start = 0; k_start < Tseq; k_start += TILE_N) {
			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int kj = idx / DMAX;
				int d = idx % DMAX;

				size_t j = k_start + kj;

				if (j < Tseq) {
					size_t k_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						C +
						h * D + d;

					size_t v_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						2 * C +
						h * D + d;

					shK[kj][d] = qkv[k_index];
					shV[kj][d] = qkv[v_index];
				}
				else {
					shK[kj][d] = (T)0;
					shV[kj][d] = (T)0;
				}

				shDK[kj][d] = (T)0;
				shDV[kj][d] = (T)0;
			}

			__syncthreads();

			for (int kj = 0; kj < TILE_N; kj++) {
				size_t j = k_start + kj;

				if (j >= Tseq) break;
				if (causal && j > t) continue;

				float score = 0.0f;
				float dp = 0.0f;

				for (int i = 0; i < 2; i++) {
					int d = lane + i * 32;

					score += (float)q_reg[i] * (float)shK[kj][d];
					dp += (float)do_reg[i] * (float)shV[kj][d];
				}

#pragma unroll
				for (int offset = 16; offset > 0; offset >>= 1) {
					score += __shfl_down_sync(0xffffffff, score, offset);
					dp += __shfl_down_sync(0xffffffff, dp, offset);
				}

				score = __shfl_sync(0xffffffff, score, 0);
				dp = __shfl_sync(0xffffffff, dp, 0);

				score *= scale;

				float p = __expf(score - row_m) / row_l;

				float dS = p * (dp - delta) * scale;

				for (int i = 0; i < 2; i++) {
					int d = lane + i * 32;

					dq_reg[i] = (T)((float)dq_reg[i] + dS * (float)shK[kj][d]);

					atomicAdd(&shDK[kj][d], (T)(dS * (float)q_reg[i]));
					atomicAdd(&shDV[kj][d], (T)(p * (float)do_reg[i]));
				}
			}

			__syncthreads();

			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int kj = idx / DMAX;
				int d = idx % DMAX;

				size_t j = k_start + kj;

				if (j < Tseq) {
					size_t dk_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						C +
						h * D + d;

					size_t dv_index =
						b * Tseq * 3 * C +
						j * 3 * C +
						2 * C +
						h * D + d;

					atomicAdd(&dqkv[dk_index], shDK[kj][d]);
					atomicAdd(&dqkv[dv_index], shDV[kj][d]);
				}
			}

			__syncthreads();
		}

		// Store dQ for this query row.
		for (int i = 0; i < 2; i++) {
			int d = lane + i * 32;

			size_t dq_index =
				b * Tseq * 3 * C +
				t * 3 * C +
				h * D + d;

			dqkv[dq_index] = dq_reg[i];
		}
	}

	// flash_backward_fused_kernel.cu
//
// Memory-efficient FlashAttention backward pass.
//
// Computes dQ, dK, dV from packed QKV and dOut without materialising the full
// [Tseq x Tseq] attention matrix.
//
// Grid:  (ceil(Tseq/BLOCK_M),  B*H)
// Block: (BLOCK_M * 32)  threads   (one warp per query row)
//
// Template parameters
//   T       – element type (float, __half, __bfloat16)
//   BLOCK_M – number of query rows handled per block (= warps per block)
//   TILE_N  – number of KV rows loaded into shared memory per tile
//   DMAX    – head dimension; must be 64
//
// Tensor layout  qkv / dqkv : [B, Tseq, 3, H, D]  (Q, K, V packed)
//                dout        : [B, Tseq,    H, D]

#include <cuda_fp16.h>
#include <cuda_runtime.h>

	template<typename T, int BLOCK_M, int TILE_N, int DMAX>
	__global__ void flash_backward_fused_kernel2(
		const T* __restrict__ qkv,
		const T* __restrict__ dout,
		T* __restrict__       dqkv,
		size_t B,
		size_t Tseq,
		size_t C,       // H * D
		size_t H,
		size_t D,
		bool   causal
	) {
		// -------------------------------------------------------------------------
		// Compile-time checks
		// -------------------------------------------------------------------------
		constexpr int WARPS = BLOCK_M;
		constexpr int THREADS = WARPS * 32;
		static_assert(DMAX == 64, "DMAX must be 64");

		// -------------------------------------------------------------------------
		// Shared memory
		//
		// All accumulators are float so atomicAdd is always valid regardless of T.
		// -------------------------------------------------------------------------
		__shared__ float shK[TILE_N][DMAX];   // current K tile
		__shared__ float shV[TILE_N][DMAX];   // current V tile
		__shared__ float shDK[TILE_N][DMAX];   // dK accumulator (zeroed per tile)
		__shared__ float shDV[TILE_N][DMAX];   // dV accumulator (zeroed per tile)

		// -------------------------------------------------------------------------
		// Thread / warp / block identity
		// -------------------------------------------------------------------------
		int tid = threadIdx.x;
		int warp = tid / 32;
		int lane = tid % 32;

		size_t q_start = (size_t)blockIdx.x * BLOCK_M;
		size_t bh = blockIdx.y;
		size_t b = bh / H;
		size_t h = bh % H;
		size_t t = q_start + warp;

		// FIX 1: never return early – every thread must reach every __syncthreads().
		// Use a boolean flag to guard computation for out-of-bounds warps instead.
		bool active = (warp < BLOCK_M) && (t < Tseq);

		// scale = 1 / sqrt(D), applied to the QK dot product.
		float scale = rsqrtf((float)D);

		// -------------------------------------------------------------------------
		// Per-warp registers (only meaningful when active)
		// -------------------------------------------------------------------------
		float q_reg[2] = { 0.f, 0.f };   // Q[t]
		float do_reg[2] = { 0.f, 0.f };   // dO[t]
		float dq_reg[2] = { 0.f, 0.f };   // accumulator for dQ[t]

		if (active) {
			for (int i = 0; i < 2; i++) {
				int    d = lane + i * 32;
				size_t q_index = b * Tseq * 3 * C + t * 3 * C + h * D + d;
				size_t do_index = b * Tseq * C + t * C + h * D + d;
				q_reg[i] = (float)qkv[q_index];
				do_reg[i] = (float)dout[do_index];
			}
		}

		// =========================================================================
		// Pass 1 – Recompute per-row online-softmax statistics
		//
		//   row_m = max_j  score_j               (running maximum)
		//   row_l = sum_j  exp(score_j - row_m)  (running normaliser)
		//
		// where  score_j = dot(Q_t, K_j) * scale
		//
		// Uses the numerically-stable online algorithm (Milakov & Gimelshein 2018):
		// when a new maximum is found the running sum is rescaled by
		//   exp(old_m - new_m)
		// so that all terms share the same subtracted maximum.
		// =========================================================================
		float row_m = -INFINITY;
		float row_l = 0.f;

		for (size_t k_start = 0; k_start < Tseq; k_start += TILE_N) {

			// Cooperative load of K tile – all threads participate so that
			// __syncthreads() is always reached with the correct number of writers.
			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int    kj = idx / DMAX;
				int    d = idx % DMAX;
				size_t j = k_start + kj;
				if (j < Tseq) {
					size_t k_index = b * Tseq * 3 * C + j * 3 * C + C + h * D + d;
					shK[kj][d] = (float)qkv[k_index];
				}
				else {
					shK[kj][d] = 0.f;
				}
			}
			__syncthreads();   // barrier 1: tile fully loaded

			if (active) {
				for (int kj = 0; kj < TILE_N; kj++) {
					size_t j = k_start + kj;
					if (j >= Tseq)       break;
					if (causal && j > t) continue;

					// Warp-parallel dot product Q_t · K_j
					float score = 0.f;
					for (int i = 0; i < 2; i++) {
						int d = lane + i * 32;
						score += q_reg[i] * shK[kj][d];
					}
#pragma unroll
					for (int off = 16; off > 0; off >>= 1)
						score += __shfl_down_sync(0xffffffff, score, off);
					score = __shfl_sync(0xffffffff, score, 0);  // broadcast to all lanes
					score *= scale;

					// Online softmax update
					float old_m = row_m;
					row_m = fmaxf(row_m, score);
					row_l = row_l * __expf(old_m - row_m) + __expf(score - row_m);
				}
			}
			__syncthreads();   // barrier 2: safe to overwrite shK next iteration
		}

		// =========================================================================
		// Pass 2 – Compute per-row delta
		//
		//   dP_ij = dot(dO_t, V_j)
		//   P_ij  = exp(score_ij - row_m) / row_l
		//   delta = sum_j  P_ij * dP_ij
		//
		// delta must be fully accumulated over ALL j before Pass 3 begins because
		// every dS_ij depends on it.
		// =========================================================================
		float delta = 0.f;

		for (size_t k_start = 0; k_start < Tseq; k_start += TILE_N) {

			// Load K and V tiles cooperatively
			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int    kj = idx / DMAX;
				int    d = idx % DMAX;
				size_t j = k_start + kj;
				if (j < Tseq) {
					size_t k_index = b * Tseq * 3 * C + j * 3 * C + C + h * D + d;
					size_t v_index = b * Tseq * 3 * C + j * 3 * C + 2 * C + h * D + d;
					shK[kj][d] = (float)qkv[k_index];
					shV[kj][d] = (float)qkv[v_index];
				}
				else {
					shK[kj][d] = 0.f;
					shV[kj][d] = 0.f;
				}
			}
			__syncthreads();

			if (active) {
				for (int kj = 0; kj < TILE_N; kj++) {
					size_t j = k_start + kj;
					if (j >= Tseq)       break;
					if (causal && j > t) continue;

					float score = 0.f, dp = 0.f;
					for (int i = 0; i < 2; i++) {
						int d = lane + i * 32;
						score += q_reg[i] * shK[kj][d];
						dp += do_reg[i] * shV[kj][d];
					}
#pragma unroll
					for (int off = 16; off > 0; off >>= 1) {
						score += __shfl_down_sync(0xffffffff, score, off);
						dp += __shfl_down_sync(0xffffffff, dp, off);
					}
					score = __shfl_sync(0xffffffff, score, 0);
					dp = __shfl_sync(0xffffffff, dp, 0);
					score *= scale;

					// FIX 3: guard against fully-masked rows (row_l == 0 → NaN)
					float p = (row_l > 0.f) ? __expf(score - row_m) / row_l : 0.f;
					delta += p * dp;
				}
			}
			__syncthreads();
		}

		// =========================================================================
		// Pass 3 – Accumulate dQ, dK, dV
		//
		// For each (t, j) pair:
		//
		//   dP_ij = dot(dO_t, V_j)
		//   dS_ij = P_ij * (dP_ij - delta_t)
		//
		// Gradient through  S = QK^T * scale  (scale applied to the dot product):
		//
		//   dQ_t  += dS_ij * scale * K_j       ← scale here, NOT inside dS
		//   dK_j  += dS_ij * scale * Q_t       ← scale here, NOT inside dS
		//   dV_j  += P_ij  * dO_t              ← no scale factor
		//
		// FIX 4: scale appears in the Q/K updates, not baked into dS.
		//        The original code had  dS = p*(dp-delta)*scale  which was correct
		//        in magnitude but conceptually wrong; the refactored version is
		//        cleaner and easier to verify against the math.
		//
		// Atomics strategy:
		//   dQ: accumulate in register dq_reg (one warp owns row t exclusively)
		//   dK/dV: atomicAdd into shared memory first, then flush to global.
		//          Multiple warps within the block write to the same K/V row j,
		//          so intra-block atomics into shared are required.  Only one
		//          thread-group then atomicAdds the tile result into global memory.
		// =========================================================================
		for (size_t k_start = 0; k_start < Tseq; k_start += TILE_N) {

			// Load K, V; zero dK, dV accumulators.
			// Zeroing is done in the same loop to guarantee every element is
			// initialised before any warp writes to shDK/shDV.
			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int    kj = idx / DMAX;
				int    d = idx % DMAX;
				size_t j = k_start + kj;
				if (j < Tseq) {
					size_t k_index = b * Tseq * 3 * C + j * 3 * C + C + h * D + d;
					size_t v_index = b * Tseq * 3 * C + j * 3 * C + 2 * C + h * D + d;
					shK[kj][d] = (float)qkv[k_index];
					shV[kj][d] = (float)qkv[v_index];
				}
				else {
					shK[kj][d] = 0.f;
					shV[kj][d] = 0.f;
				}
				// FIX 3 (part 2): zero accumulators unconditionally for every
				// element, including out-of-bounds slots, so stale values from a
				// previous iteration can never contaminate the flush to global mem.
				shDK[kj][d] = 0.f;
				shDV[kj][d] = 0.f;
			}
			__syncthreads();   // barrier: tile + zero complete before any compute

			if (active) {
				for (int kj = 0; kj < TILE_N; kj++) {
					size_t j = k_start + kj;
					if (j >= Tseq)       break;
					if (causal && j > t) continue;

					float score = 0.f, dp = 0.f;
					for (int i = 0; i < 2; i++) {
						int d = lane + i * 32;
						score += q_reg[i] * shK[kj][d];
						dp += do_reg[i] * shV[kj][d];
					}
#pragma unroll
					for (int off = 16; off > 0; off >>= 1) {
						score += __shfl_down_sync(0xffffffff, score, off);
						dp += __shfl_down_sync(0xffffffff, dp, off);
					}
					score = __shfl_sync(0xffffffff, score, 0);
					dp = __shfl_sync(0xffffffff, dp, 0);
					score *= scale;

					float p = (row_l > 0.f) ? __expf(score - row_m) / row_l : 0.f;
					float dS = p * (dp - delta);      // softmax-corrected gradient

					for (int i = 0; i < 2; i++) {
						int d = lane + i * 32;
						// FIX 4: scale applied in the update equations, not in dS
						dq_reg[i] += dS * scale * shK[kj][d];
						atomicAdd(&shDK[kj][d], dS * scale * q_reg[i]);
						atomicAdd(&shDV[kj][d], p * do_reg[i]);
					}
				}
			}
			__syncthreads();   // barrier: all atomics into shDK/shDV are complete

			// Flush tile dK, dV accumulators to global memory.
			// All threads cooperate so no warp sits idle during the writeback.
			for (int idx = tid; idx < TILE_N * DMAX; idx += THREADS) {
				int    kj = idx / DMAX;
				int    d = idx % DMAX;
				size_t j = k_start + kj;
				if (j < Tseq) {
					size_t dk_index = b * Tseq * 3 * C + j * 3 * C + C + h * D + d;
					size_t dv_index = b * Tseq * 3 * C + j * 3 * C + 2 * C + h * D + d;
					// atomicAdd because different blocks (different query-row tiles)
					// all write into the same K/V rows.
					atomicAdd(&dqkv[dk_index], (T)shDK[kj][d]);
					atomicAdd(&dqkv[dv_index], (T)shDV[kj][d]);
				}
			}
			__syncthreads();   // barrier before next tile overwrites shK/shV/shDK/shDV
		}

		// -------------------------------------------------------------------------
		// Write dQ for query row t.
		//
		// FIX 5: plain store is correct here.  The grid partitions query rows into
		// non-overlapping BLOCK_M-sized chunks, so exactly one block owns each row t
		// and no other block writes to dqkv[dq_index].  No atomic needed.
		// -------------------------------------------------------------------------
		if (active) {
			for (int i = 0; i < 2; i++) {
				int    d = lane + i * 32;
				size_t dq_index = b * Tseq * 3 * C + t * 3 * C + h * D + d;
				dqkv[dq_index] = (T)dq_reg[i];
			}
		}
	}


	template<typename T>
	__global__ void flash_backward_reference_kernel(
		const T* __restrict__ qkvptr,
		const T* __restrict__ doutptr,
		T* __restrict__ dqkvptr,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		size_t t = blockIdx.x;
		size_t h = blockIdx.y;
		size_t b = blockIdx.z;

		if (threadIdx.x != 0) {
			return;
		}

		T scale = (T)1 / flash_sqrt((T)D);

		// ============================================================
		// Pass 1: row max
		// ============================================================

		T row_max = -(T)1e30;

		for (size_t j = 0; j < Tseq; j++) {

			if (causal && j > t) {
				continue;
			}

			T score = (T)0;

			for (size_t d = 0; d < D; d++) {

				size_t q_idx =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				score += qkvptr[q_idx] * qkvptr[k_idx];
			}

			score *= scale;

			if (score > row_max) {
				row_max = score;
			}
		}

		// ============================================================
		// Pass 2: softmax denominator
		// ============================================================

		T row_sum = (T)0;

		for (size_t j = 0; j < Tseq; j++) {

			if (causal && j > t) {
				continue;
			}

			T score = (T)0;

			for (size_t d = 0; d < D; d++) {

				size_t q_idx =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				score += qkvptr[q_idx] * qkvptr[k_idx];
			}

			score *= scale;

			row_sum += flash_exp(score - row_max);
		}

		// ============================================================
		// Pass 3: delta
		// ============================================================

		T delta = (T)0;

		for (size_t j = 0; j < Tseq; j++) {

			if (causal && j > t) {
				continue;
			}

			T score = (T)0;
			T dp = (T)0;

			for (size_t d = 0; d < D; d++) {

				size_t q_idx =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				size_t v_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					2 * C +
					h * D + d;

				size_t do_idx =
					b * Tseq * C +
					t * C +
					h * D + d;

				score += qkvptr[q_idx] * qkvptr[k_idx];
				dp += doutptr[do_idx] * qkvptr[v_idx];
			}

			score *= scale;

			T p = flash_exp(score - row_max) / row_sum;

			delta += p * dp;
		}

		// ============================================================
		// Pass 4: dQ dK dV
		// ============================================================

		for (size_t j = 0; j < Tseq; j++) {

			if (causal && j > t) {
				continue;
			}

			T score = (T)0;
			T dp = (T)0;

			for (size_t d = 0; d < D; d++) {

				size_t q_idx =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				size_t v_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					2 * C +
					h * D + d;

				size_t do_idx =
					b * Tseq * C +
					t * C +
					h * D + d;

				score += qkvptr[q_idx] * qkvptr[k_idx];
				dp += doutptr[do_idx] * qkvptr[v_idx];
			}

			score *= scale;

			T p = flash_exp(score - row_max) / row_sum;

			T dscore = p * (dp - delta);

			T ddot = dscore * scale;

			for (size_t d = 0; d < D; d++) {

				size_t q_idx =
					b * Tseq * 3 * C +
					t * 3 * C +
					h * D + d;

				size_t k_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					C +
					h * D + d;

				size_t v_idx =
					b * Tseq * 3 * C +
					j * 3 * C +
					2 * C +
					h * D + d;

				size_t do_idx =
					b * Tseq * C +
					t * C +
					h * D + d;

				atomicAdd(&dqkvptr[q_idx], ddot * qkvptr[k_idx]);

				atomicAdd(&dqkvptr[k_idx], ddot * qkvptr[q_idx]);

				atomicAdd(&dqkvptr[v_idx], p * doutptr[do_idx]);
			}
		}
	}


	template <typename T>
	void cuda_flash_backward_fused(
		const T* qkvptr,
		const T* doutptr,
		T* dqkvptr,
		size_t B,
		size_t Tseq,
		size_t C,
		size_t H,
		size_t D,
		bool causal
	) {
		constexpr int BLOCK_M = 8;
		constexpr int TILE_N = 16;
		constexpr int DMAX = 64;

		constexpr int THREADS = BLOCK_M * 32;

		if (D != DMAX) {
			std::cerr << "Expected D == 64\n";
			std::exit(1);
		}

		size_t qkv_numel = B * Tseq * 3 * C;

		//cudaMemset(dqkvptr, 0, qkv_numel * sizeof(float));
		cudaMemset(dqkvptr, 0, qkv_numel * sizeof(T));

		dim3 grid(
			(unsigned int)((Tseq + BLOCK_M - 1) / BLOCK_M),
			(unsigned int)(B * H)
		);


		/*dim3 grid(
			(unsigned int)Tseq,
			(unsigned int)H,
			(unsigned int)B
		);*/

		//dim3 block(1);

		
		flash_backward_fused_kernel2<T, BLOCK_M, TILE_N, DMAX>
		//flash_backward_reference_kernel<T>
			<< <grid, THREADS >> > (
				qkvptr,
				doutptr,
				dqkvptr,
				B,
				Tseq,
				C,
				H,
				D,
				causal
				);

		cudaError_t err = cudaGetLastError();

		if (err != cudaSuccess) {
			std::cerr
				<< "flash_backward_fused_kernel launch failed: "
				<< cudaGetErrorString(err)
				<< std::endl;

			std::exit(1);
		}

		//cudaDeviceSynchronize();
	}


	template void cuda_flash_backward_fused<float>(const float*, const float*, float*, size_t, size_t, size_t, size_t, size_t, bool);
	template void cuda_flash_backward_fused<double>(const double*, const double*, double*, size_t, size_t, size_t, size_t, size_t, bool);

}