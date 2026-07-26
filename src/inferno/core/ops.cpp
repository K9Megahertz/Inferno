#include <inferno/core/ops.h>
#include "inferno/core/ops_impl.h"
#include "inferno/gradfn/addbackward.h"
#include "inferno/gradfn/subtractbackward.h"
#include "inferno/gradfn/multiplybackward.h"
#include "inferno/gradfn/dividebackward.h"
#include "inferno/gradfn/negatebackward.h"
#include "inferno/gradfn/mmbackward.h"
#include "inferno/gradfn/mselossbackward.h"
#include "inferno/gradfn/slicebackward.h"
#include "inferno/gradfn/reshapebackward.h"
#include "inferno/gradfn/concatbackward.h"
#include "inferno/gradfn/selectbackward.h"
#include "inferno/gradfn/contiguousbackward.h"
#include "inferno/gradfn/flashattentionbackward.h"
#include "inferno/gradfn/maskedfillbackward.h"
#include <logger/logger.h>
#include "tensorimpl.h"
#include "inferno/gradengine/engine.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/core/cpuops.h"
#include "inferno/core/dtype_dispatch.h"

int g_mmcountcublasSgemm = 0;
int g_mmcountcublasSgemmStridedBatched = 0;
int g_mmcountslow = 0;

std::unordered_map<std::string, size_t> g_matmul_counts;

namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function add
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor add(const Tensor& A, const Tensor& B) {


		if (A.device() != B.device()) {
			INFERNO_LOG_ERROR() << "Incompatible device types on tensor parameters in Add" << std::endl;
			exit(1);
		}

		return dispatchAnyTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;



			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			bool gradreq = false;
			if (Inferno::grad_enabled) {
				gradreq = A.requires_grad() || B.requires_grad();
			}

			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "add", A.device(), gradreq);

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();



			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal add path" << std::endl;
				cpu_add<AT,BT,RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal add path" << std::endl;
				cuda_add<AT, BT, RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad() || B.requires_grad())) {
				INFERNO_LOG_DEBUG() << "Add - Making a AddBackward node" << std::endl;
				implout->gradfn() = std::make_shared<AddBackward>(A, B);
			}


			return out;
		});
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function subtract
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor subtract(const Tensor& A, const Tensor& B) {


		if (A.device() != B.device()) {
			INFERNO_LOG_ERROR() << "Incompatible device types on tensor parameters in Add" << std::endl;
			exit(1);
		}

		return dispatchAnyTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;



			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			bool gradreq = false;
			if (Inferno::grad_enabled) {
				gradreq = A.requires_grad() || B.requires_grad();
			}

			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "subtract", A.device(), gradreq);

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();



			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal subtract path" << std::endl;
				cpu_subtract<AT, BT, RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal subtract path" << std::endl;
				cuda_subtract<AT, BT, RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad() || B.requires_grad())) {
				INFERNO_LOG_DEBUG() << "Subtract - Making a SubtractBackward node" << std::endl;
				implout->gradfn() = std::make_shared<SubtractBackward>(A, B);
			}


			return out;
			});
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function multiply
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor multiply(const Tensor & A, const Tensor & B) {


		if (A.device() != B.device()) {
			INFERNO_LOG_ERROR() << "Incompatible device types on tensor parameters in Add" << std::endl;
			exit(1);
		}

		return dispatchAnyTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;

			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			bool gradreq = false;
			if (Inferno::grad_enabled) {
				gradreq = A.requires_grad() || B.requires_grad();
			}

			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "multiply", A.device(), gradreq);

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();


			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal multiply path" << std::endl;
				cpu_multiply<AT, BT, RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal multiply path" << std::endl;
				cuda_multiply<AT, BT, RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad() || B.requires_grad())) {
				INFERNO_LOG_DEBUG() << "Multiply - Making a MultiplyBackward node" << std::endl;
				implout->gradfn() = std::make_shared<MultiplyBackward>(A, B);
			}


			return out;
		});

	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function divide
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor divide(const Tensor& A, const Tensor& B) {


		if (A.device() != B.device()) {
			INFERNO_LOG_ERROR() << "Incompatible device types on tensor parameters in divide" << std::endl;
			exit(1);
		}

		return dispatchAnyTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;

			using ADiv = promote_t<AT, float>;
			using BDiv = promote_t<BT, float>;
			using RT = promote_t<ADiv, BDiv>;



			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			bool gradreq = false;
			if (Inferno::grad_enabled) {
				gradreq = A.requires_grad() || B.requires_grad();
			}

			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "divide", A.device(), gradreq);

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();



			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal divide path" << std::endl;
				cpu_divide<AT, BT, RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal divide path" << std::endl;
				cuda_divide<AT, BT, RT>(aptr, bptr, optr, A.shape(), A.strides(), A.offset(), B.shape(), B.strides(), B.offset(), out.shape(), out.numel());
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad() || B.requires_grad())) {
				INFERNO_LOG_DEBUG() << "Divide - Making a DivideBackward node" << std::endl;
				implout->gradfn() = std::make_shared<DivideBackward>(A, B);
			}


			return out;
			});
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function negate
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor negate(const Tensor& A) {
				

		return dispatchAny(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;			

			auto implA = GetImpl(A);		

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();

			bool gradreq = false;
			if (Inferno::grad_enabled) {
				gradreq = A.requires_grad();
			}

			Inferno::Tensor out(dtype_of_v<AT>, A.shape(), "negate", A.device(), gradreq);

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<AT>();


			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal negate path" << std::endl;
				cpu_negate<AT>(aptr, optr, out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal negate path" << std::endl;
				cuda_negate<AT>(aptr, optr, out.numel());
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}

			if ((Inferno::grad_enabled) && (A.requires_grad())) {
				implout->gradfn() = std::make_shared<NegateBackward>(A);
			}

			return out;
			});

	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function matmul
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor matmul(const Tensor& A, const Tensor& B, std::string label, bool transA, bool transB) {


		bool a_vec = (A.ndim() == 1);
		bool b_vec = (B.ndim() == 1);

		Tensor A2 =  make_view(A, A.shape(), A.strides(), A.offset(), "A2");
		Tensor B2 =  make_view(B, B.shape(), B.strides(), B.offset(), "B2");


		//if (a_vec) {
            //A2.shape() = { 1, A2.shape()[0] };
			//A2.strides() = A2.calculate_strides(A2.shape());
		//}

		if (a_vec) {
			A2.shape() = { 1, A.shape()[0] };
			A2.strides() = { 0, A.strides()[0] };
		}

		//if (b_vec) {
			//B2.shape() = { B2.shape()[0], 1 };
			//B2.strides() = B2.calculate_strides(B2.shape());
		//}

		if (b_vec) {
			B2.shape() = { B.shape()[0], 1 };
			B2.strides() = { B.strides()[0], 0 };
		}

		Tensor out = matmul_impl(A2, B2, label, transA, transB);

		if (a_vec && b_vec)
			out.shape()={1};   // scalar	
		else if (a_vec) 
			out.shape().erase(out.shape().begin() + 0);
		else if (b_vec) 
			out.shape().erase(out.shape().begin() + 1);	
			
		out.strides() = out.calculate_strides(out.shape());

		
		if ((Inferno::grad_enabled) && (A.requires_grad() || B.requires_grad())) {
			INFERNO_LOG_DEBUG() << "Matmul - Making a MMBackward node" << std::endl;
			GetImpl(out)->gradfn() = std::make_shared<MMBackward>(A, B);
		}
			
		
		return out;

	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function matmul_impl
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor matmul_impl(const Tensor& A, const Tensor& B, std::string label, bool transA, bool transB) {

		return dispatchAnyTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;

			size_t a_ndim = A.ndim();
			size_t b_ndim = B.ndim();

				

			// Ensure tensors have at least 1 dimension
			if (a_ndim < 1 || b_ndim < 1) {
				INFERNO_LOG_ERROR() << "Tensors must have at least 1 dimension for matmul" << std::endl;
				exit(1);
			}


			//get the rank size of the bigger Tensor
			size_t max_rank = std::max(a_ndim, b_ndim);

			//we gonna pad the shapes fill it with 1's to start off with
			std::vector<size_t> a_padded_shape(max_rank, 1);
			std::vector<size_t> b_padded_shape(max_rank, 1);

			std::vector<size_t> a_padded_strides(max_rank, 0);
			std::vector<size_t> b_padded_strides(max_rank, 0);

			//where to start padding
			size_t a_pad_offset = max_rank - a_ndim;
			size_t b_pad_offset = max_rank - b_ndim;

			//pad the shapes to match the bigger size e.g. A = {3,4} B={2,3,4,5} A would get padded to {1,1,3,4}
			for (size_t i = 0; i < a_ndim; i++) {
				a_padded_shape[a_pad_offset + i] = A.shape()[i];
				a_padded_strides[a_pad_offset + i] = A.strides()[i];
			}

			for (size_t i = 0; i < b_ndim; i++) {
				b_padded_shape[b_pad_offset + i] = B.shape()[i];
				b_padded_strides[b_pad_offset + i] = B.strides()[i];
			}

			size_t a_rows = transA ? a_padded_shape[a_padded_shape.size() - 1] : a_padded_shape[a_padded_shape.size() - 2];
			size_t a_cols = transA ? a_padded_shape[a_padded_shape.size() - 2] : a_padded_shape[a_padded_shape.size() - 1];

			size_t b_rows = transB ? b_padded_shape[b_padded_shape.size() - 1] : b_padded_shape[b_padded_shape.size() - 2];
			size_t b_cols = transB ? b_padded_shape[b_padded_shape.size() - 2] : b_padded_shape[b_padded_shape.size() - 1];

		

			if (a_cols != b_rows) {
				INFERNO_LOG_ERROR() << "Incompatible dimensions for matrix multiplication" << std::endl;
				exit(1);
			}


			// Extract batch dims, take everything on the left but the last two dimensions
			std::vector<size_t> a_batch_shape(a_padded_shape.begin(), a_padded_shape.end() - 2);
			std::vector<size_t> b_batch_shape(b_padded_shape.begin(), b_padded_shape.end() - 2);				

			//figure out the final batch dimensions. 
			std::vector<size_t> broadcast_batch_shape = Tensor::get_broadcast_shape(a_batch_shape, b_batch_shape);	

			//figure out the final shape size
			std::vector<size_t> out_shape = broadcast_batch_shape;
			out_shape.push_back(a_rows);
			out_shape.push_back(b_cols);

											
			//get pointers to data
			AT* aptr = GetImpl(A)->data_as_ptr<AT>();
			BT* bptr = GetImpl(B)->data_as_ptr<BT>();


			bool gradreq = false;
			if (Inferno::grad_enabled) {
				gradreq = A.requires_grad() || B.requires_grad();
			}

			Inferno::Tensor out(dtype_of_v<RT>, out_shape, "matmul", A.device(), gradreq);

			auto implout = GetImpl(out);
			RT* optr = implout->data_as_ptr<RT>();


			switch (A.device().m_type) {

			////////////////////////////////////////////////////
			// CPU Code Path
			////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal matmul path" << std::endl;
				//TODO: need to add trans to this and use it
				cpu_matmul<AT, BT, RT>(aptr, bptr, optr, a_padded_shape, a_padded_strides, b_padded_shape, b_padded_strides, out_shape);
				break;

			////////////////////////////////////////////////////
			// CUDA Code Path
			////////////////////////////////////////////////////			
			case DeviceType::CUDA:

				INFERNO_LOG_DEBUG() << "CUDA Code path - Using matmul path" << std::endl;

				const size_t M = a_rows;
				const size_t K = a_cols;
				const size_t N = b_cols;

				const std::vector<size_t> out_batch_shape(out_shape.begin(), out_shape.end() - 2);
				const size_t batch_count_sz = product(out_batch_shape);
				const bool is_batched = batch_count_sz > 1;

				if constexpr (cublas_supported_v<AT, BT, RT>) {

					// Plain 2D fast path uses CublasSgemm (or cublasDgemm if dtype is double)
					if (A.is_contiguous() && B.is_contiguous() && !is_batched) {
						g_mmcountcublasSgemm++;						
						g_matmul_counts[label]++;
						cublas_mm_row_major<AT, BT, RT>(aptr, bptr, optr, M, K, N, transA, transB);
					}

					// Batched fast path can use this is either but not both the matrices are just a batch of 1
					// uses cublasSgemmStridedBatched
					else if (A.is_contiguous() && B.is_contiguous() && can_use_strided_batched_fastpath(a_batch_shape, b_batch_shape, out_batch_shape))	{

						long long strideA = 0;
						long long strideB = 0;
						long long strideC = static_cast<long long>(M * N);

						size_t A_phys_rows = a_padded_shape[max_rank - 2];
						size_t A_phys_cols = a_padded_shape[max_rank - 1];

						size_t B_phys_rows = b_padded_shape[max_rank - 2];
						size_t B_phys_cols = b_padded_shape[max_rank - 1];

						long long physicalStrideA = static_cast<long long>(A_phys_rows * A_phys_cols);
						long long physicalStrideB = static_cast<long long>(B_phys_rows * B_phys_cols);
						long long physicalStrideC = static_cast<long long>(M * N);

						if (a_batch_shape == out_batch_shape) {
							//strideA = static_cast<long long>(M * K);
							strideA = physicalStrideA;
						}
						else if (all_ones(a_batch_shape)) {
							strideA = 0;
						}
						else {
							strideA = -1;
						}

						if (b_batch_shape == out_batch_shape) {
							//strideB = static_cast<long long>(K * N);
							strideB = physicalStrideB;
						}
						else if (all_ones(b_batch_shape)) {
							strideB = 0;
						}
						else {
							strideB = -1;
						}

						if (strideA >= 0 && strideB >= 0) {
							g_mmcountcublasSgemmStridedBatched++;
							g_matmul_counts[label]++;
							cublas_mm_strided_batched_row_major<AT, BT, RT>(
								aptr,
								bptr,
								optr,
								M,
								K,
								N,
								strideA,
								strideB,
								strideC,
								static_cast<int>(batch_count_sz),
								transA,
								transB
							);
						}
						else {
							g_mmcountslow++;
							g_matmul_counts[label]++;
							cuda_matmul<AT, BT, RT>(
								aptr, bptr, optr,
								a_padded_shape, a_padded_strides,
								b_padded_shape, b_padded_strides,
								out_shape,
								transA,
								transB
							);
						}
					}

					// Fallback
					else {
						g_mmcountslow++;
						g_matmul_counts[label]++;
						cuda_matmul<AT, BT, RT>(
							aptr, bptr, optr,
							a_padded_shape, a_padded_strides,
							b_padded_shape, b_padded_strides,
							out_shape,
							transA,
							transB
						);
					}
				}
				else { //cublas not supported, so failback to slow 
					g_mmcountslow++;
					g_matmul_counts[label]++;
					cuda_matmul<AT, BT, RT>(
						aptr, bptr, optr,
						a_padded_shape, a_padded_strides,
						b_padded_shape, b_padded_strides,
						out_shape,
						transA,
						transB
					);
				}

				break;
			}

			return out;
			});

	}


	Tensor matmul_nt(const Tensor& A, const Tensor& B, std::string label) {

		return dispatchAnyTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;

			if (A.ndim() != 2 || B.ndim() != 2) {
				INFERNO_LOG_ERROR() << "matmul_nt expects 2D tensors" << std::endl;
				std::cout << A << std::endl;
				std::cout << B << std::endl;
				exit(1);
			}

			if (!A.is_contiguous() || !B.is_contiguous()) {
				INFERNO_LOG_ERROR() << "matmul_nt expects contiguous tensors" << std::endl;
				exit(1);
			}

			if (A.device() != B.device()) {
				INFERNO_LOG_ERROR() << "matmul_nt expects tensors on same device" << std::endl;
				exit(1);
			}



			const size_t M = A.shape()[0];
			const size_t K = A.shape()[1];

			const size_t N = B.shape()[0];
			const size_t Kb = B.shape()[1];

			if (K != Kb) {
				INFERNO_LOG_ERROR() << "matmul_nt incompatible shapes" << std::endl;
				exit(1);
			}

			Tensor out(dtype_of_v<RT>, { M, N }, "matmul_nt", A.device());

			const AT* aptr = GetImpl(A)->data_as_ptr<AT>();
			const BT* bptr = GetImpl(B)->data_as_ptr<BT>();
			RT* optr = GetImpl(out)->data_as_ptr<RT>();

			switch (A.device().m_type) {
			case DeviceType::CUDA:
				if constexpr (std::is_same_v<AT, RT> && std::is_same_v<BT, RT> && cublas_supported_v<AT, BT, RT>) {

					g_mmcountcublasSgemm++;
					g_matmul_counts[label]++;

					//this will only ever be called with float or double types
					cublas_mm_row_major_nt<RT>(
						aptr,
						bptr,
						optr,
						M,
						K,
						N
					);
				}
				else {
					INFERNO_LOG_ERROR() << "matmul_nt unsupported dtype for cuBLAS or dtypes were mixed" << std::endl;
					exit(1);
				}
				break;

			default:
				INFERNO_LOG_ERROR() << "matmul_nt currently only supports CUDA" << std::endl;
				exit(1);
			}

			return out;
			});
	}

	Tensor matmul_tn(const Tensor& A, const Tensor& B, std::string label) {

		return dispatchAnyTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;

			if (A.ndim() != 2 || B.ndim() != 2) {
				INFERNO_LOG_ERROR() << "matmul_nt expects 2D tensors" << std::endl;
				exit(1);
			}

			if (!A.is_contiguous() || !B.is_contiguous()) {
				INFERNO_LOG_ERROR() << "matmul_nt expects contiguous tensors" << std::endl;
				exit(1);
			}

			if (A.device() != B.device()) {
				INFERNO_LOG_ERROR() << "matmul_nt expects tensors on same device" << std::endl;
				exit(1);
			}



			const size_t M = A.shape()[0];
			const size_t K = A.shape()[1];

			const size_t N = B.shape()[0];
			const size_t Kb = B.shape()[1];

			if (K != Kb) {
				INFERNO_LOG_ERROR() << "matmul_nt incompatible shapes" << std::endl;
				exit(1);
			}

			Tensor out(dtype_of_v<RT>, { M, N }, "matmul_nt", A.device());

			const AT* aptr = GetImpl(A)->data_as_ptr<AT>();
			const BT* bptr = GetImpl(B)->data_as_ptr<BT>();
			RT* optr = GetImpl(out)->data_as_ptr<RT>();

			switch (A.device().m_type) {
			case DeviceType::CUDA:
				if constexpr (std::is_same_v<AT, RT> && std::is_same_v<BT, RT> && cublas_supported_v<AT, BT, RT>) {

					g_mmcountcublasSgemm++;
					g_matmul_counts[label]++;

					//this will only ever be called with float or double types
					cublas_mm_row_major_tn<RT>(
						aptr,
						bptr,
						optr,
						M,
						K,
						N
					);
				}
				else {
					INFERNO_LOG_ERROR() << "matmul_nt unsupported dtype for cuBLAS or dtypes were mixed" << std::endl;
					exit(1);
				}
				break;

			default:
				INFERNO_LOG_ERROR() << "matmul_nt currently only supports CUDA" << std::endl;
				exit(1);
			}

			return out;
			});
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function transpose_impl
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor transpose_impl(const Tensor& A, int dima, int dimb) {


		if (dima < 0)
			dima += static_cast<int>(A.ndim());

		if (dimb < 0)
			dimb += static_cast<int>(A.ndim());

		if (dima < 0 || dima >= static_cast<int>(A.ndim()) ||
			dimb < 0 || dimb >= static_cast<int>(A.ndim())) {
			INFERNO_LOG_ERROR() << "Transpose: dimension out of range" << std::endl;
			exit(1);
		}

		if (dima == dimb) {
			INFERNO_LOG_ERROR() << "Transpose: Dim A and Dim B match" << std::endl;
			exit(1);
		}	
						

		std::vector<size_t> newshape = A.shape();
		std::vector<size_t> newstrides = A.strides();					
		

		std::swap(newshape[dima], newshape[dimb]);
		std::swap(newstrides[dima], newstrides[dimb]);

		INFERNO_LOG_DEBUG() << "AGN Code path - Using normal transpose path" << std::endl;

		
		Tensor out = make_view(A,newshape,newstrides,GetImpl(A)->offset(),"transpose_"+A.name());

		return out;

	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function GetImpl/SetImpl
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::shared_ptr<TensorImpl> GetImpl(Tensor & t) {
		return t.m_impl;
	}

	std::shared_ptr<TensorImpl> GetImpl(const Tensor & t) {
		return t.m_impl;
	}

	void SetImpl(Tensor& t, std::shared_ptr<TensorImpl> impl) {
		t.m_impl = impl;
	}
			


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function make_view
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	Tensor make_view(const Tensor& base,const std::vector<size_t>& new_shape,const std::vector<size_t>& new_strides,size_t new_storage_offset,const std::string& name)
	{
		Tensor out;

		auto base_impl = GetImpl(base);
		auto impl = std::make_shared<TensorImpl>();

		impl->data() = base_impl->data();
		impl->shape() = new_shape;
		impl->strides() = new_strides;
		impl->offset() = new_storage_offset;
		impl->dtype() = base_impl->dtype();
		impl->device() = base_impl->device();
		impl->set_is_view(true);
	
		impl->name() = name;		
		impl->id() = Inferno::IDBroker::GenID();			
		impl->set_requires_grad(base_impl->requires_grad());
		//Inferno::NodeTracker::addID(this->m_id, this->m_name);

		out.device() = base.device();
		
		SetImpl(out,impl);
		return out;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function slice_impl
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor slice_impl(const Tensor& A, int axis, const size_t start, const size_t end, const size_t step) {

		if (axis < 0)
			axis += static_cast<int>(A.shape().size());

		if (axis < 0 || axis >= static_cast<int>(A.shape().size())) {
			INFERNO_LOG_ERROR() << "Slice axis out of bounds.Axis specified : " << std::to_string(axis) << " Tensor rank: " << std::to_string(A.shape().size()) << std::endl;
			exit(1);
		}

		if (step == 0) {
			INFERNO_LOG_ERROR() << "Slice step cannot be 0." << std::endl;
			exit(1);
		}

		const size_t axis_size = A.shape()[axis];

		if (start >= axis_size) {
			INFERNO_LOG_ERROR() << "Slice start out of bounds. Start: " << std::to_string(start) << " Axis size: " << std::to_string(axis_size) << std::endl;
			exit(1);
		}

		if (end >= axis_size) {
			INFERNO_LOG_ERROR() << "Slice end out of bounds. End: " << std::to_string(end) << " Axis size: " << std::to_string(axis_size) << std::endl;
			exit(1);
		}

		if (end < start) {
			INFERNO_LOG_ERROR() << "Slice end must be >= start. Start: " << std::to_string(start) << " End: " << std::to_string(end) << std::endl;
			exit(1);
		}

		std::vector<size_t> newshape = A.shape();
		std::vector<size_t> newstrides = A.strides();

		// inclusive end:
		// count = ceil((end - start + 1) / step)
		const size_t span = end - start;
		const size_t count = (span / step) + 1;

		newshape[axis] = count;
		newstrides[axis] *= step;

		size_t offset = A.offset() + A.strides()[axis] * start;

		INFERNO_LOG_DEBUG() << "AGN Code path - Using normal slice path" << std::endl;
		Tensor view = make_view(A, newshape, newstrides, offset, "slice_of_" + A.name());

		if ((Inferno::grad_enabled) && (A.requires_grad())) {
			GetImpl(view)->gradfn() = std::make_shared<SliceBackward>(A, axis, start, step, view.shape());
			
		}

		return view;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function reshape_impl
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	Tensor reshape_impl(const Tensor& A, const std::vector<size_t>& newshape) {

		std::vector<size_t> newstrides = Tensor::calculate_strides(newshape);

		INFERNO_LOG_DEBUG() << "AGN Code path - Using normal reshape path" << std::endl;
		
		Tensor out = make_view(A, newshape, newstrides, A.offset(), "reshape_of_" + A.name());

		if ((Inferno::grad_enabled) && (A.requires_grad())) {
			GetImpl(out)->gradfn() = std::make_shared<ReshapeBackward>(A);

		}


		return out;


	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function concat
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor concat(const std::vector<Tensor>& tensors, int axis) {


		//if there are no tensors in the list, error out
		if (tensors.empty()) {
			INFERNO_LOG_ERROR() << "concat: tensor list is empty." << std::endl;
			exit(1);
		}


		//get the first tensor in the list
		const Tensor& first = tensors[0];

		//get the number of dimensions in the first tensor
		const size_t ndim = first.ndim();

		//if there are no dimensions, error out
		if (ndim == 0) {
			INFERNO_LOG_ERROR() << "concat: scalar tensors not supported." << std::endl;
			exit(1);
		}


		//convert negative axis specification to positive
		if (axis < 0) {
			axis += static_cast<int>(ndim);
		}


		//verify the axis specified is within the range of the tensor
		if (axis < 0 || axis >= static_cast<int>(ndim)) {
			INFERNO_LOG_ERROR() << "concat: axis out of bounds." << std::endl;
			exit(1);
		}


		//get the type and device
		DType dtype = first.dtype();
		Device device = first.device();


		//the shape of the final tensor will be the same as the first one except along the dimension we want to concatenate on
		std::vector<size_t> out_shape = first.shape();		
		out_shape[axis] = 0;   //not required, but just set it to 0 for S&G's



		//we are going to loop over every source tensor
		for (size_t i = 0; i < tensors.size(); ++i) {
			const Tensor& t = tensors[i];  //alias


			//current tensor data type does not match the first one
			if (t.dtype() != dtype) {
				INFERNO_LOG_ERROR() << "concat: dtype mismatch." << std::endl;
				exit(1);
			}

			//current tensor device does not match the first one
			if (t.device() != device) {
				INFERNO_LOG_ERROR() << "concat: device mismatch." << std::endl;
				exit(1);
			}

			//current tensor rank does not match the first one
			if (t.ndim() != ndim) {
				INFERNO_LOG_ERROR() << "concat: rank mismatch." << std::endl;
				exit(1);
			}

			//make sure all the dimensions match except for the concat axis (the can be different sizes)
			for (size_t d = 0; d < ndim; ++d) {
				if (d == static_cast<size_t>(axis)) continue;

				if (t.shape()[d] != first.shape()[d]) {
					INFERNO_LOG_ERROR() << "concat: shapes must match on all non-concat axes." << std::endl;
					exit(1);
				}
			}

			//accumulate the concat axis of the tensor to the total
			//e.g.  concat on axis 2
			//      Tensor 1 size = 3
			//      Tensor 2 size = 5
			//      Tensor 3 size = 2
			//      Total size     10
			out_shape[axis] += t.shape()[axis];
		}


		bool req_grad = false;
		for (const auto& t : tensors) {
			if (t.requires_grad()) {
				req_grad = true;
				break;
			}
		}
		//GetImpl(out)->set_requires_grad(req_grad);


		//create the base for the output Tensor
		Tensor out(dtype, out_shape, "concat", device, req_grad);

		// prefix starts along concat axis
		std::vector<size_t> axis_starts(tensors.size(), 0);
		size_t running = 0;

		//loop over all the tensors and figure out how far down the axis in the output each one of them would start
		//e.g. start0--->3----->8-->10end
		for (size_t i = 0; i < tensors.size(); ++i) {
			axis_starts[i] = running;
			running += tensors[i].shape()[axis];
		}

		// flatten source metadata
		// were going to save all of the shapes,strides and offsets for all the tensors and pass those to the device specific concat function
		std::vector<size_t> src_shapes_flat;
		std::vector<size_t> src_strides_flat;
		std::vector<size_t> src_offsets;
		src_shapes_flat.reserve(tensors.size() * ndim);    // all the shapes
		src_strides_flat.reserve(tensors.size() * ndim);   // all the strides
		src_offsets.reserve(tensors.size());               // all the offsets

		for (const auto& t : tensors) {
			for (size_t d = 0; d < ndim; ++d) {
				src_shapes_flat.push_back(t.shape()[d]);
				src_strides_flat.push_back(t.strides()[d]);
			}
			src_offsets.push_back(t.offset());
		}

		

		//if all tensors are contiguous we can use the optimized contiguious concat functions for speed increase.
		bool fast_mode = true;
		for (const auto& t : tensors) {
			if (!t.is_contiguous()) {
				fast_mode = false;
				break;
			}
		}

		dispatchAny(dtype, [&](auto TagA) {
			using AT = typename decltype(TagA)::type;


			//save all the pointers for the source tensors
			std::vector<const AT*> src_ptrs;
			src_ptrs.reserve(tensors.size());
			for (const auto& t : tensors) {
				auto blah = GetImpl(t)->data_as_ptr<AT>();
				src_ptrs.push_back(GetImpl(t)->data_as_ptr<AT>());
			}


			//get the pointer of the output 
			AT* optr = GetImpl(out)->data_as_ptr<AT>();

			switch (device.m_type) {

			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal concat path" << std::endl;
				cpu_concat<AT>(
					src_ptrs,
					optr,
					src_shapes_flat,
					src_strides_flat,
					src_offsets,
					axis_starts,
					out.shape(),
					out.strides(),
					out.offset(),
					out.numel(),
					static_cast<size_t>(axis),
					ndim
				);
				break;

			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal concat path" << std::endl;
				cuda_concat<AT>(
					src_ptrs,
					optr,
					src_shapes_flat,
					src_strides_flat,
					src_offsets,
					axis_starts,
					out.shape(),
					out.strides(),
					out.offset(),
					out.numel(),
					static_cast<size_t>(axis),
					ndim
				);
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}
			});

		if ((Inferno::grad_enabled) && (req_grad)) {
			INFERNO_LOG_DEBUG() << "Concat - Making a ConcatBackward node" << std::endl;
			GetImpl(out)->gradfn() = std::make_shared<ConcatBackward>(tensors, axis);

		}

		return out;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function select
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor select(const Tensor& A, int axis, size_t index) {
		const auto& shape = A.shape();
		const auto& strides = A.strides();
		const int ndim = static_cast<int>(shape.size());

		int ax = axis;
		if (ax < 0) ax += ndim;

		if (ax < 0 || ax >= ndim) {
			INFERNO_LOG_ERROR() << "select: invalid axis" << std::endl;
			exit(1);
		}

		if (index >= shape[ax]) {
			INFERNO_LOG_ERROR() << "select: index out of bounds" << std::endl;
			exit(1);
		}

		std::vector<size_t> new_shape = shape;
		std::vector<size_t> new_strides = strides;

		size_t new_offset = GetImpl(A)->offset() + index * strides[ax];

		new_shape.erase(new_shape.begin() + ax);
		new_strides.erase(new_strides.begin() + ax);

		INFERNO_LOG_DEBUG() << "AGN Code path - Using normal select path" << std::endl;
		Tensor out = make_view(A, new_shape, new_strides, new_offset, A.name() + ".select");

		if (Inferno::grad_enabled && A.requires_grad()) {
			INFERNO_LOG_DEBUG() << "Select - Making a SelectBackward node" << std::endl;
			GetImpl(out)->gradfn() = std::make_shared<SelectBackward>(A, ax, index);
		}

		return out;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function masked_fill
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor masked_fill(const Tensor& input, const Tensor& mask, float value)
	{
		// output shape should match input shape
		Tensor out(input.dtype(), input.shape(), "masked_fill_out", input.device(), input.requires_grad());

		// You should probably also validate broadcast compatibility here
		// between mask.shape() and input.shape().

		
		dispatchAny(input.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;

			dispatchInt(mask.dtype(), [&](auto TM) {
				using MT = typename decltype(TM)::type;

				const AT* iptr = GetImpl(input)->data_as_ptr<AT>();
				const MT* mptr = GetImpl(mask)->data_as_ptr<MT>();
				AT* optr = GetImpl(out)->data_as_ptr<AT>();



				switch (input.device().m_type) {

					////////////////////////////////////////////////////
					// CPU Code Path
					////////////////////////////////////////////////////
				case DeviceType::CPU:
					INFERNO_LOG_DEBUG() << "CPU Code path - Using normal masked fill path" << std::endl;
					cpu_masked_fill<AT, MT>(
						iptr,mptr,optr,
						input.shape(),input.strides(),input.offset(),
						mask.shape(),mask.strides(),mask.offset(),
						out.numel(),
						static_cast<AT>(value)
					);
					break;

					////////////////////////////////////////////////////
					// CUDA Code Path
					////////////////////////////////////////////////////
				case DeviceType::CUDA: {
					INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal masked fill path" << std::endl;
					cuda_masked_fill<AT, MT>(iptr, mptr, optr, input.shape(), input.strides(), input.offset(), mask.shape(), mask.strides(), mask.offset(), out.numel(), static_cast<AT>(value));
					break;
				}
				default:
					INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
					exit(1);
				}
			});
		});
	
		if (input.requires_grad()) {			
			GetImpl(out)->gradfn() = std::make_shared<MaskedFillBackward>(input, mask);
		}

		return out;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function contiguous_impl
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor contiguous_impl(const Tensor& A) {
		// already contiguous -> no need to copy
		if (A.is_contiguous()) {
			return A;
		}

		bool gradreq = Inferno::grad_enabled && A.requires_grad();

		Tensor out(A.dtype(), A.shape(), "contiguous_of_" + A.name(), A.device(), gradreq);


		dispatchAny(A.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			//get pointers to data
			auto ImplA = GetImpl(A);
			auto Implout = GetImpl(out);

			AT* aptr = ImplA->data_as_ptr<AT>();
			AT* optr = Implout->data_as_ptr<AT>();

			std::vector<size_t> shape = A.shape();
			std::vector<size_t> strides = A.strides();
			size_t offset = A.offset();
			size_t N = A.numel();

			if (A.device().is_cpu()) {
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal contiguous path" << std::endl;
				cpu_contiguous_copy<AT>(aptr, optr, shape, strides, offset, N);
			}
			else {
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal contiguous path" << std::endl;
				cuda_contiguous_copy<AT>(aptr, optr, shape, strides, offset, N);
			}
			});

		if (Inferno::grad_enabled && A.requires_grad()) {
			INFERNO_LOG_DEBUG() << "Select - Making a ContiguousBackward node" << std::endl;
			GetImpl(out)->gradfn() = std::make_shared<ContiguousBackward>(A);
		}

		return out;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function triu
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor triu(const Tensor& A, int diagonal)
	{
		if (A.ndim() < 2) {
			INFERNO_LOG_ERROR() << "triu requires tensor rank >= 2" << std::endl;
			exit(1);
		}

		Tensor out(A.dtype(), A.shape(), "triu(" + A.name() + ")", A.device());

		dispatchAny(A.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* aptr = GetImpl(A)->data_as_ptr<AT>();
			AT* optr = GetImpl(out)->data_as_ptr<AT>();

			switch (out.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				INFERNO_LOG_DEBUG() << "CPU Code path - Using normal triu path" << std::endl;
				cpu_triu<AT>(aptr, optr, A.shape(), A.strides(), A.offset(), out.numel(), diagonal);
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using normal triu path" << std::endl;
				cuda_triu<AT>(aptr, optr, A.shape(), A.strides(), A.offset(), out.numel(), diagonal);
				break;

			default:
				INFERNO_LOG_ERROR() << "Invalid device type" << std::endl;
				exit(1);
			}


			});

		return out;

	}

	/*Tensor flash_attention_simple_forward(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal) {


		std::vector<size_t> qshape = Q.shape();
		std::vector<size_t> kshape = K.shape();
		std::vector<size_t> vshape = V.shape();

		if (qshape.size() != 4 || kshape.size() != 4 || vshape.size() != 4) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR , "flash_attention_simple_forward expects q, k, v to be [B, H, T, D]");
			exit(1);
		}

		if (qshape != kshape || qshape != vshape) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "flash_attention_simple_forward expects q, k, v to have the same shape");
			exit(1);
		}

		if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "flash_attention_simple_forward expects q, k, v to be contiguous");
			exit(1);
		}

		if (Q.device() != K.device() || Q.device() != V.device()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "flash_attention_simple_forward expects q, k, v to be on the same device");
			exit(1);
		}

		if (!Q.device().is_cuda()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "flash_attention_simple_forward currently only supports CUDA tensors");
			exit(1);
		}

		if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "flash_attention_simple_forward expects q, k, v to have the same dtype");
			exit(1);
		}

		const size_t B = qshape[0];
		const size_t H = qshape[1];
		const size_t T = qshape[2];
		const size_t D = qshape[3];

		Inferno::Tensor out(Q.dtype(), qshape, "flash_attention_simple_forward_out", Q.device());		

		dispatchFloat(Q.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* qptr = GetImpl(Q)->data_as_ptr<AT>();
			const AT* kptr = GetImpl(K)->data_as_ptr<AT>();
			const AT* vptr = GetImpl(V)->data_as_ptr<AT>();
			AT* optr = GetImpl(out)->data_as_ptr<AT>();

			switch (out.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using normal triu path");
				//cpu_flash_attention_simple_forward<AT>(qptr, kptr, vptr, optr,	B, H, T, D, causal);
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using normal triu path");
				cuda_flash_attention_simple_forward<AT>(qptr, kptr, vptr, optr, B, H, T, D, causal);
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}


			});

		return out;
	}*/

	Tensor flash_attention_simple_forward(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal) {

		std::vector<size_t> qshape = Q.shape();
		std::vector<size_t> kshape = K.shape();
		std::vector<size_t> vshape = V.shape();

		if (qshape.size() != 4 || kshape.size() != 4 || vshape.size() != 4) {
			INFERNO_LOG_ERROR() << "flash_attention_simple_forward expects Q, K, V to be [B, H, T, D]" << std::endl;
			exit(1);
		}

		if (qshape != kshape || qshape != vshape) {
			INFERNO_LOG_ERROR() << "flash_attention_simple_forward expects Q, K, V to have the same shape" << std::endl;
			exit(1);
		}

		if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()) {
			INFERNO_LOG_ERROR() << "flash_attention_simple_forward expects Q, K, V to be contiguous" << std::endl;
			exit(1);
		}

		if (Q.device() != K.device() || Q.device() != V.device()) {
			INFERNO_LOG_ERROR() << "flash_attention_simple_forward expects Q, K, V to be on the same device" << std::endl;
			exit(1);
		} 

		if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
			INFERNO_LOG_ERROR() << "flash_attention_simple_forward expects Q, K, V to have the same dtype" << std::endl;
			exit(1);
		}

		const size_t B = qshape[0];
		const size_t H = qshape[1];
		const size_t T = qshape[2];
		const size_t D = qshape[3];

		Tensor out(Q.dtype(), qshape, "flash_attention_simple_forward_out", Q.device());

		dispatchFloat(Q.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* qptr = GetImpl(Q)->data_as_ptr<AT>();
			const AT* kptr = GetImpl(K)->data_as_ptr<AT>();
			const AT* vptr = GetImpl(V)->data_as_ptr<AT>();
			AT* optr = GetImpl(out)->data_as_ptr<AT>();

			switch (Q.device().m_type) {

			case DeviceType::CPU:
				INFERNO_LOG_ERROR() << "flash_attention_simple_forward CPU path is not implemented yet" << std::endl;
				exit(1);
				break;

			case DeviceType::CUDA:
				INFERNO_LOG_DEBUG() << "CUDA Code path - Using flash_attention_simple_forward" << std::endl;

				cuda_flash_attention_simple_forward2<AT>(
					qptr,
					kptr,
					vptr,
					optr,
					B,
					H,
					T,
					D,
					causal
				);
				break;

			default:
				INFERNO_LOG_ERROR() << "flash_attention_simple_forward invalid device type" << std::endl;
				exit(1);
			}
			});

		if (Inferno::grad_enabled && (Q.requires_grad() || K.requires_grad() || V.requires_grad())) {
			INFERNO_LOG_DEBUG() << "FlashAttention - Making a FlashAttentionBackward node" << std::endl;
			GetImpl(out)->gradfn() = std::make_shared<FlashAttentionBackward>(Q, K, V, out, causal);
			out.set_requires_grad(true);
			
		}


		return out;
	}

	/*Tensor flash_attention_simple_forward(const Tensor& qkv, bool causal) {


		Tensor out(qkv.dtype(), qkv.shape(), "flash_attention_simple_forward_out", qkv.device());

		dispatchFloat(qkv.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* qkvptr = GetImpl(qkv)->data_as_ptr<AT>();			
			AT* optr = GetImpl(out)->data_as_ptr<AT>();

			switch (qkv.device().m_type) {

			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "flash_attention_simple_forward CPU path is not implemented yet");
				exit(1);
				break;

			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using flash_attention_simple_forward");

				//cuda_flash_attention_simple_forward2<AT>(qkvptr, B, H, T, D, causal);
				break;

			default:
				Logger::Append(
					Logger::LogLevel::LOGLEVEL_ERROR, "flash_attention_simple_forward invalid device type");
				exit(1);
			}
			});

		if (Inferno::grad_enabled && qkv.requires_grad()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "FlashAttention - Making a FlashAttentionBackward node");
			//GetImpl(out)->gradfn() = std::make_shared<FlashAttentionBackward>(Q, K, V, out, causal);
		}


		return out;
	}*/



	

	Tensor flash_attention_bigdaddy_forward_my_version_check(const Tensor& qkv, size_t num_heads, bool causal) {

		std::vector<size_t> shape = qkv.shape();

		if (shape.size() != 3) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward expects qkv shape [B,T,3C]" << std::endl;
			exit(1);
		}

		if (!qkv.is_contiguous()) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward expects qkv to be contiguous" << std::endl;
			exit(1);
		}

		size_t B = shape[0];
		size_t Tseq = shape[1];
		size_t threeC = shape[2];

		if (threeC % 3 != 0) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward last dim must be 3*C" << std::endl;
			exit(1);
		}

		size_t C = threeC / 3;

		if (C % num_heads != 0) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward C must be divisible by num_heads" << std::endl;
			exit(1);
		}

		size_t H = num_heads;
		size_t D = C / H;

		Tensor out(qkv.dtype(), { B, Tseq, C }, "flash_attention_bigdaddy_out", qkv.device());

		dispatchFloat(qkv.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* qkvptr = GetImpl(qkv)->data_as_ptr<AT>();
			AT* outptr = GetImpl(out)->data_as_ptr<AT>();

			switch (qkv.device().m_type) {
			case DeviceType::CUDA:				
				cuda_flash_block_my_version_check<AT>(
					qkvptr,
					outptr,
					B,
					Tseq,
					C,
					H,
					D,
					causal
				);
				break;

			default:
				INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward only supports CUDA for now" << std::endl;
				exit(1);
			}
			});

		if (Inferno::grad_enabled && qkv.requires_grad()) {
			INFERNO_LOG_DEBUG() << "FlashAttentionBigDaddy - Making backward node" << std::endl;

			// You will need a new backward node for this packed version.
			GetImpl(out)->gradfn() = std::make_shared<FlashAttentionBigDaddyBackwardFast>(qkv, out, num_heads, causal);

			out.set_requires_grad(true);
		}

		return out;
	}



	Tensor flash_attention_bigdaddy_forward(const Tensor& qkv, size_t num_heads, bool causal) {

		std::vector<size_t> shape = qkv.shape();

		if (shape.size() != 3) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward expects qkv shape [B,T,3C]" << std::endl;
			exit(1);
		}

		if (!qkv.is_contiguous()) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward expects qkv to be contiguous" << std::endl;
			exit(1);
		}

		size_t B = shape[0];
		size_t Tseq = shape[1];
		size_t threeC = shape[2];

		if (threeC % 3 != 0) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward last dim must be 3*C" << std::endl;
			exit(1);
		}

		size_t C = threeC / 3;

		if (C % num_heads != 0) {
			INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward C must be divisible by num_heads" << std::endl;
			exit(1);
		}

		size_t H = num_heads;
		size_t D = C / H;

		Tensor out(qkv.dtype(), { B, Tseq, C }, "flash_attention_bigdaddy_out", qkv.device());

		dispatchFloat(qkv.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* qkvptr = GetImpl(qkv)->data_as_ptr<AT>();
			AT* outptr = GetImpl(out)->data_as_ptr<AT>();

			switch (qkv.device().m_type) {
			case DeviceType::CUDA:
				cuda_flash_block<AT>(
					qkvptr,
					outptr,
					B,
					Tseq,
					C,
					H,
					D,
					causal
				);
				break;

			default:
				INFERNO_LOG_ERROR() << "flash_attention_bigdaddy_forward only supports CUDA for now" << std::endl;
				exit(1);
			}
			});

		if (Inferno::grad_enabled && qkv.requires_grad()) {
			INFERNO_LOG_DEBUG() << "FlashAttentionBigDaddy - Making backward node" << std::endl;

			// You will need a new backward node for this packed version.
			GetImpl(out)->gradfn() = std::make_shared<FlashAttentionBigDaddyBackwardFast>(qkv, out, num_heads, causal);

			out.set_requires_grad(true);
		}

		return out;
	}

}

