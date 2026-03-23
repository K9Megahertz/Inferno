#include "ops.h"
#include "GradFN/addbackward.h"
#include "GradFN/subtractbackward.h"
#include "GradFN/multiplybackward.h"
#include "GradFN/dividebackward.h"
#include "GradFN/negatebackward.h"
#include "GradFN/mmbackward.h"
#include "GradFN/mselossbackward.h"


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
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Incompatible device types on tensor parameters in Add");
			exit(1);
		}

		return dispatchTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;



			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "add", A.device());

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();



			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_add(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_add(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if (Inferno::grad_enabled) {
				implout->gradfn() = std::make_shared<AddBackward>(A, B);
			}


			return out;
		});
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function add_nograd
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*Tensor add_nograd(const Tensor& A, const Tensor& B) {


		if (A.device() != B.device()) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Incompatible device types on tensor parameters in Add");
			exit(1);
		}

		return dispatchTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;



			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "add_nograd", A.device());

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();



			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_add(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_add(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}


			implout->gradfn() = nullptr;


			return out;
			});
	}*/


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
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Incompatible device types on tensor parameters in Add");
			exit(1);
		}

		return dispatchTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;



			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "subtract", A.device());

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();



			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_subtract(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_subtract(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if (Inferno::grad_enabled) {
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
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Incompatible device types on tensor parameters in Add");
			exit(1);
		}

		return dispatchTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;

			std::vector<size_t> broadcast_shape = Tensor::get_broadcast_shape(A.shape(), B.shape());

			auto implA = GetImpl(A);
			auto implB = GetImpl(B);

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();
			auto bptr = implB->data_as_ptr<BT>();


			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "multiply", A.device());

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();


			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_multiply(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");				
				cuda_multiply<AT, BT, RT>(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if (Inferno::grad_enabled) {
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
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Incompatible device types on tensor parameters in divide");
			exit(1);
		}

		return dispatchTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
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


			Inferno::Tensor out(dtype_of_v<RT>, broadcast_shape, "divide", A.device());

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<RT>();



			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_divide(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_divide(aptr, bptr, optr, A.shape(), B.shape(), out.shape(), out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if (Inferno::grad_enabled) {
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
				

		return dispatchOne(A.dtype(), [&](auto TA) {
			using AT = typename decltype(TA)::type;			

			auto implA = GetImpl(A);		

			//get pointers to data
			auto aptr = implA->data_as_ptr<AT>();

			Inferno::Tensor out(dtype_of_v<AT>, A.shape(), "negate", A.device());

			auto implout = GetImpl(out);
			auto optr = implout->data_as_ptr<AT>();


			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_negate(aptr, optr, out.numel());
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_negate<AT>(aptr, optr, out.numel());
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
				exit(1);
			}

			if (Inferno::grad_enabled) {
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

	Tensor matmul(const Tensor& A, const Tensor& B) {


		bool a_vec = (A.ndim() == 1);
		bool b_vec = (B.ndim() == 1);

		Tensor A2 =  make_view(A, A.shape(), A.strides(), 0, "A2");
		Tensor B2 =  make_view(B, B.shape(), B.strides(), 0, "B2");


		if (a_vec) {
			A2.shape() = { 1, A2.shape()[0] };
			A2.strides() = A2.calculate_strides(A2.shape());
		}

		if (b_vec) {
			B2.shape() = { B2.shape()[0], 1 };
			B2.strides() = B2.calculate_strides(B2.shape());
		}

		Tensor out = matmul_impl(A2, B2);

		if (a_vec && b_vec)
			out.shape()={1};   // scalar	
		else if (a_vec) 
			out.shape().erase(out.shape().begin() + 0);
		else if (b_vec) 
			out.shape().erase(out.shape().begin() + 1);	
			
		out.strides() = out.calculate_strides(out.shape());

		
		if (Inferno::grad_enabled) {
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

	Tensor matmul_impl(const Tensor& A, const Tensor& B) {

		return dispatchTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {
			using AT = typename decltype(TA)::type;
			using BT = typename decltype(TB)::type;
			using RT = promote_t<AT, BT>;

			size_t a_ndim = A.ndim();
			size_t b_ndim = B.ndim();

				

			// Ensure tensors have at least 1 dimension
			if (a_ndim < 1 || b_ndim < 1) {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Tensors must have at least 1 dimension for matmul");
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

			size_t a_rows = a_padded_shape[a_padded_shape.size() - 2];
			size_t a_cols = a_padded_shape[a_padded_shape.size() - 1];
			size_t b_rows = b_padded_shape[b_padded_shape.size() - 2];
			size_t b_cols = b_padded_shape[b_padded_shape.size() - 1];

			if (a_cols != b_rows) {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Incompatible dimensions for matrix multiplication");
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


			Inferno::Tensor out(dtype_of_v<RT>, out_shape, "matmul", A.device());

			auto implout = GetImpl(out);
			RT* optr = implout->data_as_ptr<RT>();


			switch (A.device().m_type) {

				////////////////////////////////////////////////////
				// CPU Code Path
				////////////////////////////////////////////////////
			case DeviceType::CPU:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
				cpu_matmul(aptr, bptr, optr, a_padded_shape, a_padded_strides, b_padded_shape, b_padded_strides, out_shape);
				break;

				////////////////////////////////////////////////////
				// CUDA Code Path
				////////////////////////////////////////////////////
			case DeviceType::CUDA:
				Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
				cuda_matmul<AT, BT, RT>(aptr, bptr, optr, a_padded_shape, a_padded_strides, b_padded_shape, b_padded_strides, out_shape);
				break;

			default:
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
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


		if (dima == dimb) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Transpose: Dim A and Dim B match");
			exit(1);
		}

						

		std::vector<size_t> newshape = A.shape();
		std::vector<size_t> newstrides = A.strides();		

			
		if (dima < 0)
			dima = dima + A.ndim();

		if (dimb < 0)
			dimb = dimb + A.ndim();

		std::swap(newshape[dima], newshape[dimb]);
		std::swap(newstrides[dima], newstrides[dimb]);

		Tensor out = make_view(A,newshape,newstrides,GetImpl(A)->offset(),"transpose_"+A.name());

		return out;

	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function GetImpl
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
		//impl->set_is_view(true);
		//impl->set_base(base);
		impl->name() = name;
		impl->id() = Inferno::IDBroker::GenID();			
		//Inferno::NodeTracker::addID(this->m_id, this->m_name);

		out.device() = base.device();
		SetImpl(out,impl);
		return out;
	}

}

