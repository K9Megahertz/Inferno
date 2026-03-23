#pragma once
#include <numeric>
#include <iostream>
#include <ostream>
#include <iomanip>
#include "tensorimpl.h"
#include "cuda/cudaops.h"
#include "gradengine/engine.h"


namespace Inferno {

	// tensor-scalar ops
	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator*(const Tensor& A, T other);

	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator*(T other, const Tensor& A);



	

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class Tensor
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	// 1) DType <-> C++ type mapping
	template <DType> struct DTypeToCpp;
	template <> struct DTypeToCpp<DType::Int32> { using type = int; };
	template <> struct DTypeToCpp<DType::Float32> { using type = float; };
	template <> struct DTypeToCpp<DType::Float64> { using type = double; };

	template <typename T> struct CppToDType;
	template <> struct CppToDType<int> { static constexpr DType value = DType::Int32; };
	template <> struct CppToDType<float> { static constexpr DType value = DType::Float32; };
	template <> struct CppToDType<double> { static constexpr DType value = DType::Float64; };

	template <DType DT>
	using cpp_type_t = typename DTypeToCpp<DT>::type;

	template <typename T>
	constexpr DType dtype_of_v = CppToDType<T>::value;

	// 2) Promotion rules (feel free to extend)
	template <typename A, typename B> struct Promote;
	template <> struct Promote<int, int> { using type = int; };
	template <> struct Promote<int, float> { using type = float; };
	template <> struct Promote<float, int> { using type = float; };
	template <> struct Promote<float, float> { using type = float; };
	template <> struct Promote<double, float> { using type = double; };
	template <> struct Promote<float, double> { using type = double; };
	template <> struct Promote<double, int> { using type = double; };
	template <> struct Promote<int, double> { using type = double; };
	template <> struct Promote<double, double> { using type = double; };


	//template <typename A> struct PromoteSingle;
	//template <> struct PromoteSingle<int> { using type = float; };
	//template <> struct PromoteSingle<float> { using type = float; };
	//template <> struct PromoteSingle<double> { using type = double; };


	template <typename A, typename B>
	using promote_t = typename Promote<A, B>::type;

	// 3) Tiny tags to pass types from runtime switches into templates
	template <typename T> struct Tag { using type = T; };

	template <typename F>
	auto dispatchOne(DType a_dt, F&& fn) {
		switch (a_dt) {
		case DType::Int32: return fn(Tag<int>{});
		case DType::Float32: return fn(Tag<float>{});
		case DType::Float64: return fn(Tag<double>{});
			break;
		}		
		Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Unsupported dtype combination");
		exit(1);
	}


	template <typename F>
	auto dispatchTwo(DType a_dt, DType b_dt, F&& fn) {
		switch (a_dt) {
		case DType::Int32: 
			switch (b_dt) {
			case DType::Int32: return fn(Tag<int>{}, Tag<int>{});
			case DType::Float32: return fn(Tag<int>{}, Tag<float>{});
			case DType::Float64: return fn(Tag<int>{}, Tag<double>{});
			}
			break;
		case DType::Float32: 
			switch (b_dt) {
			case DType::Int32: return fn(Tag<float>{}, Tag<int>{});
			case DType::Float32: return fn(Tag<float>{}, Tag<float>{});
			case DType::Float64: return fn(Tag<float>{}, Tag<double>{});
			}
			break;
		case DType::Float64: 
			switch (b_dt) {
			case DType::Int32: return fn(Tag<double>{}, Tag<int>{});
			case DType::Float32: return fn(Tag<double>{}, Tag<float>{});
			case DType::Float64: return fn(Tag<double>{}, Tag<double>{});
			}
			break;
		}
		Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Unsupported dtype combination");
		exit(1);
	}



	class Tensor;

	Tensor add(const Tensor& A, const Tensor& B);
	Tensor subtract(const Tensor& A, const Tensor& B);
	Tensor multiply(const Tensor& A, const Tensor& B);
	Tensor divide(const Tensor& A, const Tensor& B);

	Tensor transpose_impl(const Tensor& A, int dima, int dimb);

	Tensor negate(const Tensor& A);

	template <typename AT, typename BT, typename RT>
	void cpu_add_tensors(const AT aptr, const BT bptr, RT optr, std::vector<size_t> ashape, std::vector<size_t> bshape, std::vector<size_t> oshape, size_t numel);
	
	




	class Tensor {

		friend Tensor add(const Tensor& A, const Tensor& B);
		friend Tensor subtract(const Tensor& A, const Tensor& B);
		friend Tensor multiply(const Tensor& A, const Tensor& B);
		friend Tensor divide(const Tensor& A, const Tensor& B);

		friend Tensor transpose_impl(const Tensor& A, int dima, int dimb);

		friend Tensor negate(const Tensor& A);

		friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
		

	public:

		Tensor() = default;
		Tensor(DType dtype, std::vector<size_t> shape, std::string name, Inferno::Device device);

		template <typename T>
		Tensor(DType dtype, const std::vector<T>& data, std::vector<size_t> shape, std::string name, Inferno::Device device = Inferno::Device::cpu()) {
			m_impl = std::make_shared<TensorImpl>(dtype,data,shape,name,device);
			m_device = device;
			m_id = m_impl->id();
		
		}

		~Tensor() = default;

		//Device defs
		Device& device();
		const Device& device() const { return m_device; }
		Tensor to(const Device&& dst) const;

		//overload defs
		Tensor operator+(const Tensor& other) const;
		Tensor operator-(const Tensor& other) const;
		Tensor operator*(const Tensor& other) const;
		Tensor operator/(const Tensor& other) const;

		Tensor operator-() const;






		//accessor defs
		DType dtype();
		DType dtype() const;
		std::shared_ptr<Storage>& data();
		std::shared_ptr<Storage>& data() const;
		std::shared_ptr<Tensor>& grad();
		std::shared_ptr<Tensor>& grad() const;
		std::vector<size_t>& shape();
		std::vector<size_t>& shape() const;
		std::vector<size_t>& strides();
		std::vector<size_t>& strides() const;
		std::string name();
		std::string name() const;
		size_t ndim();
		size_t ndim() const;


		//properties defs
		size_t size() const;
		size_t numel() const;
		size_t dtype_size(DType dtype);


		//functional
		std::vector<size_t> calculate_strides(std::vector<size_t> shape);
		std::vector<size_t> calculate_strides(std::vector<size_t> shape) const;
		

		
		//broadcasting
		static std::vector<size_t> get_broadcast_shape(std::vector<size_t> a, std::vector<size_t> b);
		Tensor broadcast_to(std::vector<size_t> desiredshape);


		//dup defs
		static Tensor ones_like(const Tensor& t);
		static Tensor zeroes_like(const Tensor& A);


		//modifications
		Tensor transpose(int dima, int dimb);
		Tensor transpose(int dima, int dimb) const;		
		Tensor unsqueeze(int dim);
		Tensor unsqueeze(int dim) const;

		void backward();


		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//
		//  Function item
		//
		//
		//
		//
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		T item() const {

			return dispatchOne(dtype(), [&](auto TA) {
				using AT = typename decltype(TA)::type;

				auto dptr = GetImpl(*this)->data_as_ptr<AT>();
				return static_cast<T>(dptr[0]);
			});
		}


	private:

		friend void SetImpl(Tensor& t, std::shared_ptr<TensorImpl> impl);
		friend std::shared_ptr<TensorImpl> GetImpl(Tensor& t);
		friend std::shared_ptr<TensorImpl> GetImpl(const Tensor& t);
		std::shared_ptr<TensorImpl> m_impl;
		Device m_device;
		int m_id;
	

	};


	template <typename T>		
	requires std::is_arithmetic_v<T>
	Tensor operator*(const Tensor& A, T other) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return A * scalar;
	}

	template <typename T>		
	requires std::is_arithmetic_v<T>
	Tensor operator*(T other, const Tensor& A) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return scalar * A;
	}

	


	


}