#pragma once
#include <numeric>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <memory>
#include <type_traits>
#include <inferno/util/random.h>
#include <inferno/core/dtype.h>
#include <inferno/core/device.h>
#include <inferno/core/dtype_traits.h>





namespace Inferno {

	class Tensor;
	class TensorImpl;
	class Storage;


	//void SetImpl(Tensor& t, std::shared_ptr<TensorImpl> impl);
	//std::shared_ptr<TensorImpl> GetImpl(Tensor& t);
	//std::shared_ptr<TensorImpl> GetImpl(const Tensor& t);

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

	class Tensor {

		friend Tensor add(const Tensor& A, const Tensor& B);
		friend Tensor subtract(const Tensor& A, const Tensor& B);
		friend Tensor multiply(const Tensor& A, const Tensor& B);
		friend Tensor divide(const Tensor& A, const Tensor& B);

		friend Tensor negate(const Tensor& A);

		friend Tensor concat(const std::vector<Inferno::Tensor>& tensors, int axis);
		friend std::ostream& operator<<(std::ostream& os, const Tensor& t);


	public:

		Tensor() = default;

		//std::vector version for empty
		Tensor(DType dtype, std::vector<size_t> shape, std::string name, Inferno::Device device, bool requires_grad = false);

		//initializer_list version for empty
		Tensor(DType dtype, std::initializer_list<size_t> shape, std::string name, Inferno::Device device, bool requires_grad = false);


		//std::vector version for data 
		template <typename T>
		Tensor(DType dtype, const std::vector<T>& data, std::initializer_list<size_t> shape, std::string name, Inferno::Device device = Inferno::Device::cpu(), bool requires_grad = false) {
			init_from_raw(dtype, data.data(), data.size(), sizeof(T), std::vector<size_t>(shape), name, device, requires_grad);
			m_device = device;
			//m_id = m_impl->id();

		}

		//initializer_list version for data 
		template <typename T>
		Tensor(DType dtype, const std::initializer_list<T>& data, std::initializer_list<size_t> shape, std::string name, Inferno::Device device = Inferno::Device::cpu(), bool requires_grad = false) {			
			std::vector<T> tmp(data);
			init_from_raw(dtype, tmp.data(), tmp.size(), sizeof(T), std::vector<size_t>(shape), name, device, requires_grad);
			m_device = device;
			//m_id = m_impl->id();
		}

		

		~Tensor() = default;

		//Device defs
		Device& device();
		const Device& device() const { return m_device; }
		Tensor to(const Device& dst) const;

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
		Tensor Getgrad() const;

		std::vector<size_t>& shape();
		std::vector<size_t>& shape() const;
		std::vector<size_t>& strides();
		std::vector<size_t>& strides() const;
		size_t& offset();
		size_t& offset() const;
		std::string name();
		std::string name() const;
		size_t ndim();
		size_t ndim() const;
		bool requires_grad();
		bool requires_grad() const;
		void set_requires_grad(bool flag);




		//properties defs
		size_t size() const;
		size_t numel() const;
		size_t dtype_size(DType dtype);
		bool is_contiguous() const;


		//functional
		//std::vector<size_t> calculate_strides(std::vector<size_t> shape);
		static std::vector<size_t> calculate_strides(const std::vector<size_t>& shape);



		//broadcasting
		static std::vector<size_t> get_broadcast_shape(std::vector<size_t> a, std::vector<size_t> b);
		Tensor broadcast_to(std::vector<size_t> desiredshape);


		//dup defs
		static Tensor ones_like(const Tensor& t);
		static Tensor ones(const std::vector<size_t>& shape,DType dtype,Device device,const std::string& name = "ones");

		static Tensor zeros_like(const Tensor& A);
		static Tensor zeros(const std::vector<size_t>& shape, DType dtype, Device device, const std::string& name = "zeros");

		static Tensor randn(Inferno::DType dtype, const std::initializer_list<size_t>& shape, const std::string& name, Inferno::Device device = Inferno::Device::cpu());

		//modifications
		Tensor transpose(int dima, int dimb);
		Tensor transpose(int dima, int dimb) const;
		Tensor unsqueeze(int dim);
		Tensor unsqueeze(int dim) const;
		Tensor slice(int axis, const size_t start, const size_t end, const size_t step = 1);
		Tensor reshape(const std::vector<size_t>& newshape) const;
		Tensor contiguous() const;
		void backward();

		double item_double_impl() const;
		int64_t item_int64_impl() const;



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

		template<typename T>
		T item() const {
			if constexpr (std::is_integral_v<T>) {
				return static_cast<T>(item_int64_impl());
			}
			else {
				return static_cast<T>(item_double_impl());
			}
		}



	private:

		void init_from_raw(DType dtype, const void* data, size_t count, size_t elem_size, const std::vector<size_t>& shape, const std::string& name, Inferno::Device device, bool requires_grad);

		friend void SetImpl(Tensor& t, std::shared_ptr<TensorImpl> impl);
		friend std::shared_ptr<TensorImpl> GetImpl(Tensor& t);
		friend std::shared_ptr<TensorImpl> GetImpl(const Tensor& t);
		std::shared_ptr<TensorImpl> m_impl;
		Device m_device;
		int m_id;


	};


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  scalar overloads addition
	// 
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator+(const Tensor& A, T other) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return A + scalar;
	}

	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator+(T other, const Tensor& A) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return scalar + A;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  scalar overloads subtraction
	// 
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator-(const Tensor& A, T other) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return A - scalar;
	}

	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator-(T other, const Tensor& A) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return scalar - A;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  scalar overloads multiplication
	// 
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  scalar overloads divsion
	// 
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator/(const Tensor& A, T other) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return A / scalar;
	}

	template <typename T>
		requires std::is_arithmetic_v<T>
	Tensor operator/(T other, const Tensor& A) {
		Tensor scalar(dtype_of_v<T>, std::vector<T>{ other }, { 1 }, "scalar", A.device());
		return scalar / A;
	}



}