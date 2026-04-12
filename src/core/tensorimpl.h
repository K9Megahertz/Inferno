#pragma once

#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <inferno/core/device.h>
#include <inferno/core/dtype.h>
#include "storage/storage.h"
#include "storage/cpustorage.h"
#include "storage/cudastorage.h"
#include "util/logger.h"
#include "util/idbroker.h"
#include "util/nodetracker.h"


namespace Inferno {


	class Tensor;
	class Node;
	class AccumulateGrad;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class TensorImpl
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////4
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class TensorImpl : public std::enable_shared_from_this<TensorImpl> {

	public:

		TensorImpl();		
		//TensorImpl(DType type, std::vector<size_t> shape, std::string name, Inferno::Device device, bool requires_grad = false);
		//TensorImpl(DType type, std::initializer_list<size_t> shape, std::string name, Inferno::Device device, bool requires_grad = false);
		TensorImpl(DType type, const std::vector<size_t> shape, std::string name, Inferno::Device device, bool requires_grad);
		TensorImpl(DType type, const void* data, const std::vector<size_t> shape, const std::string name, Inferno::Device device, bool requires_grad);
		

	
		~TensorImpl() { 
			Inferno::NodeTracker::removeID(this->m_id);
			m_grad_fn = nullptr;
			m_grad_accum = nullptr;

		}

		
		std::shared_ptr<Node> grad_edge();
		std::shared_ptr<Node> get_or_create_accumulate_grad();

		void *raw_ptr() {
			return m_data->raw_ptr();
		}		

		template<typename T>
		T* data_as_ptr() {
			return m_data->template storage_as_ptr<T>();
		}

		template<typename T>
		const T* data_as_ptr() const {
			return m_data->template storage_as_ptr<T>();
		}

	


		DType& dtype();
		const DType& dtype() const;


		std::shared_ptr<Node>& gradfn();
		const std::shared_ptr<Node>& gradfn() const;
		std::shared_ptr<Tensor>& grad();
		const std::shared_ptr<Tensor>& grad() const;
		std::shared_ptr<Storage>& data();
		const std::shared_ptr<Storage>& data() const;
		std::vector<size_t>& shape();
		const std::vector<size_t>& shape() const;
		std::vector<size_t>& strides();
		const std::vector<size_t>& strides() const;
		size_t& offset();
		const size_t& offset() const;
		std::string& name();
		const std::string& name() const;
		size_t ndim();
		const size_t ndim() const;
		int& id();
		const int& id() const;
		Device& device();
		const Device& device() const;

		bool& requires_grad();		

		size_t numel() const;

		inline size_t dtype_size(DType dtype);
		size_t nbytes();

		std::vector<size_t> calculate_strides(std::vector<size_t> shape);

		void set_grad(Tensor& g);

		void set_is_view(bool flag);
		void set_requires_grad(bool flag);		
		bool is_view();
		//void set_base(const std::shared_ptr<TensorImpl>& base);
		//std::shared_ptr<TensorImpl> get_base();

		bool is_contiguous() const;

	private:

		Device m_device;
		std::shared_ptr<Storage> m_data;
		std::shared_ptr<Tensor>  m_grad;
		std::shared_ptr<Node> m_grad_fn;
		std::shared_ptr<Node> m_grad_accum;

		std::vector<size_t> m_shape;
		std::vector<size_t> m_strides;
		size_t m_offset;
		std::string m_name;
		size_t m_datacount;
		size_t m_gradcount;
		DType m_dtype;
		bool m_requires_grad;
		bool m_isview;
		//std::shared_ptr<TensorImpl>  m_base;
		int m_id;

	};


}