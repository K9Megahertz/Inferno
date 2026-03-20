#pragma once

#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include "device.h"
#include "dtype.h"
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
		TensorImpl(DType type, std::vector<size_t> shape, std::string name, Inferno::Device device);

		template <typename T>
		TensorImpl(DType type, const std::vector<T>& data, std::vector<size_t> shape, std::string name, Inferno::Device device) {
			m_device = device;
			m_dtype = type;
			m_shape = shape;
			m_name = name;
			m_strides = calculate_strides(shape);
			m_grad = nullptr;
			m_requires_grad = true;


			//TODO: validate T == dtype
			size_t bytes = data.size() * sizeof(T);

			if (device.m_type == DeviceType::CPU) {
				m_data = std::make_shared<CPUStorage>(bytes);
				std::memcpy(m_data->raw_ptr(), data.data(), bytes);
			}
			else if (device.m_type == DeviceType::CUDA) {
				m_data = std::make_shared<CUDAStorage>(bytes);
				cudaError_t err = cudaMemcpy(m_data->raw_ptr(), data.data(), bytes, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) {
					Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Could not cudaMemcpy in TensorImpl constructor.");
					exit(1);
				}
			}
			else {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Attempt to create TensorImpl with unknown device type.");
				exit(1);
			}

			m_id = Inferno::IDBroker::GenID();
			Inferno::NodeTracker::addID(this->m_id, this->m_name);

		}

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
		int id();
		const int id() const;
		Device& device();
		const Device& device() const;

		size_t numel() const;

		inline size_t dtype_size(DType dtype);
		size_t nbytes();

		std::vector<size_t> calculate_strides(std::vector<size_t> shape);

		void set_grad(Tensor& g);

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
		int m_id;

	};


}