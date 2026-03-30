#pragma once
#include <memory>
#include <vector>
#include <cstddef>

namespace Inferno {

	class Storage {

	public:

		Storage();
		~Storage();

		template<typename T>
		T* storage_as_ptr() {
			void* ptr = raw_ptr();
			return static_cast<T*>(ptr);
		}

		template<typename T>
		T* storage_as_ptr() const {
			void* ptr = raw_ptr();
			return static_cast<T*>(ptr);
		}


		virtual void* raw_ptr() = 0;
		size_t m_numbytes;
		



	};


}