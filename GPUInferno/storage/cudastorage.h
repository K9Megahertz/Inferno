#pragma once
#include <cuda_runtime.h>
#include "storage.h"



namespace Inferno {

	class CUDAStorage final : public Storage {

	public:

		CUDAStorage(size_t bytes);
		~CUDAStorage();

		void* raw_ptr() { return ptr; }

	private:

		void* ptr = nullptr;


	};


}