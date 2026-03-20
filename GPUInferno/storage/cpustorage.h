#pragma once
#include "storage.h"


namespace Inferno {

	class CPUStorage final : public Storage {

	public:

		CPUStorage(size_t numbytes);
		~CPUStorage();

		void* raw_ptr() { return m_data.data(); }

	private:

		std::vector<std::byte> m_data;


	};


}