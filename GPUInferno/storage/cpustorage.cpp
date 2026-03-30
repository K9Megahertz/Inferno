#include "cpustorage.h"

namespace Inferno {

	CPUStorage::CPUStorage(size_t numbytes) {
		//m_data = std::make_shared<std::vector<int>>(data.begin(), data.end());
		//m_data = std::make_shared<std::vector<std::byte>>(numbytes,0);
		m_data.resize(numbytes, std::byte{0});
		m_numbytes = numbytes;

	}


	CPUStorage::~CPUStorage() {	

	}

}