#pragma once


namespace Inferno {

	enum class DeviceType {
		CPU,
		CUDA
	};


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function name
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	class Device {

	public:

		DeviceType m_type;
		int index;

		// ---- Constructors ----
		Device(DeviceType t = DeviceType::CPU, int idx = 0)
			: m_type(t), index(idx) {}

		bool operator==(const Device& other) const {
			return m_type == other.m_type && index == other.index;
		}

		bool operator!=(const Device& other) const {
			return !(*this == other);
		}

		static Device cpu() { return Device(DeviceType::CPU, 0); }
		static Device cuda(int index) { return Device(DeviceType::CUDA, index); }

		bool is_cuda() const { return (m_type == DeviceType::CUDA) ? true : false; }
		bool is_cpu() const { return (m_type == DeviceType::CPU) ? true : false; }
		


	};


}