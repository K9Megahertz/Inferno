#include "linear.h"
 
namespace Inferno {




	Linear::Linear(size_t in_features, size_t out_features, Device device, DType dtype) {


		float lowrange = -1.0f;
		float highrange = 1.0f;

		size_t count = in_features * out_features;

		//Init weights
		dispatchOne(dtype, [&](auto T) {
			using AT = typename decltype(T)::type;

			std::vector<AT> weight_data;

			if constexpr (std::is_same_v<AT, float>) {
				weight_data = Inferno::RandomGenerator::generateRandomFloatVector(count, lowrange,highrange);
			}
			else if constexpr (std::is_same_v<AT, double>) {
				weight_data = Inferno::RandomGenerator::generateRandomDoubleVector(count, lowrange, highrange);
			}
			else if constexpr (std::is_same_v<AT, int>) {
				weight_data = Inferno::RandomGenerator::generateRandomIntVector(count, -1, 1);
			}
			else {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Unsupported dtype in Linear");
				std::exit(1);
			}

			m_weights = Tensor(dtype, weight_data, { in_features, out_features }, "weights", device);
			//std::cout << m_weights.to(Inferno::Device::cpu()) << std::endl;
			
		
		

		

			//Init biases
			std::vector<AT> bias_data;

			if constexpr (std::is_same_v<AT, float>) {
				bias_data = Inferno::RandomGenerator::generateRandomFloatVector(out_features, lowrange, highrange);
			}
			else if constexpr (std::is_same_v<AT, double>) {
				bias_data = Inferno::RandomGenerator::generateRandomDoubleVector(out_features, lowrange, highrange);
			}
			else if constexpr (std::is_same_v<AT, int>) {
				bias_data = Inferno::RandomGenerator::generateRandomIntVector(out_features, -1, 1);
			}
			else {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Unsupported dtype in Linear");
				std::exit(1);
			}

			m_biases = Tensor(dtype, bias_data, { out_features }, "biases",device);
			//std::cout << m_biases.to(Inferno::Device::cpu()) << std::endl;
			

		});


		m_in_features = in_features;
		m_out_features = out_features;

		// Register parameters
		register_parameter(m_weights);
		register_parameter(m_biases);




	}


	Tensor Linear::forward(const Tensor& input) {

	
		Tensor matmul = Inferno::matmul(input, m_weights);
		Tensor ret = matmul + m_biases;
		
		return ret;
	}



}