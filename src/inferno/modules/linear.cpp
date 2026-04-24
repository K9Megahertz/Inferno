#include <inferno/modules/linear.h>
#include "inferno/util/logger.h"
#include "inferno/core/dtype_dispatch.h"

namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  CTORS / DTORS
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Linear::Linear(size_t in_features, size_t out_features, Device device, DType dtype) {

		float limit = sqrt(6.0f / static_cast<float>(in_features + out_features));


		float lowrange = -limit;
		float highrange = limit;

		

		size_t count = in_features * out_features;

		//Init weights
		dispatchAny(dtype, [&](auto T) {
			using AT = typename decltype(T)::type;
			using RT = promote_t<AT, float>;


			std::vector<RT> weight_data;

			if constexpr (std::is_same_v<RT, float>) {
				weight_data = Inferno::RandomGenerator::generateRandomFloatVector(count, lowrange,highrange);
			}
			else if constexpr (std::is_same_v<RT, double>) {
				weight_data = Inferno::RandomGenerator::generateRandomDoubleVector(count, lowrange, highrange);
			}			
			else {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Unsupported dtype in Linear");
				std::exit(1);
			}

			m_weights = Tensor(dtype, weight_data, { in_features, out_features }, "weights", device, true);
			
			
		

			//Init biases
			std::vector<RT> bias_data;

			if constexpr (std::is_same_v<RT, float>) {
				bias_data = Inferno::RandomGenerator::generateRandomFloatVector(out_features, lowrange, highrange);
			}
			else if constexpr (std::is_same_v<RT, double>) {
				bias_data = Inferno::RandomGenerator::generateRandomDoubleVector(out_features, lowrange, highrange);
			}			
			else {
				Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Unsupported dtype in Linear");
				std::exit(1);
			}

			m_biases = Tensor(dtype, bias_data, { out_features }, "biases",device, true);
			

		});


		m_in_features = in_features;
		m_out_features = out_features;

		// Register parameters
		register_parameter("weights",&m_weights);
		register_parameter("bias", &m_biases);




	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function forward
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor Linear::forward(const Tensor& input) {

	
		Tensor matmul = Inferno::matmul(input, m_weights, "Linear");
		Tensor ret = matmul + m_biases;
		
		return ret;
	}
	
}