#pragma once
#include "../tensor.h"
#include "../ops.h"

namespace Inferno {


	class Module {


	public:

		virtual ~Module() = default;

		virtual Tensor forward(Tensor& input) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "forward(input) not implemented!");
			exit(1);
		};


		std::vector<Tensor*> parameters() {
			std::vector<Tensor*> all_params = _parameters;
			for (auto child : _children) {
				auto child_params = child->parameters();
				if (!child_params.empty())
					all_params.insert(all_params.end(), child_params.begin(), child_params.end());
			}
			return all_params;
		}

		void register_parameter(Tensor& tensor) {
			_parameters.push_back(&tensor);
		}



	private:

		std::vector<Module*> _children;
		std::vector<Tensor*> _parameters;


	};


}