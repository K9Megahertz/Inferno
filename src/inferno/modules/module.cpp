#include <inferno/modules/module.h>
#include <inferno/core/tensorimpl.h>
#include <logger/logger.h>

namespace Inferno {


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

	Tensor Module::forward(Tensor& input) {
		INFERNO_LOG_ERROR() << "forward(input) not implemented!" << std::endl;
		exit(1);
	};


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function parameters
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//std::vector<Tensor*> Module::parameters() const {
	std::vector<std::pair<std::string, Tensor*>> Module::parameters() const {
		std::vector<std::pair<std::string, Tensor*>> all_params;
		collect_parameters(all_params, "");
		return all_params;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function register_parameter
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Module::register_parameter(const std::string& name, Tensor* tensor) {
		// Check for name collisions
		if (check_name_exists(name)) {
			INFERNO_LOG_ERROR() << "Module/Parameter/Buffer name conflict, already used" << std::endl;
			exit(1);
		}

		// Store pointer to the tensor
		_parameters.push_back({name, tensor});
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function register_module
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Module::register_module(const std::string& name, Module* module) {
		// Check for name collisions
		if (check_name_exists(name)) {
			INFERNO_LOG_ERROR() << "Module/Parameter/Buffer name conflict, already used" << std::endl;
			exit(1);
		}

		// Store pointer to the tensor		
		_children.push_back({ name, module });
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function register_buffer
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Module::register_buffer(const std::string& name, Tensor* tensor) {
		// Check for name collisions
		if (check_name_exists(name)) {
			INFERNO_LOG_ERROR() << "Module/Parameter/Buffer name conflict, already used" << std::endl;
			exit(1);
		}
		// Store pointer to the tensor		
		_buffers.push_back({ name, tensor });
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function to
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Module::to(const Device& device) {
		for (auto& [name, param] : _parameters) {
			if (param) {
				*param = param->to(device);
			}
		}

		for (auto& [name, buf] : _buffers) {
			if (buf) {
				*buf = buf->to(device);
			}
		}

		for (auto& [name, child] : _children) {
			if (child) {
				child->to(device);
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function << overload
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::ostream& operator<<(std::ostream& os, const Module& module) {

		for (auto& [name, param] : module.parameters()) {
			os << *param << std::endl;
		}

		return os;
	}

	StateDict Module::state_dict() const {
		StateDict out;
		collect_state_dict(out, "");
		return out;
	}

	void Module::collect_state_dict(StateDict& out, const std::string& prefix) const {
		// print this module's parameters
		for (const auto& [name, tensor] : _parameters) {
			std::string full_name = prefix.empty() ? name : prefix + "." + name;
			out[full_name] = *tensor;

		}

		for (const auto& [name, buffer] : _buffers) {
			std::string full_name = prefix.empty() ? name : prefix + "." + name;
			out[full_name] = *buffer;

		}

		// recurse into children
		for (const auto& [child_name, child] : _children) {
			std::string child_prefix = prefix.empty() ? child_name : prefix + "." + child_name;
			child->collect_state_dict(out, child_prefix);
		}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function check_name_exists
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	bool Module::check_name_exists(std::string name) {

		for (const auto& [existing_name, _] : _parameters) {
			if (existing_name == name) {
				return true;
			}
		}

		for (const auto& [existing_name, _] : _buffers) {
			if (existing_name == name) {
				return true;
			}
		}

		for (const auto& [existing_name, _] : _children) {
			if (existing_name == name) {
				return true;
			}
		}

		return false;
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function collect_parameters
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Module::collect_parameters(std::vector<std::pair<std::string, Tensor*>>& out, const std::string& prefix) const {

		for (const auto& [name, tensor_ptr] : _parameters) {
			std::string full_name = prefix.empty() ? name : prefix + "." + name;
			out.emplace_back(full_name, tensor_ptr);
		}

		for (const auto& [child_name, child] : _children) {
			std::string child_prefix = prefix.empty() ? child_name : prefix + "." + child_name;
			child->collect_parameters(out, child_prefix);
		}
	}

	void Module::load_state_dict(const StateDict& state) {
		load_state_dict_recursive(state, "");

		// Optional: warn about keys in checkpoint that were not used
		StateDict current = state_dict();

		for (const std::pair<const std::string, Tensor>& entry : state) {
			const std::string& name = entry.first;

			if (current.find(name) == current.end()) {
				std::cout << "[WARNING] Unexpected key in state_dict: "
					<< name << std::endl;
			}
		}
	}

	void Module::load_state_dict_recursive(const StateDict& state, const std::string& prefix) {


		// ------------------------------------------------------------
	    // Load parameters
	    // ------------------------------------------------------------
		for (const std::pair<std::string, Tensor*>& entry : _parameters) {
			const std::string& local_name = entry.first;
			Tensor* param = entry.second;

			std::string full_name =	prefix.empty() ? local_name : prefix + "." + local_name;

			auto it = state.find(full_name);

			if (it == state.end()) {				
				INFERNO_LOG_ERROR() << "load_state_dict: missing parameter: " << full_name << std::endl;
				exit(1);
			}

			const Tensor& loaded = it->second;

			if (param == nullptr) {				
				INFERNO_LOG_ERROR() << "load_state_dict: null parameter pointer: " << full_name << std::endl;
				exit(1);
			}

			if (param->shape() != loaded.shape()) {				
				INFERNO_LOG_ERROR() << "load_state_dict: shape mismatch for: " << full_name << std::endl;
				exit(1);
			}

			if (param->dtype() != loaded.dtype()) {				
				INFERNO_LOG_ERROR() << "load_state_dict: dtype mismatch for: " << full_name << std::endl;
				exit(1);
			}
			
			param->copy_(loaded);
		}


		// ------------------------------------------------------------
		// Load buffers
		// ------------------------------------------------------------
		for (const std::pair<std::string, Tensor*>& entry : _buffers) {
			const std::string& local_name = entry.first;
			Tensor* buffer = entry.second;

			std::string full_name =
				prefix.empty() ? local_name : prefix + "." + local_name;

			auto it = state.find(full_name);

			if (it == state.end()) {				
				INFERNO_LOG_ERROR() << "load_state_dict: missing buffer: " << full_name << std::endl;
				exit(1);
			}

			const Tensor& loaded = it->second;

			if (buffer == nullptr) {				
				INFERNO_LOG_ERROR() << "load_state_dict: null buffer pointer: " << full_name << std::endl;
				exit(1);
			}

			if (buffer->shape() != loaded.shape()) {				
				INFERNO_LOG_ERROR() << "load_state_dict: buffer shape mismatch for: " << full_name << std::endl;
				exit(1);
			}

			if (buffer->dtype() != loaded.dtype()) {				
				INFERNO_LOG_ERROR() << "load_state_dict: buffer dtype mismatch for: " << full_name << std::endl;
				exit(1);
			}

			buffer->copy_(loaded);
		}

		// ------------------------------------------------------------
		// Recursivly load children
		// ------------------------------------------------------------
		for (const std::pair<std::string, Module*>& entry : _children) {
			const std::string& child_name = entry.first;
			Module* child = entry.second;

			if (child == nullptr) {				
				INFERNO_LOG_ERROR() << "load_state_dict: null child module: " << child_name << std::endl;
				exit(1);
			}

			std::string child_prefix = prefix.empty() ? child_name : prefix + "." + child_name;

			child->load_state_dict_recursive(state, child_prefix);
		}
	}

}