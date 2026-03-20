#pragma once
#include <memory>
#include <vector>
#include <unordered_set>
#include "../tensor.h"




namespace Inferno {


	class Node {


	public:

		Node() = default;
		virtual ~Node() = default;
		
		virtual void backward() = 0;
		virtual void release() = 0;
		virtual void get_inputs(std::vector<Tensor>& out) const = 0;

	private:
		
		

		

	};
		


}
