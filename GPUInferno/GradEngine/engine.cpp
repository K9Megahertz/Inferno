#include "engine.h"
#include <unordered_set>

namespace Inferno {

	bool grad_enabled = true;

	thread_local std::unordered_map<Inferno::Edge, Inferno::Tensor, Inferno::EdgeHash>* Inferno::Engine::s_grad_map = nullptr;

	void Engine::backward(const Tensor tensor) {

		
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "*********** running backward ***********");

		std::unordered_set<Node*> visited;
		std::vector<Node*> topo;

		
		std::shared_ptr<Node> root = GetImpl(tensor)->grad_edge();
		if (!root) {						
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "*********** didnt find a root ***********");
			exit(1);
		}

		std::unordered_map<Edge, Tensor, EdgeHash> grad_map;
		s_grad_map = &grad_map;
		
		
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "*********** building topo ***********");
		build_topo(root, visited, topo);
		
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Built topo with size: " + topo.size());

		Tensor seed = Tensor::ones_like(tensor);
		accumulate(root.get(), 0, seed);
		

		//loop through all of the nodes and call the backward function
		//these should all be in progressive order from back to front now
		int count = 0;
		for (auto it = topo.rbegin(); it != topo.rend(); it++) {			
			(*it)->backward();  // Perform the gradient accumulation for the current node							
			count++;		
		}
		

		for (Node* n : topo) {
			n->release();
		}
		topo.clear();		
		s_grad_map->clear();


	}

	void Engine::build_topo( std::shared_ptr<Node> node, std::unordered_set<Node*>& visited, std::vector<Node*>& topo) {

		std::vector<Tensor> inputlist;

		//have we visited this node before?
		if (!visited.contains(node.get())) {

			// no, so insert it into the list
			visited.insert(node.get());

			//get a list of it inputs
			node->get_inputs(inputlist);

			//loop over each input
			for (auto& input : inputlist) {
				

				auto parent = GetImpl(input)->grad_edge();
				if (parent) //does it have a parent
					build_topo(parent, visited, topo);
			}
			topo.push_back(node.get());
		}

	}

	void Engine::accumulate(Node* node,int slot, Tensor& grad)
	{
		NoGradGuard guard;
		if (!node || !s_grad_map) return;

		Edge e{ node, slot };

		auto it = s_grad_map->find(e);

		if (it == s_grad_map->end())
			s_grad_map->emplace(e, grad);			
		else
			it->second = it->second + grad;			
	}

	Tensor Engine::grad_in(Node* node, int slot)
	{
		if (!s_grad_map) return Tensor{};

		Edge e{ node, slot };
		auto it = s_grad_map->find(e);

		if (it == s_grad_map->end())
			return Tensor{};

		return it->second;
	}


}