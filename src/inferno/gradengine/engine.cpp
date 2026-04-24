#include "engine.h"
#include <unordered_set>

namespace Inferno {

	bool grad_enabled = true;

	thread_local std::unordered_map<Inferno::Edge, Inferno::Tensor, Inferno::EdgeHash>* Inferno::Engine::s_grad_map = nullptr;


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function backward
	//
	//  Entry point for running backpropagation from a final output tensor.
	//
	//  Steps performed:
	//      1. Find the root gradient node attached to the output tensor
	//      2. Build a topological ordering of all reachable nodes
	//      3. Create a temporary gradient map for this backward pass
	//      4. Seed the output tensor gradient with ones_like(output)
	//      5. Traverse the graph in reverse topological order
	//      6. Call each node's backward() function
	//      7. Release graph references and clear temporary gradient storage
	//
	//  This function is the core driver of the autograd engine.
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Engine::backward(const Tensor& tensor) {


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

		if (topo.empty())
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "*********** topo  was empty ***********");
		else
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Built topo with size: " + std::to_string(topo.size()));

		Tensor seed = Tensor::ones_like(tensor);
			accumulate(root.get(), 0, seed);


		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "*********** starting backward iterations ***********");

		//loop through all of the nodes and call the backward function
		//these should all be in progressive order from back to front now
		int count = 0;
		for (auto it = topo.rbegin(); it != topo.rend(); it++) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Engine backward step: " + (*it)->name());
			(*it)->backward();  // Perform the gradient accumulation for the current node							
			count++;
		}


		for (auto it = topo.begin(); it != topo.end(); it++) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Engine releasing: " + (*it)->name());
			(*it)->release();
		}

		topo.clear();		

		s_grad_map->clear();
		s_grad_map = nullptr;


	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function build_topo
	//
	//  Recursively builds a topological ordering of the computation graph.
	//
	//  Behavior:
	//      - Visits each node only once using the visited set
	//      - Recursively visits parent nodes first
	//      - Pushes the current node after its dependencies
	//
	//  Result:
	//      The topo vector contains nodes in dependency order, which allows
	//      backward() to process them in reverse for correct gradient flow.
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Engine::build_topo(std::shared_ptr<Node> node, std::unordered_set<Node*>& visited, std::vector<Node*>& topo) {

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


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function accumulate
	//
	//  Adds a gradient contribution into the engine's temporary gradient map.
	//
	//  Behavior:
	//      - Identifies a target edge by (node, slot)
	//      - If no gradient exists yet for that edge, inserts it
	//      - If one already exists, adds the new gradient to the existing one
	//
	//  This is used whenever multiple downstream paths contribute gradients to
	//  the same node input.
	//
	//  NoGradGuard is used so the gradient accumulation math itself does not
	//  create new autograd graph nodes.
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Engine::accumulate(Node* node, int slot, Tensor& grad) {

		NoGradGuard guard;
		if (!node || !s_grad_map) return;

		Edge e{ node, slot };

		auto it = s_grad_map->find(e);

		if (it == s_grad_map->end())
			s_grad_map->emplace(e, grad);
		else
			it->second = it->second + grad;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function grad_in
	//
	//  Retrieves the currently accumulated upstream gradient for a given node
	//  and output/input slot.
	//
	//  Returns:
	//      - The gradient tensor if one has been accumulated for that edge
	//      - An empty/default Tensor if no gradient is present
	//
	//  This is what Node::backward() implementations use to read their incoming
	//  gradient from the engine.
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Tensor Engine::grad_in(Node* node, int slot) {

		if (!s_grad_map) return Tensor{};

		Edge e{ node, slot };
		auto it = s_grad_map->find(e);

		if (it == s_grad_map->end())
			return Tensor{};

		return it->second;
	}


}