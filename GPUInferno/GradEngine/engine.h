#pragma once
#include <unordered_map>
#include "../GradFN/node.h"
#include "../tensor.h"
#include "../ops.h"




namespace Inferno {

    extern bool grad_enabled;

    class NoGradGuard {
    public:
        NoGradGuard() {
            m_prev = Inferno::grad_enabled;
            Inferno::grad_enabled = false;
        }

        ~NoGradGuard() {
            Inferno::grad_enabled = m_prev;
        }

    private:
        bool m_prev;
    };

    struct Edge {
        Node* node;
        int slot;

        bool operator==(const Edge& other) const {
            return node == other.node && slot == other.slot;
        }
    };

    struct EdgeHash {
        std::size_t operator()(const Edge& e) const {
            return std::hash<Node*>()(e.node) ^ (std::hash<int>()(e.slot) << 1);
        }
    };


	class Engine {


	public:

		static void backward(const Tensor tensor);
		static void accumulate(Node* node, int slot, Tensor& grad);
        static Tensor grad_in(Node* node, int slot);
		static void build_topo(std::shared_ptr<Node> node, std::unordered_set<Node*>& visited, std::vector<Node*>& topo);
	
	private:    
        static thread_local std::unordered_map<Edge, Tensor, EdgeHash>* s_grad_map;


	};

}
