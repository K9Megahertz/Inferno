#include "accumulategrad.h"

namespace Inferno {


	//AccumulateGrad::AccumulateGrad(const Tensor& A) {


	//}

	void AccumulateGrad::backward() {
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "AccumulateGrad::backward()");
        // gradient that arrived at this leaf accumulator
        Tensor g_in = Engine::grad_in(this, 0);

        // get the leaf tensor impl
        auto leaf = m_leaf.lock();
        if (!leaf)
            return;
      
        leaf->set_grad(g_in);

	}

	void AccumulateGrad::release() {
		
		
	}



}