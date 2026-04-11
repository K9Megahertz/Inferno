#pragma once
#include "node.h"
#include "inferno/core/tensor.h"
#include "core/broadcastops.h"




namespace Inferno {

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class CrossEntropyLossBackward 
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class CrossEntropyLossBackward : public Node {

	public:

		CrossEntropyLossBackward(const Tensor& logits, const Tensor& target);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_logits;
		Tensor m_target;


	};



    template<typename LT>
    static void cpu_cross_entropy_loss_backward(
        const LT* logits,
        const int* targets,
        const LT* upstream,
        LT* grad_logits,
        size_t rows,
        size_t vocab_size
    ) {
        const LT scale = upstream[0] / static_cast<LT>(rows);

        for (size_t r = 0; r < rows; r++) {
            const LT* row_ptr = logits + (r * vocab_size);
            LT* grad_row = grad_logits + (r * vocab_size);
            int target_id = targets[r];

            if (target_id < 0 || static_cast<size_t>(target_id) >= vocab_size) {
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR,
                    "Target index out of bounds in cpu_cross_entropy_loss_backward");
                exit(1);
            }

            LT max_logit = row_ptr[0];
            for (size_t v = 1; v < vocab_size; v++) {
                if (row_ptr[v] > max_logit) {
                    max_logit = row_ptr[v];
                }
            }

            LT sum_exp = static_cast<LT>(0);
            for (size_t v = 0; v < vocab_size; v++) {
                sum_exp += std::exp(row_ptr[v] - max_logit);
            }

            for (size_t v = 0; v < vocab_size; v++) {
                LT prob = std::exp(row_ptr[v] - max_logit) / sum_exp;
                grad_row[v] = prob * scale;
            }

            grad_row[target_id] -= scale;
        }
    }

}