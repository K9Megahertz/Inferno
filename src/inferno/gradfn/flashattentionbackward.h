#pragma once
#include "node.h"
#include <inferno/core/tensor.h>
#include "inferno/core/broadcastops.h"

namespace Inferno {

	class FlashAttentionBackward : public Node {
	public:
		FlashAttentionBackward(
			const Tensor& Q,
			const Tensor& K,
			const Tensor& V,
			const Tensor& O,
			bool causal
		);

		void backward() override;
		void release() override;
		void get_inputs(std::vector<Tensor>& out) const override;

	private:
		Tensor m_Q;
		Tensor m_K;
		Tensor m_V;
		Tensor m_O;
		bool m_causal;
	};



    class FlashAttentionBigDaddyBackward : public Node {
    public:
		FlashAttentionBigDaddyBackward(const Tensor& qkv, const Tensor& out, size_t num_heads, bool causal);

        void backward() override;
        void release() override;
        void get_inputs(std::vector<Tensor>& out) const;

    private:
        Tensor m_qkv;
        Tensor m_out;
        size_t m_num_heads;
        bool m_causal;
    };


	class FlashAttentionBigDaddyBackwardFast : public Node {
	public:
		FlashAttentionBigDaddyBackwardFast(const Tensor& qkv, const Tensor& out, const Tensor& l, size_t num_heads, bool causal);

		void backward() override;
		void release() override;
		void get_inputs(std::vector<Tensor>& out) const;

	private:
		Tensor m_qkv;
		Tensor m_o;
		Tensor m_l;
		size_t m_num_heads;
		bool m_causal;
	};

}