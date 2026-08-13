#include "flashattentionbackward.h"
#include "inferno/gradengine/engine.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/core/dtype_dispatch.h"

namespace Inferno {

	FlashAttentionBackward::FlashAttentionBackward(
		const Tensor& Q,
		const Tensor& K,
		const Tensor& V,
		const Tensor& O,
		bool causal
	) :
		m_Q(Q),
		m_K(K),
		m_V(V),
		m_O(O),
		m_causal(causal)
	{
		set_name("FlashAttentionBackward");
	}

	void FlashAttentionBackward::backward() {
		NoGradGuard guard;

		Tensor g_out = Engine::grad_in(this, 0);

		Tensor dQ(m_Q.dtype(), m_Q.shape(), "flash_attention_dQ", m_Q.device());

		Tensor dK(m_K.dtype(), m_K.shape(), "flash_attention_dK", m_K.device());

		Tensor dV(m_V.dtype(), m_V.shape(), "flash_attention_dV", m_V.device());

		std::vector<size_t> qshape = m_Q.shape();

		const size_t B = qshape[0];
		const size_t H = qshape[1];
		const size_t Tseq = qshape[2];
		const size_t D = qshape[3];

		dispatchFloat(m_Q.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* qptr = GetImpl(m_Q)->data_as_ptr<AT>();
			const AT* kptr = GetImpl(m_K)->data_as_ptr<AT>();
			const AT* vptr = GetImpl(m_V)->data_as_ptr<AT>();
			const AT* optr = GetImpl(m_O)->data_as_ptr<AT>();
			const AT* goptr = GetImpl(g_out)->data_as_ptr<AT>();

			AT* dqptr = GetImpl(dQ)->data_as_ptr<AT>();
			AT* dkptr = GetImpl(dK)->data_as_ptr<AT>();
			AT* dvptr = GetImpl(dV)->data_as_ptr<AT>();

			switch (m_Q.device().m_type) {
			case DeviceType::CUDA:
				cuda_flash_attention_simple_backward<AT>(
					qptr,
					kptr,
					vptr,
					optr,
					goptr,
					dqptr,
					dkptr,
					dvptr,
					B,
					H,
					Tseq,
					D,
					m_causal
				);
				break;

			default:
				INFERNO_LOG_ERROR() << "flash_attention_simple_backward only supports CUDA" << std::endl;
				exit(1);
			}
			});

		auto nq = GetImpl(m_Q)->grad_edge();
		auto nk = GetImpl(m_K)->grad_edge();
		auto nv = GetImpl(m_V)->grad_edge();

		if (nq) {
			Engine::accumulate(nq.get(), 0, dQ);
		}

		if (nk) {
			Engine::accumulate(nk.get(), 0, dK);
		}

		if (nv) {
			Engine::accumulate(nv.get(), 0, dV);
		}
	}

	void FlashAttentionBackward::release() {
		m_Q = Tensor{};
		m_K = Tensor{};
		m_V = Tensor{};
		m_O = Tensor{};
	}

	void FlashAttentionBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_Q);
		out.push_back(m_K);
		out.push_back(m_V);
	}




	FlashAttentionBigDaddyBackward::FlashAttentionBigDaddyBackward(const Tensor& qkv, const Tensor& out, size_t num_heads, bool causal)
		: m_qkv(qkv),
		m_out(out),
		m_num_heads(num_heads),
		m_causal(causal)
	{
		set_name("FlashAttentionBigDaddyBackward");
	}


	void FlashAttentionBigDaddyBackward::backward() {
		NoGradGuard guard;

		Tensor g_out = Engine::grad_in(this, 0);

		std::vector<size_t> qkv_shape = m_qkv.shape();

		size_t B = qkv_shape[0];
		size_t Tseq = qkv_shape[1];
		size_t threeC = qkv_shape[2];
		size_t C = threeC / 3;
		size_t H = m_num_heads;
		size_t D = C / H;

		Tensor g_qkv(m_qkv.dtype(), qkv_shape, "flash_attention_bigdaddy_grad_qkv", m_qkv.device());

		dispatchFloat(m_qkv.dtype(), [&](auto TagA) {
			using AT = typename decltype(TagA)::type;

			const AT* qkvptr = GetImpl(m_qkv)->data_as_ptr<AT>();
			const AT* outptr = GetImpl(m_out)->data_as_ptr<AT>();
			const AT* goutptr = GetImpl(g_out)->data_as_ptr<AT>();
			AT* gqkvptr = GetImpl(g_qkv)->data_as_ptr<AT>();

			switch (m_qkv.device().m_type) {
			case DeviceType::CUDA:
				cuda_flash_attention_bigdaddy_backward_tiled<AT>(
					qkvptr,
					outptr,
					goutptr,
					gqkvptr,
					B,
					Tseq,
					C,
					H,
					D,
					m_causal
				);
				break;

			default:
				INFERNO_LOG_ERROR() << "FlashAttentionBigDaddyBackward only supports CUDA for now" << std::endl;
				exit(1);
			}
			});

		auto nqkv = GetImpl(m_qkv)->grad_edge();

		if (nqkv) {
			Engine::accumulate(nqkv.get(), 0, g_qkv);
		}
	}

	void FlashAttentionBigDaddyBackward::release() {
		m_qkv = Tensor{};
		m_out = Tensor{};
	}

	void FlashAttentionBigDaddyBackward::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_qkv);
	}




	FlashAttentionBigDaddyBackwardFast::FlashAttentionBigDaddyBackwardFast(const Tensor& qkv, const Tensor& out, const Tensor& l, size_t num_heads, bool causal)
		: m_qkv(qkv),
		m_o(out),
		m_l(l),
		m_num_heads(num_heads),
		m_causal(causal)
	{
		set_name("FlashAttentionBigDaddyBackwardFast");
	}



	void FlashAttentionBigDaddyBackwardFast::backward() {
		NoGradGuard guard;

		Tensor g_out = Engine::grad_in(this, 0);

		std::vector<size_t> qkv_shape = m_qkv.shape();

		size_t B = qkv_shape[0];
		size_t Tseq = qkv_shape[1];

		// If your qkv is still [B, T, 3C]
		size_t threeC = qkv_shape[2];
		size_t C = threeC / 3;
		size_t H = m_num_heads;
		size_t D = C / H;

		Tensor g_qkv(m_qkv.dtype(), qkv_shape, "flash_attention_bigdaddy_grad_qkv", m_qkv.device());

		if (m_qkv.dtype() != DType::Float32) {
			INFERNO_LOG_ERROR()	<< "cuda_flash_backward_fused currently only supports Float32" << std::endl;
			std::exit(1);
		}

		const float* qkvptr = GetImpl(m_qkv)->data_as_ptr<float>();
		const float* optr = GetImpl(m_o)->data_as_ptr<float>();
		const float* lptr = GetImpl(m_l)->data_as_ptr<float>();
		const float* goutptr = GetImpl(g_out)->data_as_ptr<float>();
		float* gqkvptr = GetImpl(g_qkv)->data_as_ptr<float>();

		switch (m_qkv.device().m_type) {
		case DeviceType::CUDA:
			cuda_flash_backward_fused(
				qkvptr,
				optr,
				lptr,
				goutptr,
				gqkvptr,
				B,
				Tseq,
				C,
				H,
				D,
				m_causal
			);
			break;

		default:
			INFERNO_LOG_ERROR()	<< "FlashAttentionBigDaddyBackward only supports CUDA for now" << std::endl;
			std::exit(1);
		}


		
		auto nqkv = GetImpl(m_qkv)->grad_edge();

		if (nqkv) {
			Engine::accumulate(nqkv.get(), 0, g_qkv);
		}
	}

	void FlashAttentionBigDaddyBackwardFast::release() {
		m_qkv = Tensor{};
		m_o = Tensor{};
		m_l = Tensor{};
	}

	void FlashAttentionBigDaddyBackwardFast::get_inputs(std::vector<Tensor>& out) const {
		out.push_back(m_qkv);
		out.push_back(m_o);
		out.push_back(m_l);
	}


}