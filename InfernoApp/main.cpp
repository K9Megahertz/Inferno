#include <unordered_map>
#include <inferno/inferno.h>
#include <infernotokenizer/bpetokenizer.h>
#include "logger.h"
#include "timer.h"
#include "dataloader.h"

Inferno::Timer t1("matmul");

extern int g_mmcountfast;
extern int g_mmcountslow;
extern std::unordered_map<std::string, size_t> g_matmul_counts;
//Inferno::Device device = Inferno::Device::cpu();
Inferno::Device device = Inferno::Device::cuda(0);





class PositionalEncoding : public Inferno::Module {


public:

	PositionalEncoding(size_t context_size, size_t embed_dim) {

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding - Initializing buffers" << std::endl;;
		//initialize positional vectors
		std::vector<float> pe_data(context_size * embed_dim);


		for (size_t pos = 0; pos < context_size; ++pos) {
			for (size_t i = 0; i < embed_dim; ++i) {
				float exponent = 2.0f * float(i / 2) / float(embed_dim); // 2i/d_model
				float angle = float(pos) / std::pow(10000.0f, exponent);

				if (i % 2 == 0) {
					pe_data[pos * embed_dim + i] = std::sin(angle);
				}
				else {
					pe_data[pos * embed_dim + i] = std::cos(angle);
				}
			}
		}

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding - Creating tensor" << std::endl;
		pe = Inferno::Tensor(Inferno::DType::Float32, std::move(pe_data), { context_size, embed_dim }, "positional-encoding");

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding - Register Buffer" << std::endl;
		register_buffer("pe",&pe);

	}


	Inferno::Tensor forward(Inferno::Tensor& x) {
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding forward" << std::endl;
		return x + pe;
	}

	Inferno::Tensor pe;

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Class MultiHeadAttention
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*class MultiHeadAttention : public Inferno::Module {
public:
	MultiHeadAttention(size_t embed_dim, size_t num_heads) :
		m_embed_dim(embed_dim),
		m_num_heads(num_heads),
		m_head_dim(embed_dim / num_heads),
		W_out(embed_dim, embed_dim)
	{

		Wq_layers.reserve(m_num_heads);
		Wk_layers.reserve(m_num_heads);
		Wv_layers.reserve(m_num_heads);

		for (size_t i = 0; i < m_num_heads; ++i) {
			Wq_layers.emplace_back(m_embed_dim, m_head_dim);
			Wk_layers.emplace_back(m_embed_dim, m_head_dim);
			Wv_layers.emplace_back(m_embed_dim, m_head_dim);

			register_module(&Wq_layers.back());
			register_module(&Wk_layers.back());
			register_module(&Wv_layers.back());
		}
		
		register_module(&W_out); // final output projection after concatenation		

	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Multihead Attention forward" << std::endl;
		std::vector<Inferno::Tensor> heads;


		for (int i = 0; i < m_num_heads; ++i) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Head: " << i << std::endl;

			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - Q forward" << std::endl;
			auto q = Wq_layers[i].forward(x);

			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - K forward" << std::endl;
			auto k = Wk_layers[i].forward(x);

			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - V forward" << std::endl;
			auto v = Wv_layers[i].forward(x);

			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - Attn scores forward - transpose -> matmul -> Divide" << std::endl;
			auto attn_scores = Inferno::matmul(q, k.transpose(-1, -2), "QK^T") / std::sqrt(static_cast<float>(m_head_dim));

			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - softmax forward" << std::endl;
			auto attn_probs = Inferno::Softmax(attn_scores, -1);

			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - matmul forward - Attn x V" << std::endl;
			auto head = Inferno::matmul(attn_probs, v, "attn@V");

			heads.push_back(head);
		}

		// concatenate heads along embedding dim
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - concat forward" << std::endl;
		Inferno::Tensor concat = Inferno::concat(heads, -1);

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "MHA - Linear forward" << std::endl;
		return W_out.forward(concat);
	}

private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	std::vector<Inferno::Linear> Wq_layers;
	std::vector<Inferno::Linear> Wk_layers;
	std::vector<Inferno::Linear> Wv_layers;
	Inferno::Linear W_out;
};*/



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Class MultiHeadAttentionFast
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class MultiHeadAttentionFast : public Inferno::Module {
public:
	MultiHeadAttentionFast(size_t embed_dim, size_t num_heads) :
		m_embed_dim(embed_dim),
		m_num_heads(num_heads),
		m_head_dim(embed_dim / num_heads),
		W_out(embed_dim, embed_dim),
		Wqkv_layer(embed_dim, embed_dim * 3)
	{

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out); // final output projection after concatenation		

	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Multihead Attention forward" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		auto shape = x.shape();

		if (shape.size() != 3) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast expects [B, T, C]" << std::endl;
			exit(1);
		}

		size_t B = shape[0];
		size_t T = shape[1];
		size_t C = shape[2];

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Wqkv_layer weights and bias" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << Wqkv_layer << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		auto qkv = Wqkv_layer.forward(x);

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Wqkv_layer after linear" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << qkv << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor q = qkv.slice(2, 0, m_embed_dim - 1);                  // [B, T, C]
		Inferno::Tensor k = qkv.slice(2, m_embed_dim, 2 * m_embed_dim - 1);    // [B, T, C]
		Inferno::Tensor v = qkv.slice(2, 2 * m_embed_dim, 3 * m_embed_dim - 1);// [B, T, C]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Q after slice" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "K after slice" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "V after slice" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		q = q.contiguous();
		k = k.contiguous();
		v = v.contiguous();

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Q after contiguous" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "K after contiguous" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "V after contiguous" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		q = q.reshape({ B, T, m_num_heads, m_head_dim });           // [B, T, H, D]
		k = k.reshape({ B, T, m_num_heads, m_head_dim });
		v = v.reshape({ B, T, m_num_heads, m_head_dim });

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Q after reshape" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "K after reshape" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "V after reshape" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		q = q.transpose(1, 2);                                    // [B, H, T, D]
		k = k.transpose(1, 2);                                    // [B, H, T, D]
		v = v.transpose(1, 2);                                    // [B, H, T, D]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Q after transpose" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "K after transpose" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "V after transpose" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Transposing K for matmul with Q" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor kt = k.transpose(-1, -2);                          // [B, H, D, T]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "K after transpose" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Inferno::Tensor scores = matmul(q, kt, "QK^T");                             // [B, H, T, T]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "attn scores after matmul(q, kt)" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << scores << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		float scale = 1.0f / std::sqrt((float)m_head_dim);

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "calculating scaled attn scores using scale: " << scale << std::endl;		
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		
		scores = scores * scale;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "scores after scaling" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << scores << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Creating mask for attention" << std::endl;

		Inferno::Tensor ones(Inferno::DType::Int32, std::vector<int>(T * T, 1.0f), { 1, 1, T, T }, "causal_mask_ones", scores.device());

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Created Tensor with all 1's to serve as base for mask" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << ones << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor mask = Inferno::triu(ones, 1);		

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Created triu mask" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << mask << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		// mask out disallowed positions before softmax
		scores = Inferno::masked_fill(scores, mask, -1e9f);
		
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "scores after applying mask" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << scores << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Inferno::Tensor attn = Inferno::Softmax(scores, -1);                         // [B, H, T, T]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "attn scores after softmax" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << attn << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor y = matmul(attn, v, "attn@V");                                // [B, H, T, D]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "scores after matmul(attn, v)" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = y.transpose(1, 2);                                    // [B, T, H, D]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "y = y.transpose(1, 2); " << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = y.contiguous();

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "y = y.contiguous(); " << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = y.reshape({ B, T, m_embed_dim });                       // [B, T, C]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "y = y.reshape({ B, T, m_embed_dim }); " << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "W_out weights and bias" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << W_out << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = W_out.forward(y);                                   // [B, T, C]

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After Linear W_out" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		return y;
	}

private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	Inferno::Linear Wqkv_layer;
	Inferno::Linear W_out;
};



class MultiHeadAttentionFast2 : public Inferno::Module {
public:
	MultiHeadAttentionFast2(size_t embed_dim, size_t num_heads) :
		m_embed_dim(embed_dim),
		m_num_heads(num_heads),
		m_head_dim(embed_dim / num_heads),
		W_out(embed_dim, embed_dim),
		Wqkv_layer(embed_dim, embed_dim * 3),
		m_cached_T(0)
	{
		if (embed_dim % num_heads != 0) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		auto shape = x.shape();

		if (shape.size() != 3) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2 expects input of shape [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2 expected embed_dim = " << m_embed_dim
				<< " but got C = " << C << std::endl;
			exit(1);
		}

		// [B, T, 3C]
		Inferno::Tensor qkv = Wqkv_layer.forward(x);

		// Reshape once instead of slicing q/k/v first.
		// [B, T, 3, H, D]
		qkv = qkv.reshape({ B, T, 3, m_num_heads, m_head_dim });

		// Pull out q/k/v as views.
		// These slice calls assume inclusive end indices like your current code.
		Inferno::Tensor q = qkv.slice(2, 0, 0);  // [B, T, 1, H, D]
		Inferno::Tensor k = qkv.slice(2, 1, 1);  // [B, T, 1, H, D]
		Inferno::Tensor v = qkv.slice(2, 2, 2);  // [B, T, 1, H, D]

		// Remove the singleton "qkv selector" dimension.
		// [B, T, H, D]
		q = q.reshape({ B, T, m_num_heads, m_head_dim });
		k = k.reshape({ B, T, m_num_heads, m_head_dim });
		v = v.reshape({ B, T, m_num_heads, m_head_dim });

		// Move heads before sequence:
		// [B, H, T, D]
		q = q.transpose(1, 2);
		k = k.transpose(1, 2);
		v = v.transpose(1, 2);

		// [B, H, D, T]
		Inferno::Tensor kt = k.transpose(-1, -2);

		// [B, H, T, T]
		Inferno::Tensor scores = matmul(q, kt, "QK^T");

		const float scale = 1.0f / std::sqrt(static_cast<float>(m_head_dim));
		scores = scores * scale;

		const Inferno::Tensor& mask = get_or_build_causal_mask(T, scores.device());

		// mask shape is [1, 1, T, T], so it broadcasts over B and H
		scores = Inferno::masked_fill(scores, mask, -1e9f);

		// [B, H, T, T]
		Inferno::Tensor attn = Inferno::Softmax(scores, -1);

		// [B, H, T, D]
		Inferno::Tensor y = matmul(attn, v, "attn@V");

		// [B, T, H, D]
		y = y.transpose(1, 2);

		// Flatten heads back into embedding dim.
		// This contiguous() is usually the important one to keep before reshape.
		y = y.contiguous();
		y = y.reshape({ B, T, m_embed_dim });

		// Final projection: [B, T, C]
		y = W_out.forward(y);

		return y;
	}

private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	Inferno::Linear Wqkv_layer;
	Inferno::Linear W_out;

	// Cached causal mask
	Inferno::Tensor m_cached_mask;
	size_t m_cached_T;

	const Inferno::Tensor& get_or_build_causal_mask(size_t T, const Inferno::Device& device) {
		bool rebuild = false;

		if (!GetImpl(m_cached_mask)) {
			rebuild = true;
		}
		else if (m_cached_T != T) {
			rebuild = true;
		}
		else if (m_cached_mask.device() != device) {
			rebuild = true;
		}

		if (rebuild) {
			std::vector<int> ones_data(T * T, 1);

			Inferno::Tensor ones(
				Inferno::DType::Int32,
				ones_data,
				{ 1, 1, T, T },
				"causal_mask_ones",
				device
			);

			m_cached_mask = Inferno::triu(ones, 1);
			m_cached_T = T;
		}

		return m_cached_mask;
	}
};


class MultiHeadAttentionFast3 : public Inferno::Module {
public:
	MultiHeadAttentionFast3(size_t embed_dim, size_t num_heads) :
		m_embed_dim(embed_dim),
		m_num_heads(num_heads),
		m_head_dim(embed_dim / num_heads),
		W_out(embed_dim, embed_dim),
		Wqkv_layer(embed_dim, embed_dim * 3),
		m_cached_T(0)
	{
		if (embed_dim % num_heads != 0) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		auto shape = x.shape();

		if (shape.size() != 3) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2 expects [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2 expected embed_dim = " << m_embed_dim
				<< " but got " << C << std::endl;
			exit(1);
		}

		// [B, T, 3C]
		Inferno::Tensor qkv = Wqkv_layer.forward(x);

		// Keep this part compatible with your current reshape rules.
		Inferno::Tensor q = qkv.slice(2, 0, m_embed_dim - 1);                   // [B, T, C]
		Inferno::Tensor k = qkv.slice(2, m_embed_dim, 2 * m_embed_dim - 1);     // [B, T, C]
		Inferno::Tensor v = qkv.slice(2, 2 * m_embed_dim, 3 * m_embed_dim - 1); // [B, T, C]

		// Needed because your reshape currently only supports contiguous tensors.
		q = q.contiguous();
		k = k.contiguous();
		v = v.contiguous();

		// [B, T, H, D]
		q = q.reshape({ B, T, m_num_heads, m_head_dim });
		k = k.reshape({ B, T, m_num_heads, m_head_dim });
		v = v.reshape({ B, T, m_num_heads, m_head_dim });

		// [B, H, T, D]
		q = q.transpose(1, 2);
		k = k.transpose(1, 2);
		v = v.transpose(1, 2);

		// [B, H, D, T]
		Inferno::Tensor kt = k.transpose(-1, -2);

		// [B, H, T, T]
		Inferno::Tensor scores = matmul(q, kt, "QK^T");

		const float scale = 1.0f / std::sqrt(static_cast<float>(m_head_dim));
		scores = scores * scale;

		const Inferno::Tensor& mask = get_or_build_causal_mask(T, scores.device());
		scores = Inferno::masked_fill(scores, mask, -1e9f);

		// [B, H, T, T]
		Inferno::Tensor attn = Inferno::Softmax(scores, -1);

		// [B, H, T, D]
		Inferno::Tensor y = matmul(attn, v, "attn@V");

		// [B, T, H, D]
		y = y.transpose(1, 2);

		// This contiguous is still the important one before flattening heads.
		y = y.contiguous();
		y = y.reshape({ B, T, m_embed_dim });

		// [B, T, C]
		y = W_out.forward(y);

		return y;
	}

private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	Inferno::Linear Wqkv_layer;
	Inferno::Linear W_out;

	Inferno::Tensor m_cached_mask;
	size_t m_cached_T;

	const Inferno::Tensor& get_or_build_causal_mask(size_t T, const Inferno::Device& device) {
		bool rebuild = false;

		auto impl = GetImpl(m_cached_mask);
		if (!impl) {
			rebuild = true;
		}
		else if (m_cached_T != T) {
			rebuild = true;
		}
		else if (m_cached_mask.device() != device) {
			rebuild = true;
		}

		if (rebuild) {
			std::vector<int> ones_data(T * T, 1);

			Inferno::Tensor ones(
				Inferno::DType::Int32,
				ones_data,
				{ 1, 1, T, T },
				"causal_mask_ones",
				device
			);

			m_cached_mask = Inferno::triu(ones, 1);
			m_cached_T = T;
		}

		return m_cached_mask;
	}
};


class MultiHeadAttentionFast4 : public Inferno::Module {
public:
	MultiHeadAttentionFast4(size_t embed_dim, size_t num_heads) :
		m_embed_dim(embed_dim),
		m_num_heads(num_heads),
		m_head_dim(embed_dim / num_heads),
		W_out(embed_dim, embed_dim),
		Wqkv_layer(embed_dim, embed_dim * 3),
		m_cached_T(0)
	{
		if (embed_dim % num_heads != 0) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast4: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		auto shape = x.shape();

		if (shape.size() != 3) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast4 expects [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast4 expected embed_dim = " << m_embed_dim << " but got " << C << std::endl;
			exit(1);
		}

		// [B, T, 3C]
		Inferno::Tensor qkv = Wqkv_layer.forward(x);
		//t1.lap("Wqkv forward");

		// [B, T, C]
		Inferno::Tensor q = qkv.slice(2, 0, m_embed_dim - 1);
		Inferno::Tensor k = qkv.slice(2, m_embed_dim, 2 * m_embed_dim - 1);
		Inferno::Tensor v = qkv.slice(2, 2 * m_embed_dim, 3 * m_embed_dim - 1);

		// Make each projection contiguous before reshape
		q = q.contiguous();
		k = k.contiguous();
		v = v.contiguous();

		// [B, T, H, D]
		q = q.reshape({ B, T, m_num_heads, m_head_dim });
		k = k.reshape({ B, T, m_num_heads, m_head_dim });
		v = v.reshape({ B, T, m_num_heads, m_head_dim });

		// [B, H, T, D]
		// Important: materialize these layouts so batched matmul has a clean stride pattern
		q = q.transpose(1, 2).contiguous();
		k = k.transpose(1, 2).contiguous();
		v = v.transpose(1, 2).contiguous();

		// [B, H, D, T]
		Inferno::Tensor kt = k.transpose(-1, -2).contiguous();

		// [B, H, T, T]
		Inferno::Tensor scores = matmul(q, kt, "QK ^ T");
		//t1.lap("matmul QK ^T");

		const float scale = 1.0f / std::sqrt(static_cast<float>(m_head_dim));
		scores = scores * scale;
		//t1.lap("scores = scores * scale");

		Inferno::Tensor mask = get_or_build_causal_mask(T, scores.device());		
		scores = Inferno::masked_fill(scores, mask, -1e9f);
		//t1.lap("masked fill");
		// [B, H, T, T]
		Inferno::Tensor attn = Inferno::Softmax(scores, -1).contiguous();

		// [B, H, T, D]
		Inferno::Tensor y = matmul(attn, v, "attn@V");
		//t1.lap("attn@V");

		// [B, T, H, D]
		y = y.transpose(1, 2).contiguous();

		// [B, T, C]
		y = y.reshape({ B, T, m_embed_dim });

		// [B, T, C]
		y = W_out.forward(y);
		//t1.lap("W_out forward");

		return y;
	}

private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	Inferno::Linear Wqkv_layer;
	Inferno::Linear W_out;

	Inferno::Tensor m_cached_mask;
	size_t m_cached_T;

	const Inferno::Tensor& get_or_build_causal_mask(size_t T, const Inferno::Device& device) {
		bool rebuild = false;

		auto impl = GetImpl(m_cached_mask);
		if (!impl) {
			rebuild = true;
		}
		else if (m_cached_T != T) {
			rebuild = true;
		}
		else if (m_cached_mask.device() != device) {
			rebuild = true;
		}

		if (rebuild) {
			std::vector<int> ones_data(T * T, 1);

			Inferno::Tensor ones(
				Inferno::DType::Int32,
				ones_data,
				{ 1, 1, T, T },
				"causal_mask_ones",
				device
			);

			m_cached_mask = Inferno::triu(ones, 1);
			m_cached_T = T;
		}

		return m_cached_mask;
	}
};



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Class TransformerBlock
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class TransformerBlock : public Inferno::Module {
public:
	TransformerBlock(size_t embed_dim, size_t nheads)
		: attn(embed_dim, nheads),
		layernorm1(embed_dim),
		layernorm2(embed_dim),
		feedforward1(embed_dim, 4 * embed_dim),
		feedforward2(4 * embed_dim, embed_dim)
	{
		register_module("attn", &attn);
		register_module("ln1", &layernorm1);
		register_module("ln2", &layernorm2);
		register_module("ff1", &feedforward1);
		register_module("ff2", &feedforward2);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Feedforward 1" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << feedforward1 << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Feedforward 2" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << feedforward2 << std::endl;		
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Transformer Block forward" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		
		Inferno::Tensor normed = layernorm1.forward(x);
		//t1.lap("layernorm1 forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << normed << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor attn_out = attn.forward(normed);
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << attn_out << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		x = x + attn_out;
		//t1.lap("x = x + attn_out");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "after x = x + attn_out" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor normed2 = layernorm2.forward(x);
		//t1.lap("layernorm2 forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << normed2 << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Feedforward1 weights and bias" << std::endl;		
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << feedforward1 << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;		
				
		Inferno::Tensor n = feedforward1.forward(normed2);
		//t1.lap("feeforward1 forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After Feedforward 1" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << n << std::endl;

		n = Inferno::gelu(n);
		//t1.lap("gelu forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After gelu" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << n << std::endl;
		

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Feedforward2 weights and bias" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << feedforward2 << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor ff = feedforward2.forward(n);
		//t1.lap("feeforward2 forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After Feedforward 2" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << ff << std::endl;

		Inferno::Tensor out = x + ff;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "after out = x + ff" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << out << std::endl;		

		return out;
	}

private:
	MultiHeadAttentionFast4 attn;
	Inferno::LayerNorm layernorm1, layernorm2;
	Inferno::Linear feedforward1, feedforward2;
};






//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Class GPTModel
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GPTModel : public Inferno::Module {

public:



	GPTModel(size_t vocab_size, size_t context_size, size_t embed_dim, size_t nheads, size_t nblocks) :
		emb1(vocab_size, embed_dim),
		pos_enc(context_size, embed_dim),
		linear1(embed_dim, vocab_size),
		layernorm1(embed_dim) {

		m_embed_dim = embed_dim;
		m_context_size = context_size;
		m_vocab_size = vocab_size;

		//TODO: add these to the constructors?
		this->register_module("tok_embedding", &emb1);
		this->register_module("pos_encoding", &pos_enc);

		transblks.reserve(nblocks);
		for (size_t i = 0; i < nblocks; i++) {
			transblks.emplace_back(embed_dim, nheads);  // constructs Head(i)
			this->register_module("block" + std::to_string(i), & transblks[i]);
		}

		this->register_module("linear1", &linear1);
		this->register_module("ln1", &layernorm1);


	}

	Inferno::Tensor forward(Inferno::Tensor& input) {

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "GPTModel forward" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Input tensor" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << input << std::endl;
		//Get embedding vectors
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Embedding weights and bias" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << emb1 << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Inferno::Tensor x = emb1.forward(input);
		//t1.lap("Embedding forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After embedding layer" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		//Add positional encoding
		x = pos_enc.forward(x);
		//t1.lap("PE forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After positional encoding" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Starting loop of " << transblks.size() << " transormer blocks" << std::endl;		
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		// pump it through the Transformer blocks
		for (int blk_idx = 0; blk_idx < transblks.size(); blk_idx++) {
			//for (TransformerBlock tblk : transblks) {
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Block: " << blk_idx << std::endl;
			x = transblks[blk_idx].forward(x);
		}

		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Output of transformer blocks and input to layernorm" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		

		x = layernorm1.forward(x);
		//t1.lap("GPT model layernorm1 forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After layer norm" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		
		Inferno::Tensor logits = linear1.forward(x);
		//t1.lap("GPT Model linear1 forward");
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "After Linear" << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << logits << std::endl;
		Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		return logits;
	}

	Inferno::Embedding emb1;
	PositionalEncoding pos_enc;
	std::vector<TransformerBlock> transblks;
	Inferno::Linear linear1;
	Inferno::LayerNorm layernorm1;

	size_t m_context_size;
	size_t m_embed_dim;
	size_t m_vocab_size;




};



int main() {

	

	//Logg::SetLevel(Logg::LogLevel::LOGLEVEL_ERROR);
	//Logg::SetLevel(Logg::LogLevel::LOGLEVEL_DEBUG);
	Logg::SetLevel(Logg::LogLevel::LOGLEVEL_INFO);
	Logg::Start("logs/applicationlog.txt");

	

	Inferno::RandomGenerator::initializeWithSeed(42);

	
	
	//RunTests();



	InfernoTokenizer::BPETokenizer tok;
	tok.Initialize({ "data/shakemerges.txt", "data/shakevocab.txt" });


	DataLoader loader("data/shake.tokens",8,1024);
	

	
	
	


	///////////////////////////////////////////////////
	//
	//  HyperParams
	//
	///////////////////////////////////////////////////


	//Quick test
	//size_t vocabulary_size = 4;
	//size_t context_size = 4;
	//size_t embedding_dim = 5;
	//size_t numheads = 1;
	//size_t numblocks = 1;


	//Sane
	//size_t vocabulary_size = 32;
	//size_t context_size = 128;
	//size_t embedding_dim = 256;
	//size_t numheads = 1;
	//size_t numblocks = 1;


	//GPT 2
	size_t vocabulary_size = 22197;
	size_t context_size = 1024;
	size_t embedding_dim = 768;
	size_t numheads = 12;
	size_t numblocks = 12;

	 
	size_t batch_size = 8;


	/*std::vector<int> data(batch_size * context_size, 0);
	data[0] = 1;
	Inferno::Tensor target(Inferno::DType::Int32, data, { batch_size, context_size }, "target", device);
	Inferno::Tensor tokens(Inferno::DType::Int32, Inferno::RandomGenerator::generateRandomIntVector(batch_size * context_size, 0, vocabulary_size - 1), { batch_size, context_size }, "tokens", device);*/


	//Inferno::Tensor tokens(Inferno::DType::Int32, { 42, 13, 1, 0, 99, 34, 23, 78, 1, 25, 22, 45, 02, 13, 67, 88 }, { 16 }, "tokens", device);
	//Inferno::Tensor input = Inferno::Tensor(Inferno::DType::Float32, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 }, { 10 }, "input", device);


	//for mnist test
	//std::vector<size_t> layers({ 784,512,256,10 });
	//Inferno::Tensor input = Inferno::Tensor(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(layers[0],-0.5f,0.5f), { layers[0] }, "input", device);
	//Inferno::Tensor target = Inferno::Tensor(Inferno::DType::Float32, { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 10 }, "target", device);


	
	GPTModel model(vocabulary_size, context_size, embedding_dim, numheads, numblocks);

	Inferno::StateDict sd  = model.state_dict();
	

	/*for (const auto& [name, tensor] : sd) {
		std::cout << name << std::endl;

	}*/

//	Inferno::Checkpoint chkpt;
//	chkpt.set_state_dict(sd);
//	chkpt.save("myfirstcheckpoint");

	

	model.to(device);

	Inferno::CrossEntropyLoss loss_fn;

	auto params = model.parameters();
	//Inferno::OptimizerSGD optimizer(params, 0.01f);
	Inferno::OptimizerAdamW optimizer(params);

	

	/*Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Tokens into model" << std::endl;
	Logg::Append(Logg::LogLevel::LOGLEVEL_INFO) << tokens << std::endl;
	Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;

	Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Target" << std::endl;
	Logg::Append(Logg::LogLevel::LOGLEVEL_INFO) << target << std::endl;
	Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << std::endl;*/

	int epochs = 1;
	int loopcount = 10000;
	for (int e = 0; e < epochs; e++) {
		for (int i = 0; i < loopcount; i++) {

			t1.start();			

			std::pair<Inferno::Tensor, Inferno::Tensor> pair = loader.next_batch();

			Inferno::Tensor x = pair.first;
			Inferno::Tensor y = pair.second;

			x = x.to(device);
			y = y.to(device);
			
			Inferno::Tensor logits = model.forward(x);
			

			//for inference
			//Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG, "next logits slice");
			//Inferno::Tensor next_logits = x.slice(-2, m_context_size - 1, m_context_size - 1);
			//Inferno::Tensor next_logits = Inferno::select(x, -2, m_context_size - 1); // {B,V}
			//std::cout << next_logits << std::endl;		

			//std::cout << prediction << std::endl;
			//std::cout << target << std::endl;

			Inferno::Tensor loss = loss_fn(logits, y);

			//std::cout << loss << std::endl;

			
			loss.backward();


			optimizer.step();
			optimizer.zero_grad();


			t1.stop();



			Inferno::Tensor lossp = loss.to(Inferno::Device::cpu());

			Logg::Append(Logg::LogLevel::LOGLEVEL_INFO)
				<< std::fixed
				<< "Epoch: " << e
				<< " Iter: " << i
				<< "  total took: "
				<< std::setw(7) << std::setfill('0') << std::setprecision(3) << t1.elapsed_ms()
				<< " ms  Loss: "
				<< std::setw(13) << std::setfill('0') << std::setprecision(9) << lossp.item<float>()
				<< std::endl;	
 			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Fast mm: " << g_mmcountfast << std::endl;
			Logg::Append(Logg::LogLevel::LOGLEVEL_DEBUG) << "Slow mm: " << g_mmcountslow << std::endl;
			/*for (const auto& [label, count] : g_matmul_counts) {
				std::cout << label << ": " << count << std::endl;
			}*/
			g_matmul_counts.clear();
			g_mmcountfast = g_mmcountslow = 0;

		}
	}
	

	


	return 0;

}


