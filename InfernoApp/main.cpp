#include <unordered_map>
#include <inferno/inferno.h>
#include <infernotokenizer/bpetokenizer.h>
#include <inferno/util/logging.h>
#include "timer.h"
#include "dataloader.h"
#include <queue>

Timer t1("Performance Counter");

extern int g_mmcountcublasSgemm;
extern int g_mmcountslow;
extern int g_mmcountcublasSgemmStridedBatched;
extern std::unordered_map<std::string, size_t> g_matmul_counts;


//Inferno::Device device = Inferno::Device::cpu();
Inferno::Device device = Inferno::Device::cuda(0);

bool laptimingenabled = false;
bool mmstatsenabled = false;


CoreLogger::Logger logger;

class RunningAverage {
public:
	RunningAverage(size_t window)
		: m_window(window), m_sum(0.0)
	{
	}

	void add(double value) {
		m_values.push_back(value);
		m_sum += value;

		if (m_values.size() > m_window) {
			m_sum -= m_values.front();
			m_values.pop_front();
		}
	}

	double average() const {
		if (m_values.empty()) {
			return 0.0;
		}

		return m_sum / static_cast<double>(m_values.size());
	}

private:
	size_t m_window;
	double m_sum;
	std::deque<double> m_values;
};



class PositionalEncoding : public Inferno::Module {


public:

	PositionalEncoding(size_t context_size, size_t embed_dim) {

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding - Initializing buffers" << std::endl;;
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

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding - Creating tensor" << std::endl;
		pe = Inferno::Tensor(Inferno::DType::Float32, std::move(pe_data), { context_size, embed_dim }, "positional-encoding");

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding - Register Buffer" << std::endl;
		register_buffer("pe",&pe);

	}


	Inferno::Tensor forward(Inferno::Tensor& x) {
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding forward" << std::endl;
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

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Multihead Attention forward" << std::endl;
		std::vector<Inferno::Tensor> heads;


		for (int i = 0; i < m_num_heads; ++i) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Head: " << i << std::endl;

			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - Q forward" << std::endl;
			auto q = Wq_layers[i].forward(x);

			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - K forward" << std::endl;
			auto k = Wk_layers[i].forward(x);

			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - V forward" << std::endl;
			auto v = Wv_layers[i].forward(x);

			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - Attn scores forward - transpose -> matmul -> Divide" << std::endl;
			auto attn_scores = Inferno::matmul(q, k.transpose(-1, -2), "QK^T") / std::sqrt(static_cast<float>(m_head_dim));

			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - softmax forward" << std::endl;
			auto attn_probs = Inferno::Softmax(attn_scores, -1);

			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - matmul forward - Attn x V" << std::endl;
			auto head = Inferno::matmul(attn_probs, v, "attn@V");

			heads.push_back(head);
		}

		// concatenate heads along embedding dim
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - concat forward" << std::endl;
		Inferno::Tensor concat = Inferno::concat(heads, -1);

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "MHA - Linear forward" << std::endl;
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

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Multihead Attention forward" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		auto shape = x.shape();

		if (shape.size() != 3) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast expects [B, T, C]" << std::endl;
			exit(1);
		}

		size_t B = shape[0];
		size_t T = shape[1];
		size_t C = shape[2];

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Wqkv_layer weights and bias" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << Wqkv_layer << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		auto qkv = Wqkv_layer.forward(x);

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Wqkv_layer after linear" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << qkv << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor q = qkv.slice(2, 0, m_embed_dim - 1);                  // [B, T, C]
		Inferno::Tensor k = qkv.slice(2, m_embed_dim, 2 * m_embed_dim - 1);    // [B, T, C]
		Inferno::Tensor v = qkv.slice(2, 2 * m_embed_dim, 3 * m_embed_dim - 1);// [B, T, C]


		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Q after slice" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "K after slice" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "V after slice" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		q = q.contiguous();
		k = k.contiguous();
		v = v.contiguous();

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Q after contiguous" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "K after contiguous" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "V after contiguous" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		q = q.reshape({ B, T, m_num_heads, m_head_dim });           // [B, T, H, D]
		k = k.reshape({ B, T, m_num_heads, m_head_dim });
		v = v.reshape({ B, T, m_num_heads, m_head_dim });

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Q after reshape" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "K after reshape" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "V after reshape" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		q = q.transpose(1, 2);                                    // [B, H, T, D]
		k = k.transpose(1, 2);                                    // [B, H, T, D]
		v = v.transpose(1, 2);                                    // [B, H, T, D]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Q after transpose" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "K after transpose" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "V after transpose" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Transposing K for matmul with Q" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor kt = k.transpose(-1, -2);                          // [B, H, D, T]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "K after transpose" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Inferno::Tensor scores = matmul(q, kt, "QK^T");                             // [B, H, T, T]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "attn scores after matmul(q, kt)" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << scores << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		float scale = 1.0f / std::sqrt((float)m_head_dim);

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "calculating scaled attn scores using scale: " << scale << std::endl;		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		
		scores = scores * scale;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "scores after scaling" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << scores << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Creating mask for attention" << std::endl;

		Inferno::Tensor ones(Inferno::DType::Int32, std::vector<int>(T * T, 1.0f), { 1, 1, T, T }, "causal_mask_ones", scores.device());

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Created Tensor with all 1's to serve as base for mask" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << ones << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor mask = Inferno::triu(ones, 1);		

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Created triu mask" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << mask << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		// mask out disallowed positions before softmax
		scores = Inferno::masked_fill(scores, mask, -1e9f);
		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "scores after applying mask" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << scores << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Inferno::Tensor attn = Inferno::Softmax(scores, -1);                         // [B, H, T, T]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "attn scores after softmax" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << attn << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor y = matmul(attn, v, "attn@V");                                // [B, H, T, D]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "scores after matmul(attn, v)" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = y.transpose(1, 2);                                    // [B, T, H, D]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y = y.transpose(1, 2); " << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = y.contiguous();

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y = y.contiguous(); " << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = y.reshape({ B, T, m_embed_dim });                       // [B, T, C]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y = y.reshape({ B, T, m_embed_dim }); " << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "W_out weights and bias" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << W_out << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		y = W_out.forward(y);                                   // [B, T, C]

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After Linear W_out" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

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
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		auto shape = x.shape();

		if (shape.size() != 3) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2 expects input of shape [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
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
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		auto shape = x.shape();

		if (shape.size() != 3) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast2 expects [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
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
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast4: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		auto shape = x.shape();

		if (shape.size() != 3) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast4 expects [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast4 expected embed_dim = " << m_embed_dim << " but got " << C << std::endl;
			exit(1);
		}

		// [B, T, 3C]
		Inferno::Tensor qkv = Wqkv_layer.forward(x);
		if (laptimingenabled) t1.lap("Wqkv forward");

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
		if (laptimingenabled) t1.lap("matmul QK ^T");

		const float scale = 1.0f / std::sqrt(static_cast<float>(m_head_dim));
		scores = scores * scale;
		if (laptimingenabled) t1.lap("scores = scores * scale");

		Inferno::Tensor mask = get_or_build_causal_mask(T, scores.device());		
		scores = Inferno::masked_fill(scores, mask, -1e9f);
		if (laptimingenabled) t1.lap("masked fill");
		// [B, H, T, T]
		Inferno::Tensor attn = Inferno::Softmax(scores, -1).contiguous();

		// [B, H, T, D]
		Inferno::Tensor y = matmul(attn, v, "attn@V");
		if (laptimingenabled) t1.lap("attn@V");

		// [B, T, H, D]
		y = y.transpose(1, 2).contiguous();

		// [B, T, C]
		y = y.reshape({ B, T, m_embed_dim });

		// [B, T, C]
		y = W_out.forward(y);
		if (laptimingenabled) t1.lap("W_out forward");

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



class MultiHeadAttentionFast5 : public Inferno::Module {
public:
	MultiHeadAttentionFast5(size_t embed_dim, size_t num_heads) :
		m_embed_dim(embed_dim),
		m_num_heads(num_heads),
		m_head_dim(embed_dim / num_heads),
		W_out(embed_dim, embed_dim),
		Wqkv_layer(embed_dim, embed_dim * 3)
	{
		if (embed_dim % num_heads != 0) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast5: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		std::vector<size_t> shape = x.shape();

		if (shape.size() != 3) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast5 expects [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast5 expected embed_dim = " << m_embed_dim << " but got " << C << std::endl;
			exit(1);
		}

		// [B, T, 3C]
		Inferno::Tensor qkv = Wqkv_layer.forward(x);
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "qkv after Wqkv_layer.forward(x)" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << qkv << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		// [B, T, C]
		Inferno::Tensor q = qkv.slice(2, 0, m_embed_dim - 1);
		Inferno::Tensor k = qkv.slice(2, m_embed_dim, 2 * m_embed_dim - 1);
		Inferno::Tensor v = qkv.slice(2, 2 * m_embed_dim, 3 * m_embed_dim - 1);

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "q after slice" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << q << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "k after slice" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << k << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "v after slice" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << v << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		q = q.contiguous();
		k = k.contiguous();
		v = v.contiguous();

		// [B, T, H, D]
		q = q.reshape({ B, T, m_num_heads, m_head_dim });
		k = k.reshape({ B, T, m_num_heads, m_head_dim });
		v = v.reshape({ B, T, m_num_heads, m_head_dim });

		// [B, H, T, D]
		q = q.transpose(1, 2).contiguous();
		k = k.transpose(1, 2).contiguous();
		v = v.transpose(1, 2).contiguous();

		// [B, H, T, D]
		//
		// This replaces:
		//
		// kt = k.transpose(-1, -2).contiguous();
		// scores = matmul(q, kt);
		// scores = scores * scale;
		// scores = masked_fill(scores, mask, -1e9f);
		// attn = Softmax(scores, -1).contiguous();
		// y = matmul(attn, v);
		//
		Inferno::Tensor y = Inferno::flash_attention_simple_forward(q, k, v, true);
		if (laptimingenabled) t1.lap("flash_attention");

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y after flash_attention" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		// [B, T, H, D]
		y = y.transpose(1, 2).contiguous();

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y after transpose" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		// [B, T, C]
		y = y.reshape({ B, T, m_embed_dim });
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y after reshape" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		// [B, T, C]
		y = W_out.forward(y);
		if (laptimingenabled) t1.lap("W_out forward");

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "W_out weights and biases" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << W_out << std::endl;		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;



		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y after  W_out.forward(y)" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		
		
		return y;
	}

//private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	Inferno::Linear W_out;
	Inferno::Linear Wqkv_layer;
};


class MultiHeadAttentionFast6 : public Inferno::Module {
public:
	MultiHeadAttentionFast6(size_t embed_dim, size_t num_heads) :
		m_embed_dim(embed_dim),
		m_num_heads(num_heads),
		m_head_dim(embed_dim / num_heads),
		W_out(embed_dim, embed_dim),
		Wqkv_layer(embed_dim, embed_dim * 3)
	{
		if (embed_dim % num_heads != 0) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR)
				<< "MultiHeadAttentionFast5: embed_dim must be divisible by num_heads" << std::endl;
			exit(1);
		}

		register_module("Wqkv", &Wqkv_layer);
		register_module("W_out", &W_out);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		std::vector<size_t> shape = x.shape();

		if (shape.size() != 3) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast6 expects [B, T, C]" << std::endl;
			exit(1);
		}

		const size_t B = shape[0];
		const size_t T = shape[1];
		const size_t C = shape[2];

		if (C != m_embed_dim) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_ERROR) << "MultiHeadAttentionFast6 expected embed_dim = " << m_embed_dim << " but got " << C << std::endl;
			exit(1);
		}

		// [B, T, 3C]
		Inferno::Tensor qkv = Wqkv_layer.forward(x);
		

		// [B, H, T, D]
		//
		// This replaces:
		//
		// kt = k.transpose(-1, -2).contiguous();
		// scores = matmul(q, kt);
		// scores = scores * scale;
		// scores = masked_fill(scores, mask, -1e9f);
		// attn = Softmax(scores, -1).contiguous();
		// y = matmul(attn, v);
		//
		Inferno::Tensor y = Inferno::flash_attention_bigdaddy_forward(qkv, m_num_heads, true);
		if (laptimingenabled) t1.lap("flash_attention");
		
		// [B, T, C]
		y = W_out.forward(y);

		if (laptimingenabled) t1.lap("W_out forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y after  W_out.forward(y)" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		return y;
	}

	//private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	Inferno::Linear W_out;
	Inferno::Linear Wqkv_layer;
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

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Feedforward 1" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << feedforward1 << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Feedforward 2" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << feedforward2 << std::endl;		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Transformer Block forward" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		
		Inferno::Tensor normed = layernorm1.forward(x);
		//t1.lap("layernorm1 forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << normed << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor attn_out = attn.forward(normed);
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "attn_out after  attn.forward(normed)" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << attn_out << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		x = x + attn_out;
		if (laptimingenabled) t1.lap("x = x + attn_out");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "after x = x + attn_out" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor normed2 = layernorm2.forward(x);
		if (laptimingenabled) t1.lap("layernorm2 forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << normed2 << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Feedforward1 weights and bias" << std::endl;		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << feedforward1 << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;		
				
		Inferno::Tensor n = feedforward1.forward(normed2);
		if (laptimingenabled) t1.lap("feedforward1 forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After Feedforward 1" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << n << std::endl;

		n = Inferno::gelu(n);
		if (laptimingenabled) t1.lap("gelu forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After gelu" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << n << std::endl;
		

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Feedforward2 weights and bias" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << feedforward2 << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		Inferno::Tensor ff = feedforward2.forward(n);
		if (laptimingenabled) t1.lap("feedforward2 forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After Feedforward 2" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << ff << std::endl;

		Inferno::Tensor out = x + ff;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "after out = x + ff" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << out << std::endl;		

		return out;
	}

//private:
	MultiHeadAttentionFast6 attn;
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

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "GPTModel forward" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Input tensor" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << input << std::endl;
		//Get embedding vectors
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Embedding weights and bias" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << emb1 << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		Inferno::Tensor x = emb1.forward(input);
		if (laptimingenabled) t1.lap("Embedding forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After embedding layer" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		//Add positional encoding
		x = pos_enc.forward(x);
		if (laptimingenabled) t1.lap("PE forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After positional encoding" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;


		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Starting loop of " << transblks.size() << " transormer blocks" << std::endl;		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		// pump it through the Transformer blocks
		for (int blk_idx = 0; blk_idx < transblks.size(); blk_idx++) {
			//for (TransformerBlock tblk : transblks) {
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Block: " << blk_idx << std::endl;
			x = transblks[blk_idx].forward(x);
		}

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Output of transformer blocks and input to layernorm" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		

		x = layernorm1.forward(x);
		if (laptimingenabled) t1.lap("GPT model layernorm1 forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After layer norm" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << x << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Linear1 weights and bias" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << linear1 << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		//Inferno::Tensor logits = linear1.forward(x);				

		size_t B = x.shape()[0];
		size_t T = x.shape()[1];
		size_t C = x.shape()[2];

		Inferno::Tensor x2d = x.reshape({ B * T, C });          // [B*T, C]

		Inferno::Tensor logits2d = linear1.forward(x2d);        // [B*T, V]

		size_t V = logits2d.shape()[1];

		Inferno::Tensor logits = logits2d.reshape({ B, T, V }); // [B, T, V]

	
		if (laptimingenabled) t1.lap("GPT Model linear1 forward");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "After Linear" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << logits << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

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


void save_checkpoint(Inferno::Module model,Inferno::OptimizerAdamW optimizer, size_t step, size_t total_steps) {
	logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Writing Checkpoint" << std::endl;
	Inferno::Checkpoint chkpt;
	chkpt.meta = Inferno::TrainingMetadata(step, total_steps, 0, 1);
	chkpt.model = model.state_dict();
	chkpt.optimizer = optimizer.state_dict();
	chkpt.save("checkpoints\\largeckpt.bin");
}


Inferno::Tensor naive_attention_from_qkv(
	Inferno::Tensor& qkv,
	size_t num_heads,
	bool causal
) {
	std::vector<size_t> s = qkv.shape();

	size_t B = s[0];
	size_t T = s[1];
	size_t threeC = s[2];
	size_t C = threeC / 3;
	size_t D = C / num_heads;

	Inferno::Tensor q = qkv.slice(2, 0, C - 1).contiguous();
	Inferno::Tensor k = qkv.slice(2, C, 2 * C - 1).contiguous();
	Inferno::Tensor v = qkv.slice(2, 2 * C, 3 * C - 1).contiguous();

	q = q.reshape({ B, T, num_heads, D }).transpose(1, 2).contiguous();
	k = k.reshape({ B, T, num_heads, D }).transpose(1, 2).contiguous();
	v = v.reshape({ B, T, num_heads, D }).transpose(1, 2).contiguous();

	Inferno::Tensor kt = k.transpose(-1, -2).contiguous();

	Inferno::Tensor scores = matmul(q, kt, "naive_qk");

	float scale = 1.0f / std::sqrt(static_cast<float>(D));
	scores = scores * scale;

	if (causal) {
		std::vector<float> mask_data;

		for (size_t b = 0; b < B; b++) {
			for (size_t h = 0; h < num_heads; h++) {
				for (size_t i = 0; i < T; i++) {
					for (size_t j = 0; j < T; j++) {
						mask_data.push_back(j > i ? 1.0f : 0.0f);
					}
				}
			}
		}

		Inferno::Tensor mask(
			Inferno::DType::Int32,
			mask_data,
			{ B, num_heads, T, T },
			"mask",
			qkv.device(),
			false
		);

		scores = masked_fill(scores, mask, -1.0e9f);
	}

	Inferno::Tensor attn = Inferno::Softmax(scores, -1).contiguous();

	Inferno::Tensor y = matmul(attn, v, "naive_attn_v");

	y = y.transpose(1, 2).contiguous();
	y = y.reshape({ B, T, C });

	return y;
}


int main(int argc, char* argv[]) {


	bool resume = false;
	std::string ckpt_path;

	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];

		if (arg == "--resume" && i + 1 < argc) {
			resume = true;
			ckpt_path = argv[++i];
		}
	}


	/*{
		

		const size_t B = 1;
		const size_t T = 3;
		const size_t H = 1;
		const size_t D = 64;
		const size_t C = H * D;*/

		//std::vector<float> qkv_data;
		//qkv_data.reserve(B * T * 3 * C);

		/*for (size_t i = 0; i < B * T * 3 * C; i++) {
			float v = std::sin(float(i) * 0.1f) * 0.5f;
			qkv_data.push_back(v);
		}

		std::vector<float> w_data;
		w_data.reserve(B * T * C);

		for (size_t i = 0; i < B * T * C; i++) {
			float v = std::cos(float(i) * 0.07f) * 0.3f;
			w_data.push_back(v);
		}*/

		/*std::vector<float> qkv_data(B * T * 3 * C, 0.0f);
		std::vector<float> w_data(B * T * C, 0.0f);

		auto Q = [&](size_t t, size_t d) -> float& {
			return qkv_data[t * 3 * C + d];
		};

		auto K = [&](size_t t, size_t d) -> float& {
			return qkv_data[t * 3 * C + C + d];
		};

		auto V = [&](size_t t, size_t d) -> float& {
			return qkv_data[t * 3 * C + 2 * C + d];
		};

		auto dO = [&](size_t t, size_t d) -> float& {
			return w_data[t * C + d];
		};

		// token 0
		Q(0, 0) = 1.0f; Q(0, 1) = 0.0f;
		K(0, 0) = 1.0f; K(0, 1) = 0.0f;
		V(0, 0) = 10.0f; V(0, 1) = 20.0f;

		// token 1
		Q(1, 0) = 0.0f; Q(1, 1) = 1.0f;
		K(1, 0) = 0.0f; K(1, 1) = 1.0f;
		V(1, 0) = 30.0f; V(1, 1) = 40.0f;

		// token 2
		Q(2, 0) = 1.0f; Q(2, 1) = 1.0f;
		K(2, 0) = 1.0f; K(2, 1) = 1.0f;
		V(2, 0) = 50.0f; V(2, 1) = 60.0f;

		// upstream grad only for first two output dims
		dO(0, 0) = 1.0f; dO(0, 1) = 1.0f;
		dO(1, 0) = 1.0f; dO(1, 1) = 1.0f;
		dO(2, 0) = 1.0f; dO(2, 1) = 1.0f;

		
		Inferno::Tensor qkv_flash(
			Inferno::DType::Float32,
			qkv_data,
			{ B, T, 3 * C },
			"qkv_flash",
			device,
			true
		);

		Inferno::Tensor qkv_naive(
			Inferno::DType::Float32,
			qkv_data,
			{ B, T, 3 * C },
			"qkv_naive",
			device,
			true
		);

		Inferno::Tensor w(
			Inferno::DType::Float32,
			w_data,
			{ B, T, C },
			"w",
			device,
			false
		);


		/*Inferno::Tensor x(Inferno::DType::Float32, { 1,2,3,4,5,6 }, { 2,3 }, "x", device, true);

		Inferno::Tensor mask(Inferno::DType::Int32, { 0,1,0,1,0,1 }, { 2,3 }, "mask", device, false);

		Inferno::Tensor y = masked_fill(x, mask, -1000.0f);

		Inferno::Tensor loss = y.sum();

		loss.backward();

		std::cout << *x.grad() << std::endl;*/






		/*Inferno::Tensor y_flash = Inferno::flash_attention_bigdaddy_forward(qkv_flash, H, true);

		Inferno::Tensor y_naive = naive_attention_from_qkv(qkv_naive, H, true);

		std::cout << "================ y_flash ================\n";
		std::cout << y_flash << std::endl;

		std::cout << "================ y_naive ================\n";
		std::cout << y_naive << std::endl;

		Inferno::Tensor loss_flash = (y_flash * w).sum();
		Inferno::Tensor loss_naive = (y_naive * w).sum();

		loss_flash.backward();
		loss_naive.backward();

		std::cout << "================ qkv_flash.grad ================\n";
		std::cout << *qkv_flash.grad() << std::endl;

		std::cout << "================ qkv_naive.grad ================\n";
		std::cout << *qkv_naive.grad() << std::endl;
	}





	exit(1);*/




	logger.Start("logs/inferno.txt");

	Inferno::Logger::SetLogger(&logger);
	Inferno::Logger::EnableLogging();

	logger.SetLevel(CoreLogger::Logger::LogLevel::LOGLEVEL_INFO);
	//logger.SetLevel(CoreLogger::Logger::LogLevel::LOGLEVEL_DEBUG);



	//Inferno::Logger::SetLevel(Inferno::Logger::LogLevel::LOGLEVEL_ERROR);
	//Inferno::Logger::SetLevel(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG);
	//Inferno::Logger::SetLevel(Inferno::Logger::LogLevel::LOGLEVEL_INFO);	
	//Inferno::Logger::EnableLogging();


	//Inferno::EnableLogging("test.txt");	

	Inferno::RandomGenerator::initializeWithSeed(42);



	//RunTests();

//	Inferno::Tensor a(Inferno::DType::Float32, { -0.607237f, 0.448901f, 0.110358f, -0.072336f, -0.554881f, 0.489027f, 0.025490f, -0.068161f, -0.490913f, 0.548747f, -0.155777f, 0.127474f, -0.411033f, 0.337617f, 0.063233f, -0.092226f }, {2, 2, 4 }, "a", device, true);
//	Inferno::Tensor b(Inferno::DType::Float32, { 0.317400f, -0.842100f, 0.559300f, -0.128700f, 0.903500f, -0.476200f, 0.241800f, 0.668900f, -0.735400f, 0.182600f, -0.591700f, 0.427300f, 0.804100f, -0.259800f, 0.613200f, -0.094500f }, { 4, 4 }, "b", device, true);
//	Inferno::Tensor c(Inferno::DType::Float32, { 0.482100f, -0.713400f, 0.256800f, -0.905700f }, { 4 }, "c", device, true);

//	Inferno::Tensor y = Inferno::matmul(a, b);
//	std::cout << y << std::endl;
//	y = y+c;
//	std::cout << y << std::endl;
















	///////////////////////////////////////////////////
	//
	//  HyperParams
	//
	///////////////////////////////////////////////////


	//Quick test
	//size_t vocabulary_size = 6;
	//size_t context_size = 2;
	//size_t embedding_dim = 4;
	//size_t numheads = 2;
	//size_t numblocks = 1;


	//Sane
	//size_t vocabulary_size = 32;
	//size_t context_size = 128;
	//size_t embedding_dim = 256;
	//size_t numheads = 1;
	//size_t numblocks = 1;


	//GPT 2
	size_t vocabulary_size = 60259;
	size_t context_size = 1024;
	size_t embedding_dim = 768;
	size_t numheads = 12;
	size_t numblocks = 12;


	size_t batch_size = 8;
	size_t steps_per_chunk = 8192;

	InfernoTokenizer::BPETokenizer tok;
	tok.Initialize({ "data\\openwebtext_merges2.txt", "data\\openwebtext_vocab2.txt" });


	DataLoader loader("data\\openwebtext_clean3.tokens", batch_size, context_size, steps_per_chunk);


	


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

	int checkpoint_interval = 1000;
	int total_steps = 1000000;
	int step = 0;
	float lowestloss = 99;


	//Inferno::Tensor input = Inferno::Tensor(Inferno::DType::Int32, { 0,1,1,0 }, { 2, 2 }, "input", device, true);
	//Inferno::Tensor target = Inferno::Tensor(Inferno::DType::Int32, { 1,0,0,1 }, { 2, 2 }, "target", device, true);

	RunningAverage avg(500);

	
	GPTModel model(vocabulary_size, context_size, embedding_dim, numheads, numblocks);


	/*model.emb1.m_embeddings = Inferno::Tensor(Inferno::DType::Float32, { 0.7f, 0.2f, 0.3f, 0.7f, 0.1f, 0.8f, 0.3f, 0.2f }, { 2, 4 }, "embedding", device, true);
	
	model.transblks[0].attn.Wqkv_layer.m_weights = Inferno::Tensor(Inferno::DType::Float32, { 0.2341f, -0.1123f,  0.0456f,  0.1789f,   -0.0678f,  0.2567f, -0.1987f,  0.0345f,    0.1456f, -0.0891f,  0.0678f, -0.1234f,
                                                                                             -0.0567f,  0.1987f, -0.1678f,  0.0891f,    0.1456f, -0.0789f,  0.2345f, -0.1567f,   -0.0345f,  0.1678f, -0.0987f,  0.0567f,
                                                                                              0.1789f,  0.0345f,  0.2123f, -0.1456f,   -0.1234f,  0.1890f,  0.0789f, -0.0678f,    0.1987f, -0.1678f,  0.0456f,  0.0345f,
                                                                                             -0.0891f,  0.1234f, -0.0456f,  0.2567f,    0.0678f, -0.1987f,  0.1456f,  0.0891f,   -0.1567f,  0.0345f,  0.1789f, -0.0678f }, { 4, 12 }, "weights", device, true);
	model.transblks[0].attn.Wqkv_layer.m_biases = Inferno::Tensor(Inferno::DType::Float32, { 0.0345f, -0.0123f, 0.0567f, -0.0234f,  0.0456f, 0.0000f, -0.0345f, 0.0123f,  -0.0456f, 0.0678f, 0.0000f, -0.0234f }, { 12 }, "biases", device, true);
	model.transblks[0].attn.W_out.m_weights = Inferno::Tensor(Inferno::DType::Float32, { 0.3174f, -0.8421f, 0.5593f, -0.1287f, 0.9035f, -0.4762f, 0.2418f, 0.6689f,-0.7354f, 0.1826f, -0.5917f, 0.4273f,0.8041f, -0.2598f, 0.6132f, -0.0945f }, { 4, 4 }, "weights", device, true);
	model.transblks[0].attn.W_out.m_biases = Inferno::Tensor(Inferno::DType::Float32, { 0.4821f, -0.7134f, 0.2568f, -0.9057f }, { 4 }, "biases", device, true);


	model.transblks[0].feedforward1.m_weights = Inferno::Tensor(Inferno::DType::Float32, { 0.1325f, -0.4821f, 0.7734f, -0.2156f, -0.9043f, 0.5567f, -0.3312f, 0.1189f, 0.6721f, -0.7458f, 0.2294f, -0.5673f, 0.4410f, 0.0897f, -0.9982f, 0.3105f,
		-0.2736f, 0.8142f, -0.6521f, 0.1933f, 0.5076f, -0.1198f, 0.2844f, -0.7765f, -0.0612f, 0.9327f, -0.4875f, 0.3651f, -0.7219f, 0.2480f, -0.1567f, 0.6014f,
		0.3945f, -0.8423f, 0.7108f, -0.2299f, 0.1556f, 0.4782f, -0.6124f, 0.8871f, -0.3407f, 0.0625f, 0.5248f, -0.9089f, 0.2713f, -0.4441f, 0.7932f, -0.1180f,
		0.6679f, -0.7354f, 0.2011f, -0.5896f, 0.4538f, 0.0976f, -0.9652f, 0.3227f, -0.2845f, 0.8013f, -0.6317f, 0.1742f, -0.5128f, 0.2364f, 0.9147f, -0.4072f }, { 4, 16 }, "weights", device, true);
	model.transblks[0].feedforward1.m_biases = Inferno::Tensor(Inferno::DType::Float32, { 0.1243f, -0.5521f, 0.3417f, -0.2198f, 0.7785f, -0.6612f, 0.0954f, 0.4376f, -0.3821f, 0.5298f, -0.1476f, 0.6892f, -0.9044f, 0.2167f, -0.0735f, 0.5589f }, { 16 }, "biases", device, true);

	model.transblks[0].feedforward2.m_weights = Inferno::Tensor(Inferno::DType::Float32, { 0.2154f, -0.7812f, 0.4421f, -0.1093f, 0.6638f, -0.5527f, 0.1789f, 0.3045f, -0.9211f, 0.5872f, -0.3348f, 0.7110f, -0.2486f, 0.4953f, -0.6674f, 0.1327f,
		0.8531f, -0.4025f, 0.2197f, -0.7568f, 0.3742f, 0.6189f, -0.1456f, 0.0891f, -0.5324f, 0.7773f, -0.2985f, 0.5602f, -0.6147f, 0.2439f, 0.9812f, -0.4763f,
		0.1248f, -0.3397f, 0.7056f, -0.8821f, 0.4683f, 0.1914f, -0.7235f, 0.3569f, -0.2076f, 0.6408f, -0.5189f, 0.2741f, 0.8327f, -0.9614f, 0.1175f, 0.5032f,
		-0.6893f, 0.4428f, -0.1359f, 0.7981f, -0.3774f, 0.6205f, 0.2149f, -0.5562f, 0.9037f, -0.2481f, 0.4716f, -0.6670f, 0.3592f, -0.8043f, 0.5268f, 0.1124f }, { 16, 4 }, "weights", device, true);
	model.transblks[0].feedforward2.m_biases = Inferno::Tensor(Inferno::DType::Float32, { 0.1432f, -0.5526f, 0.3387f, 0.7611f }, { 4 }, "biases", device, true);
	




	model.linear1.m_weights = Inferno::Tensor(Inferno::DType::Float32, { 0.4127f, -0.7351f, 0.2894f, 0.9632f, -0.1548f, 0.6783f, -0.4926f, 0.1059f}, { 4,2 }, "weights", device, true);
	model.linear1.m_biases = Inferno::Tensor(Inferno::DType::Float32, { -0.3842f, 0.9176f }, { 2 }, "bias", device, true);
	*/

	std::optional<Inferno::Checkpoint> ckpt;

	//resume = false;

	if (resume) {
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "Resuming training from: " << ckpt_path << std::endl;
		ckpt = Inferno::Checkpoint::load(ckpt_path);

		step = ckpt->meta.step;
		total_steps = ckpt->meta.total_steps;

		model.load_state_dict(ckpt->model);
	}
	
	model.to(device);

	auto params = model.parameters();

	//Inferno::OptimizerAdamW optimizer(params);	
	Inferno::OptimizerAdamW optimizer(model.parameters(), 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f);

	if (resume) {
		optimizer.load_state_dict(ckpt->optimizer);
	}	


	Inferno::CrossEntropyLoss loss_fn;
	//std::pair<Inferno::Tensor, Inferno::Tensor> pair = loader.next_batch();
	
	for (; step < total_steps; step++) {

		t1.start();			

		std::pair<Inferno::Tensor, Inferno::Tensor> pair = loader.next_batch();

		Inferno::Tensor x = pair.first;
		Inferno::Tensor y = pair.second;

		/*auto blahx = x[0].to_vector<int>();
		auto blahy = y[0].to_vector<int>();

		std::string sx = tok.decode(blahx);
		std::string sy = tok.decode(blahy);

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "**************************** Tensor X ****************************" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << sx.substr(0,32) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;	


		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "**************************** Tensor Y ****************************" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << sy.substr(0, 32) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;*/

		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "********************** Chars per token" << std::endl;
		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << (float)sx.size() / (float)x[0].numel() << std::endl;
		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;
		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;

		x = x.to(device);
		y = y.to(device);
			
		//Inferno::Tensor logits = model.forward(input);
		Inferno::Tensor logits = model.forward(x);
			

		//for inference
		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG, "next logits slice");
		//Inferno::Tensor next_logits = x.slice(-2, m_context_size - 1, m_context_size - 1);
		//Inferno::Tensor next_logits = Inferno::select(x, -2, m_context_size - 1); // {B,V}
		//std::cout << next_logits << std::endl;		

		//std::cout << prediction << std::endl;
		//std::cout << target << std::endl;

		//Inferno::Tensor loss = loss_fn(logits, target);
		Inferno::Tensor loss = loss_fn(logits, y);
		if (laptimingenabled) t1.lap("loss");
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Loss" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << loss << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

			
		loss.backward();
		if (laptimingenabled) t1.lap("backward");


		optimizer.step();
		optimizer.zero_grad();


		t1.stop();


		//std::cout << model.emb1.m_embeddings;

		if (laptimingenabled) {
			std::vector<TimerLapResult> results = t1.lap_results();
			for (TimerLapResult res : results) {
				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO)					
					<< res.label
					<< ": "
					<< std::fixed
					<< std::setprecision(3)
					<< res.ms
					<< " ms\n";
			}
		}

		Inferno::Tensor lossp = loss.to(Inferno::Device::cpu());
		if (lossp.item<float>() < lowestloss)
			lowestloss = lossp.item<float>();

		avg.add(lossp.item<float>());

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO)
			<< std::fixed
			<< "Iter: " << step
			<< " | Percent complete: "
			<< std::setw(7) << std::setfill(' ') << std::setprecision(3) << static_cast<float>(step) / static_cast<float>(total_steps) * 100.0f
			<< "% | total took: "
			<< std::setw(7) << std::setfill('0') << std::setprecision(3) << t1.elapsed_ms()
			<< " ms | Loss: "
			<< std::setw(9) << std::setfill('0') << std::setprecision(5) << lossp.item<float>()
			<< " | Lowest: "
			<< std::setw(9) << std::setfill('0') << std::setprecision(5) << lowestloss
			<< " | Average: "
			<< std::setw(9) << std::setfill('0') << std::setprecision(5) << avg.average()
			<< std::endl;


		if (mmstatsenabled) {
 			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "cublasSgemm mm: " << g_mmcountcublasSgemm << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "cublasSgemmStridedBatched mm: " << g_mmcountcublasSgemmStridedBatched << std::endl;
			logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "Slow mm: " << g_mmcountslow << std::endl;
			/*for (const auto& [label, count] : g_matmul_counts) {
				std::cout << label << ": " << count << std::endl;
			}*/
		}
		g_matmul_counts.clear();
		g_mmcountcublasSgemm = g_mmcountcublasSgemmStridedBatched = g_mmcountslow = 0;

		//save incremental checkpoint
		if (step != 0 && step % checkpoint_interval == 0) {
			save_checkpoint(model, optimizer, step, total_steps);
		}			
	}

	//training done, save final checkpoint
	save_checkpoint(model, optimizer, step, total_steps);


	return 0;

}


