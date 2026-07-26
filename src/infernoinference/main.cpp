#include <unordered_map>
#include <inferno/inferno.h>
#include <infernotokenizer/bpetokenizer.h>
#include <inferno/util/logging.h>
#include "timer.h"
#include <queue>

Timer t1("Performance Counter");



bool laptimingenabled = false;

//Inferno::Device device = Inferno::Device::cpu();
Inferno::Device device = Inferno::Device::cuda(0);

CoreLogger::Logger logger;


// Returns true if every element of O_test matches O_ref within tolerance, false otherwise.
bool outputs_match(float* O_ref, float* O_test, size_t n, float abs_tol = 1e-3f, float rel_tol = 1e-3f) {

	for (size_t i = 0; i < n; i++) {
		float a = O_ref[i];
		float b = O_test[i];

		if (!std::isfinite(a) || !std::isfinite(b)) {
			return false;
		}

		float abs_diff = std::fabs(a - b);
		float rel_diff = abs_diff / std::max(std::fabs(a), 1e-6f);

		if (abs_diff > abs_tol && rel_diff > rel_tol) {
			return false;
		}
	}
	return true;
}



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
		register_buffer("pe", &pe);

	}


	Inferno::Tensor forward(Inferno::Tensor& x) {
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Positional Encoding forward" << std::endl;
		int T = x.shape()[1];
		Inferno::Tensor pe_slice = pe.slice(0, 0, T-1);  // take rows [0:T]
		return x + pe_slice;
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
		//Inferno::Tensor y = Inferno::flash_attention_bigdaddy_forward(qkv, m_num_heads, true);
		Inferno::Tensor y = Inferno::flash_attention_bigdaddy_forward_my_version_check(qkv, m_num_heads, true);



		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y after flash_attention_bigdaddy_forward(qkv, m_num_heads, true)" << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << y << std::endl;
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

		
		// [B, T, C]
		y = W_out.forward(y);
		
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "y after W_out.forward(y)" << std::endl;
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
			this->register_module("block" + std::to_string(i), &transblks[i]);
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
		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << emb1 << std::endl;
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

		
		//Inferno::Tensor logits = linear1.forward(x);

		
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


// Top-k: zero out all logits outside the top k
Inferno::Tensor top_k_filter(Inferno::Tensor logits, int k) {
	// Copy to CPU for manipulation
	auto cpu_logits = logits.to(Inferno::Device::cpu());
	std::vector<float> vals = cpu_logits.to_vector<float>();

	// Find the k-th largest value
	std::vector<float> sorted = vals;
	std::nth_element(sorted.begin(), sorted.begin() + (sorted.size() - k), sorted.end());
	float threshold = sorted[sorted.size() - k];

	// Mask everything below threshold to -inf
	for (auto& v : vals)
		if (v < threshold) v = -std::numeric_limits<float>::infinity();

	Inferno::Tensor out = Inferno::Tensor(logits.dtype(), vals, logits.shape(), "topk");

	return out; 


	
}

// Multinomial sampling from a probability distribution
int32_t sample_multinomial(Inferno::Tensor probs) {
	auto cpu_probs = probs.to(Inferno::Device::cpu());
	std::vector<float> p = cpu_probs.to_vector<float>();

	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<int32_t> dist(p.begin(), p.end());
	return dist(gen);
}

int32_t argmax(Inferno::Tensor logits) {

	auto cpu = logits.to(Inferno::Device::cpu());
	std::vector<float> vals = cpu.to_vector<float>();

	return static_cast<int32_t>(
		std::distance(
			vals.begin(),
			std::max_element(vals.begin(), vals.end())
		)
		);
}

void print_topk_logits(
	Inferno::Tensor logits,
	InfernoTokenizer::BPETokenizer& tok,
	int k)
{
	auto cpu = logits.to(Inferno::Device::cpu());
	std::vector<float> vals = cpu.to_vector<float>();

	std::vector<int> idx(vals.size());

	for (int i = 0; i < static_cast<int>(vals.size()); i++) {
		idx[i] = i;
	}

	std::partial_sort(
		idx.begin(),
		idx.begin() + std::min(k, (int)idx.size()),
		idx.end(),
		[&](int a, int b) {
			return vals[a] > vals[b];
		}
	);

	std::cout << "\n===== TOP " << k << " LOGITS =====\n";

	for (int i = 0; i < std::min(k, (int)idx.size()); i++) {

		int token_id = idx[i];

		std::string piece;

		try {
			piece = tok.decode(static_cast<uint32_t>(token_id));
		}
		catch (...) {
			piece = "<decode error>";
		}

		std::cout
			<< "#" << i
			<< "  id=" << token_id
			<< "  logit=" << vals[token_id]
			<< "  piece=[" << piece << "]"
			<< "\n";
	}

	std::cout << "=========================\n";
}


int main(int argc, char* argv[]) {





	logger.Start("logs/inferno.txt");

	Inferno::Logger::SetLogger(&logger);
	Inferno::Logger::EnableLogging();

	logger.SetLevel(CoreLogger::Logger::LogLevel::LOGLEVEL_INFO);
	//logger.SetLevel(CoreLogger::Logger::LogLevel::LOGLEVEL_DEBUG);

	Inferno::RandomGenerator::initializeWithSeed(42);


	///////////////////////////////////////////////////
	//
	//  HyperParams
	//
	///////////////////////////////////////////////////

	//GPT 2
	size_t vocabulary_size = 60259;
	size_t context_size = 1024;
	size_t embedding_dim = 768;
	size_t numheads = 12;
	size_t numblocks = 12;
	

	InfernoTokenizer::BPETokenizer tok;
	tok.Initialize({ "data\\openwebtext_merges2.txt", "data\\openwebtext_vocab2.txt" });


	GPTModel model(vocabulary_size, context_size, embedding_dim, numheads, numblocks);

	//Inferno::Checkpoint ckpt = Inferno::Checkpoint::load("checkpoints\\largeckpt1000000.bin");
	//Inferno::Checkpoint ckpt = Inferno::Checkpoint::load("checkpoints\\largeckpt030000.bin");
	//Inferno::Checkpoint ckpt = Inferno::Checkpoint::load("checkpoints\\largeckpt500000.bin");
	Inferno::Checkpoint ckpt = Inferno::Checkpoint::load("checkpoints\\latest_checkpoint.bin");
	

	 

	model.load_state_dict(ckpt.model);


	

	model.to(device);

	int max_new_tokens = 8192;
	float temperature = 0.9f;
	int top_k = 40;




	std::string prompt = "The old lighthouse stood at the edge of the cliff, its paint long faded by decades of salt wind and driving rain. Marta had visited it every summer since she was a child, climbing the spiral staircase with her grandfather while he told stories about the ships that used to pass through these waters. Back then, the light still worked, sweeping its slow beam across the dark sea every ten seconds, a rhythm as familiar to her as her own heartbeat. Now the lamp was dark, decommissioned years ago when the shipping lanes moved further out, and the building had been left to the gulls and the wind.";

	std::vector<uint32_t> tokens = tok.encode(prompt);
	
	std::cout << prompt << std::flush;

	Inferno::NoGradGuard guard;

	for (int i =0; i<max_new_tokens; i++) {

		//t1.start();
		std::vector<uint32_t> ctx = tokens;


		//if our context of tokens is bigger than the context size, stip 
		if (ctx.size() > context_size)
			ctx = std::vector<uint32_t>(ctx.end() - context_size, ctx.end());

		Inferno::Tensor x = Inferno::Tensor(Inferno::DType::Int32, ctx, { 1, ctx.size() }, "prompt").to(device);

		
		
		Inferno::Tensor logits = model.forward(x);
		
		// Grab only the last token's logits -> [vocab_size]
		//Inferno::Tensor last_logits = logits.slice(1, ctx.size() - 1, ctx.size() - 1).squeeze();


		//bypass
		auto vals = logits.to(Inferno::Device::cpu()).to_vector<float>();

		size_t T = ctx.size();
		size_t V = vocabulary_size;
		size_t last_offset = (T - 1) * V;

		std::vector<float> last(
			vals.begin() + last_offset,
			vals.begin() + last_offset + V
		);

		Inferno::Tensor last_logits(Inferno::DType::Float32, last, { V }, "last_logits");


		

		// Apply temperature
		last_logits = last_logits / temperature;

		// Optional top-k filtering
		last_logits = top_k_filter(last_logits, top_k);

		// Softmax + sample
		
		//int32_t next_token = argmax(last_logits);
		Inferno::Tensor probs = Inferno::Softmax(last_logits, -1);
		int32_t next_token = sample_multinomial(probs); // see below

		if (next_token == 60257) {
			break;
		}

		tokens.push_back(next_token);

		// Decode and print incrementally
		std::string piece = tok.decode(static_cast<uint32_t>(next_token));
		std::cout << piece << std::flush;

	}



	std::cout << "\n\n\n\nEnd Program" << std::flush;



	return 0;

}



