#include <inferno/inferno.h>
#include "logger.h"
#include "timer.h"

//Inferno::Device device = Inferno::Device::cpu();
Inferno::Device device = Inferno::Device::cuda(0);





class PositionalEncoding : public Inferno::Module {


public:

	PositionalEncoding(size_t context_size, size_t embed_dim) {

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Positional Encoding - Initializing buffers");
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

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Positional Encoding - Creating tensor");
		pe = Inferno::Tensor(Inferno::DType::Float32, std::move(pe_data), { context_size, embed_dim }, "positional-encoding");

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Positional Encoding - Register Buffer");
		register_buffer(pe);

	}


	Inferno::Tensor forward(Inferno::Tensor& x) {
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Positional Encoding forward");
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


class MultiHeadAttention : public Inferno::Module {
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

		// final output projection after concatenation		
		register_module(&W_out);

	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Multihead Attention forward");
		std::vector<Inferno::Tensor> heads;


		for (int i = 0; i < m_num_heads; ++i) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Head: " + std::to_string(i));

			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - Q forward");
			auto q = Wq_layers[i].forward(x);

			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - K forward");
			auto k = Wk_layers[i].forward(x);

			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - V forward");
			auto v = Wv_layers[i].forward(x);

			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - Attn scores forward - transpose -> matmul -> Divide");
			auto attn_scores = Inferno::matmul(q, k.transpose(-1, -2)) / std::sqrt(static_cast<float>(m_head_dim));

			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - softmax forward");
			auto attn_probs = Inferno::Softmax(attn_scores, -1);

			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - matmul forward - Attn x V");
			auto head = Inferno::matmul(attn_probs, v);

			heads.push_back(head);
		}

		// concatenate heads along embedding dim
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - concat forward");
		Inferno::Tensor concat = Inferno::concat(heads, -1);

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "MHA - Linear forward");
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
};



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


		register_module(&Wqkv_layer);


		// final output projection after concatenation		
		register_module(&W_out);

	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Multihead Attention forward");


		auto shape = x.shape();

		if (shape.size() != 3) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "MultiHeadAttentionFast expects [B, T, C]");
			exit(1);
		}

		size_t B = shape[0];
		size_t T = shape[1];
		size_t C = shape[2];


		//std::cout << Wqkv_layer << std::endl;


		auto qkv = Wqkv_layer.forward(x);


		Inferno::Tensor q = qkv.slice(2, 0, m_embed_dim - 1);                  // [B, T, C]
		Inferno::Tensor k = qkv.slice(2, m_embed_dim, 2 * m_embed_dim - 1);    // [B, T, C]
		Inferno::Tensor v = qkv.slice(2, 2 * m_embed_dim, 3 * m_embed_dim - 1);// [B, T, C]

		q = q.contiguous();
		k = k.contiguous();
		v = v.contiguous();

		q = q.reshape({ B, T, m_num_heads, m_head_dim });           // [B, T, H, D]
		k = k.reshape({ B, T, m_num_heads, m_head_dim });
		v = v.reshape({ B, T, m_num_heads, m_head_dim });

		q = q.transpose(1, 2);                                    // [B, H, T, D]
		k = k.transpose(1, 2);                                    // [B, H, T, D]
		v = v.transpose(1, 2);                                    // [B, H, T, D]



		Inferno::Tensor kt = k.transpose(-1, -2);                          // [B, H, D, T]

		Inferno::Tensor scores = matmul(q, kt);                             // [B, H, T, T]
		//std::cout << scores << std::endl;
		float scale = 1.0f / std::sqrt((float)m_head_dim);
		scores = scores * scale;
		//std::cout << scores << std::endl;


		Inferno::Tensor ones(Inferno::DType::Int32, std::vector<int>(T * T, 1.0f), { 1, 1, T, T }, "causal_mask_ones", scores.device());
		Inferno::Tensor mask = Inferno::triu(ones, 1);

		// mask out disallowed positions before softmax
		scores = Inferno::masked_fill(scores, mask, -1e9f);
		//std::cout << scores << std::endl;


		Inferno::Tensor attn = Inferno::Softmax(scores, -1);                         // [B, H, T, T]
		//std::cout << attn << std::endl;

		Inferno::Tensor y = matmul(attn, v);                                // [B, H, T, D]

		//std::cout << "attn @ V" << std::endl;
		//std::cout << y << std::endl;

		y = y.transpose(1, 2);                                    // [B, T, H, D]
		y = y.contiguous();
		y = y.reshape({ B, T, m_embed_dim });                       // [B, T, C]

		//std::cout << "after transpose/contig/reshape" << std::endl;
		//std::cout << y << std::endl;

		//std::cout << "W_out" << std::endl;
		//std::cout << W_out << std::endl;

		y = W_out.forward(y);                                   // [B, T, C]

		//std::cout << y << std::endl;

		return y;
	}

private:
	size_t m_embed_dim;
	size_t m_num_heads;
	size_t m_head_dim;

	Inferno::Linear Wqkv_layer;
	Inferno::Linear W_out;
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
		register_module(&attn);
		register_module(&layernorm1);
		register_module(&layernorm2);
		register_module(&feedforward1);
		register_module(&feedforward2);
	}

	Inferno::Tensor forward(Inferno::Tensor& x) override {
		//std::cout << feedforward1 << std::endl;
		//std::cout << feedforward2 << std::endl;
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Transformer Block forward");
		//std::cout << x << std::endl;
		Inferno::Tensor normed = layernorm1.forward(x);
		//std::cout << normed << std::endl;

		Inferno::Tensor attn_out = attn.forward(normed);
		//std::cout << "attn_out" << std::endl;
		//std::cout << attn_out << std::endl;

		x = x + attn_out;
		//std::cout << "after x = x + attn_out" << std::endl;
		//std::cout << x << std::endl;

		Inferno::Tensor normed2 = layernorm2.forward(x);
		//std::cout << "normed2" << std::endl;
		//std::cout << normed2 << std::endl;

		//std::cout << "feedforward1 weights and bias" << std::endl;
		//std::cout << feedforward1 << std::endl;
		
		Inferno::Tensor n = feedforward1.forward(normed2);
		//std::cout << "after feedfoward1" << std::endl;
		//std::cout << n << std::endl;

		n = Inferno::gelu(n);
		//std::cout << "after gelu" << std::endl;
		//std::cout << n << std::endl;

		//std::cout << "feedforward2 weights and bias" << std::endl;
		//std::cout << feedforward2 << std::endl;

		Inferno::Tensor ff = feedforward2.forward(n);
		//std::cout << "after feedfoward2" << std::endl;
		//std::cout << ff << std::endl;

		Inferno::Tensor out = x + ff;
		//std::cout << "after out = x + ff" << std::endl;
		//std::cout << out << std::endl;

		return out;
	}

private:
	MultiHeadAttentionFast attn;
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
		this->register_module(&emb1);
		this->register_module(&pos_enc);

		transblks.reserve(nblocks);
		for (size_t i = 0; i < nblocks; i++) {
			transblks.emplace_back(embed_dim, nheads);  // constructs Head(i)
			this->register_module(&transblks[i]);
		}

		this->register_module(&linear1);
		this->register_module(&layernorm1);


	}

	Inferno::Tensor forward(Inferno::Tensor& input) {

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "\n");

		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "GPTModel forward");
		//Get embedding vectors
		//std::cout << emb1 << std::endl;
		Inferno::Tensor x = emb1.forward(input);
		//std::cout << x << std::endl;

		//Add positional encoding
		x = pos_enc.forward(x);
		//std::cout << x << std::endl;


		// pump it through the Transformer blocks
		for (int blk_idx = 0; blk_idx < transblks.size(); blk_idx++) {
			//for (TransformerBlock tblk : transblks) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Block: " + std::to_string(blk_idx));
			x = transblks[blk_idx].forward(x);
		}
		//Layer norm
		//std::cout << "LayerNorm" << std::endl;
		//Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "LayerNorm");
		x = layernorm1.forward(x);
		//std::cout << x << std::endl;
		//Linear
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Linear");
		Inferno::Tensor logits = linear1.forward(x);
		//std::cout << x << std::endl;		

		

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



	//Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_ERROR);
	//Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_DEBUG);
	Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_INFO);
	Logger::Start("logs/applicationlog.txt");

	Inferno::RandomGenerator::initializeWithSeed(42);

	
	
	//RunTests();




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
	size_t vocabulary_size = 50257;
	size_t context_size = 1024;
	size_t embedding_dim = 768;
	size_t numheads = 16;
	size_t numblocks = 16;



	std::vector<int> data(context_size, 0);
	data[0] = 1;
	Inferno::Tensor target(Inferno::DType::Int32, data, { 1, context_size }, "target", device);
	Inferno::Tensor tokens = Inferno::Tensor(Inferno::DType::Int32, Inferno::RandomGenerator::generateRandomIntVector(context_size, 0, vocabulary_size - 1), { 1,context_size }, "tokens", device);


	//Inferno::Tensor tokens(Inferno::DType::Int32, { 42, 13, 1, 0, 99, 34, 23, 78, 1, 25, 22, 45, 02, 13, 67, 88 }, { 16 }, "tokens", device);
	//Inferno::Tensor input = Inferno::Tensor(Inferno::DType::Float32, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 }, { 10 }, "input", device);


	//for mnist test
	//std::vector<size_t> layers({ 784,512,256,10 });
	//Inferno::Tensor input = Inferno::Tensor(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(layers[0],-0.5f,0.5f), { layers[0] }, "input", device);
	//Inferno::Tensor target = Inferno::Tensor(Inferno::DType::Float32, { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 10 }, "target", device);



	//MyModel model(layers);
	GPTModel model(vocabulary_size, context_size, embedding_dim, numheads, numblocks);

	//TestModel model(layers);

	model.to(device);

	Inferno::CrossEntropyLoss loss_fn;

	auto params = model.parameters();
	Inferno::OptimizerSGD optimizer(params, 0.001f);

	Inferno::Timer t1("matmul");

	std::cout << tokens << std::endl;

	int epochs = 1;
	int loopcount = 10000;
	for (int e = 0; e < epochs; e++) {
		for (int i = 0; i < loopcount; i++) {

			t1.start();			

			Inferno::Tensor logits = model.forward(tokens);
			//Inferno::Tensor prediction = model.forward(input);	

			//for intference
			//Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "next logits slice");
			//Inferno::Tensor next_logits = x.slice(-2, m_context_size - 1, m_context_size - 1);
			//Inferno::Tensor next_logits = Inferno::select(x, -2, m_context_size - 1); // {B,V}
			//std::cout << next_logits << std::endl;		

			//std::cout << prediction << std::endl;
			//std::cout << target << std::endl;

			Inferno::Tensor loss = loss_fn(logits, target);

			//std::cout << loss << std::endl;

			loss.backward();


			optimizer.step();
			optimizer.zero_grad();


			t1.stop();



			Inferno::Tensor lossp = loss.to(Inferno::Device::cpu());

			std::cout
				<< std::fixed
				<< "Epoch: " << e
				<< " Iter: " << i
				<< "  total took: "
				<< std::setw(7) << std::setfill('0') << std::setprecision(3) << t1.elapsed_ms()
				<< " ms  Loss: "
				<< std::setw(13) << std::setfill('0') << std::setprecision(9) << lossp.item<float>()
				<< std::endl;			
		}
	}

	//std::cout << model.fc1.m_weights << std::endl;
	//std::cout << model.fc1.m_biases << std::endl;

	//std::cout << input << std::endl;
	//std::cout << tokens << std::endl;
	//std::cout << target << std::endl;


	


	return 0;

}


