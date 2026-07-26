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
bool printtrainingtokens = false;


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

		if (laptimingenabled) t1.lap("Start");
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




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function save_checkpoint
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void save_checkpoint(Inferno::Module model,Inferno::OptimizerAdamW optimizer, size_t step, size_t total_steps) {
	logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Writing Checkpoint" << std::endl;
	Inferno::Checkpoint chkpt;
	chkpt.meta = Inferno::TrainingMetadata(step, total_steps, 0, 1);
	chkpt.model = model.state_dict();
	chkpt.optimizer = optimizer.state_dict();

	std::ostringstream oss;


	//TODO: move this out of save, save should not have this responsibility
	//always save a latest checkpoint
	chkpt.save("checkpoints\\latest_checkpoint.bin");


	//write out one with step number
	if (step % 1000 == 0) {
		oss << "checkpoints\\checkpoint_"
			<< std::setw(8) << std::setfill('0') << step
			<< ".bin";
		chkpt.save(oss.str());
	}
	
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function main
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
	

	InfernoTokenizer::BPETokenizer tok;
	tok.Initialize({ "data\\openwebtext_merges_gold.txt", "data\\openwebtext_vocab_gold.txt" });

	DataLoader2 loader("data\\openwebtext_clean_gold.tokens", batch_size, context_size);

	int checkpoint_interval = 25;
	int total_steps = 32000;
	int micro_steps = 1;// 32;
	int step = 0;
	float lowestloss = INFINITY;

	RunningAverage avg(200);
	
	GPTModel model(vocabulary_size, context_size, embedding_dim, numheads, numblocks);

	std::optional<Inferno::Checkpoint> ckpt;

	if (resume) {
		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "Resuming training from: " << ckpt_path << std::endl;
		ckpt = Inferno::Checkpoint::load(ckpt_path);

		step = ckpt->meta.step;
		total_steps = ckpt->meta.total_steps;

		model.load_state_dict(ckpt->model);
	}
	
	model.to(device);

	auto params = model.parameters();
	
	Inferno::OptimizerAdamW optimizer(model.parameters(), total_steps, 1.5e-4f, 0.9f, 0.95f, 1e-8f, 0.1f);

	if (resume) {
		optimizer.load_state_dict(ckpt->optimizer);
	}	
	

	Inferno::CrossEntropyLoss loss_fn;

	//training loop
	for (; step < total_steps; step++) {

		float accum_loss = 0.0f;

		for (int micro = 0; micro < micro_steps; micro++) {


			t1.start();

			if (laptimingenabled) t1.lap("Start Load pair");
			std::pair<Inferno::Tensor, Inferno::Tensor> pair = loader.next_batch();

			Inferno::Tensor x = pair.first;
			Inferno::Tensor y = pair.second;


			if (printtrainingtokens) {
				auto blahx = x[0].to_vector<int>();
				auto blahy = y[0].to_vector<int>();

				std::string sx = tok.decode(blahx);
				std::string sy = tok.decode(blahy);

				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "**************************** Tensor X ****************************" << std::endl;
				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << sx.substr(0, 32) << std::endl;
				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;
				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;


				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << "**************************** Tensor Y ****************************" << std::endl;
				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << sy.substr(0, 32) << std::endl;
				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;
				logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO) << std::endl;
			}

			if (laptimingenabled) t1.lap("Start to device");
			x = x.to(device);
			y = y.to(device);
			
			if (laptimingenabled) t1.lap("Start Forward");
			Inferno::Tensor logits = model.forward(x);

			if (laptimingenabled) t1.lap("Start loss");
			Inferno::Tensor loss = loss_fn(logits, y);
			if (laptimingenabled) t1.lap("loss");

			loss = loss / micro_steps;

			accum_loss += loss.to(Inferno::Device::cpu()).item<float>();
			

			std::cout << ".";

			if (laptimingenabled) t1.lap("Start backward");
			loss.backward();
			if (laptimingenabled) t1.lap("backward");


		}
		//std::cout << std::endl;

		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << "Loss" << std::endl;
		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << accum_loss << std::endl;
		//logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_DEBUG) << std::endl;

				
		if (laptimingenabled) t1.lap("Optimizer start");
		optimizer.step();
		if (laptimingenabled) t1.lap("Optimizer step");
		optimizer.zero_grad();
		if (laptimingenabled) t1.lap("Zero Grad");

		t1.stop();

		
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

		
		if (accum_loss < lowestloss)
			lowestloss = accum_loss ;
		
		avg.add(accum_loss);

		logger.Append(Inferno::Logger::LogLevel::LOGLEVEL_INFO)
			<< std::fixed
			<< "Iter: " << step
			<< " | Percent complete: "
			<< std::setw(7) << std::setfill(' ') << std::setprecision(3) << static_cast<float>(step) / static_cast<float>(total_steps) * 100.0f
			<< "% | total took: "
			<< std::setw(7) << std::setfill('0') << std::setprecision(3) << t1.elapsed_ms()
			<< " ms | LR: "
			<< std::setw(9) << std::setfill('0') << std::setprecision(8) << optimizer.getLR()
			<< " | Loss: "
			//<< std::setw(9) << std::setfill('0') << std::setprecision(5) << lossp.item<float>()
			<< std::setw(9) << std::setfill('0') << std::setprecision(5) << accum_loss
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


