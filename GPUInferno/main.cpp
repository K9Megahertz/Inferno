#include "inferno.h"
#include "tests/tests.h"

Inferno::Device device = Inferno::Device::cpu();
//Inferno::Device device = Inferno::Device::cuda(0);



// helper: read big-endian int
uint32_t read_uint32(std::ifstream& f) {
	uint32_t val = 0;
	f.read(reinterpret_cast<char*>(&val), 4);
	return ((val & 0xFF000000) >> 24) |
		((val & 0x00FF0000) >> 8) |
		((val & 0x0000FF00) << 8) |
		((val & 0x000000FF) << 24);
}

void LoadSampleData(
	const std::string& image_file,
	const std::string& label_file,
	std::vector<std::vector<float>>& inputs,
	std::vector<std::vector<float>>& targets)
{
	std::ifstream img(image_file, std::ios::binary);
	std::ifstream lbl(label_file, std::ios::binary);

	if (!img || !lbl) {
		std::cerr << "Failed to open MNIST files\n";
		exit(1);
	}

	// --- Read image header ---
	uint32_t img_magic = read_uint32(img);
	uint32_t num_images = read_uint32(img);
	uint32_t rows = read_uint32(img);
	uint32_t cols = read_uint32(img);

	if (img_magic != 2051) {
		std::cerr << "Invalid image file\n";
		exit(1);
	}

	// --- Read label header ---
	uint32_t lbl_magic = read_uint32(lbl);
	uint32_t num_labels = read_uint32(lbl);

	if (lbl_magic != 2049) {
		std::cerr << "Invalid label file\n";
		exit(1);
	}

	if (num_images != num_labels) {
		std::cerr << "Image/label count mismatch\n";
		exit(1);
	}

	const size_t image_size = rows * cols; // should be 784

	inputs.resize(num_images);
	targets.resize(num_images);

	// --- Read data ---
	for (size_t i = 0; i < num_images; ++i) {

		// --- INPUTS ---
		inputs[i].resize(image_size);

		for (size_t j = 0; j < image_size; ++j) {
			unsigned char pixel = 0;
			img.read(reinterpret_cast<char*>(&pixel), 1);

			// normalize to [0,1]
			inputs[i][j] = static_cast<float>(pixel) / 255.0f;
		}

		// --- TARGETS (one-hot) ---
		unsigned char label = 0;
		lbl.read(reinterpret_cast<char*>(&label), 1);

		targets[i].assign(10, 0.0f);
		targets[i][label] = 1.0f;
	}
}

void GenerateSampleData(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets) {

	const size_t num_samples = inputs.size();

	// Ensure targets matches inputs size
	targets.resize(num_samples);

	for (size_t i = 0; i < num_samples; ++i) {

		// --- INPUTS ---
		inputs[i].resize(784);
		for (size_t j = 0; j < 784; ++j) {
			inputs[i][j] = Inferno::RandomGenerator::generateRandomFloat(0.0f, 1.0f);
		}

		// --- TARGETS (one-hot) ---
		targets[i].assign(10, 0.0f);

		int cls = Inferno::RandomGenerator::generateRandomInt(0, 9);
		targets[i][cls] = 1.0f;
	}
}


/*class PositionalEncoding : public Inferno::Module {


public:

	PositionalEncoding(size_t context_size, size_t embed_dim) {




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


		pe = Inferno::Tensor(Inferno::DType::Float32, std::move(pe_data), { context_size, embed_dim }, "positional-encoding");



	}


	Inferno::Tensor forward(Inferno::Tensor& x) {
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
	MultiHeadAttention(size_t embed_dim, size_t num_heads)
		: m_embed_dim(embed_dim),
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
		std::vector<Inferno::Tensor> heads;



		for (int i = 0; i < m_num_heads; ++i) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Head: " + std::to_string(i));
			auto q = Wq_layers[i].forward(x);
			auto k = Wk_layers[i].forward(x);
			auto v = Wv_layers[i].forward(x);

			auto attn_scores = Inferno::matmul(q, k.transpose(-1, -2)) / std::sqrt(static_cast<float>(m_head_dim));

			auto attn_probs = Inferno::softmax(attn_scores, -1);
			auto head = Inferno::matmul(attn_probs, v);

			heads.push_back(head);
		}

		// concatenate heads along embedding dim
		Inferno::Tensor concat = Inferno::concat(heads, -1);

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
		auto normed = layernorm1.forward(x);
		auto attn_out = attn.forward(normed);
		x = x + attn_out;
		auto normed2 = layernorm2.forward(x);
		auto ff = feedforward2.forward(Inferno::gelu(feedforward1.forward(normed2)));
		return x + ff;
	}

private:
	MultiHeadAttention attn;
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


		//Get embedding vectors
		Inferno::Tensor x = emb1.forward(input);

		//Add positional encoding
		x = pos_enc.forward(x);

		// pump it through the Transformer blocks
		for (int blk_idx = 0; blk_idx < transblks.size(); blk_idx++) {
			//for (TransformerBlock tblk : transblks) {
			Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Block: " + std::to_string(blk_idx));
			x = transblks[blk_idx].forward(x);
		}
		//Layer norm
		//std::cout << "LayerNorm" << std::endl;
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "LayerNorm");
		x = layernorm1.forward(x);
		std::cout << x << std::endl;
		//Linear
		Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Linear");
		x = linear1.forward(x);
		std::cout << x << std::endl;
		//Softmax
		//Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "SoftMax");
		//x = Inferno::softmax(x);
		std::cout << x << std::endl;

		std::cout << "next logits slice" << std::endl;
		Inferno::Tensor next_logits = x.slice(-2, m_context_size - 1, m_context_size - 1);
		std::cout << next_logits << std::endl;

		//Inferno::Tensor out = Inferno::softmax(next_logits);

		return next_logits;
	}

	Inferno::Embedding emb1;
	PositionalEncoding pos_enc;
	std::vector<TransformerBlock> transblks;
	Inferno::Linear linear1;
	Inferno::LayerNorm layernorm1;

	size_t m_context_size;
	size_t m_embed_dim;
	size_t m_vocab_size;




};*/


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Class MyModel
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MyModel : public Inferno::Module {

public:


	MyModel(std::vector<int>&layers) : fc1(layers[0], layers[1], device),
									   fc2(layers[1], layers[2], device),
									   fc3(layers[2], layers[3], device) {


		//TODO: add these to the constructors
		//act1 = Inferno::Sigmoid(Inferno::Device::cuda(0));
		//act2 = Inferno::Sigmoid(Inferno::Device::cuda(0));
		//act3 = Inferno::Sigmoid(Inferno::Device::cuda(0));

		act1 = Inferno::Sigmoid(device);
		act2 = Inferno::Sigmoid(device);
		act3 = Inferno::Sigmoid(device);
		this->register_module(&fc1);
		this->register_module(&fc2);
		this->register_module(&fc3);
		this->register_module(&act1);
		this->register_module(&act2);
		this->register_module(&act3);

	}

	Inferno::Tensor forward(Inferno::Tensor& input) {

		Inferno::Tensor out = fc1.forward(input);		
		out = act1.forward(out);
		out = fc2.forward(out);
		out = act2.forward(out);
		out = fc3.forward(out);
		out = act3.forward(out);
		return out;
	}


	Inferno::Linear fc1;
	Inferno::Linear fc2;
	Inferno::Linear fc3;

	Inferno::Sigmoid act1;
	Inferno::Sigmoid act2;
	Inferno::Sigmoid act3;





};


int main() {



	//Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_ERROR);
	//Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_DEBUG);
	Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_INFO);
	Logger::Start("logs/applicationlog.txt");

	Inferno::RandomGenerator::initializeWithSeed(42);

	

	//RunTests();
	

		
	Inferno::Tensor input = Inferno::Tensor::randn(Inferno::DType::Float32, { 784 }, "input", device);	
	Inferno::Tensor target(Inferno::DType::Float32, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, { 10 }, "target", device);
	Inferno::Tensor tokens(Inferno::DType::Int32, {42, 13, 1, 0, 99, 34, 23, 78, 1, 25 }, { 10 }, "tokens", device);
	Inferno::Tensor normfwd(Inferno::DType::Float32, std::vector<float> {0.2, 0.1, 0.3, 0.5, 0.1, 0.1}, { 2,3 }, "normfwd", device);



	Inferno::Tensor tensor1(Inferno::DType::Float32, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, { 2,3,4 }, "tensor1", device);
	Inferno::Tensor tensor2(Inferno::DType::Float32, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1 }, { 2,3,4 }, "tensor2", device);

	std::vector<Inferno::Tensor> list = { tensor1, tensor2 };

	std::cout << tensor1 << std::endl;
	std::cout << tensor2 << std::endl;

	Inferno::Tensor concat = Inferno::concat(list, 2);

	std::cout << concat << std::endl;

	//Inferno::Embedding e = Inferno::Embedding(10, 10, device);

	//Inferno::Tensor blah = e(tokens);


	
	

	std::vector<std::vector<float>> inputs(50000, { 784 });
	std::vector<std::vector<float>> targets(50000, { 10 });

	//GenerateSampleData(inputs, targets);

	//LoadSampleData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", inputs, targets);

	/*Inferno::LayerNorm layernorm1 = Inferno::LayerNorm(3);

	
	Inferno::Tensor l = layernorm1(normfwd);
	
	std::cout << "Printing Layernorm after forward\n\n";
	std::cout << l << std::endl;
	
	l.backward();	

	std::cout << "Printing Layernorm after backward\n\n";
	std::cout << layernorm1 << std::endl;

	*/
	
	std::vector<int> layers({ 784,576,256,10 });
	//std::vector<int> layers({ 100,80,40,10});
	//std::vector<int> layers({ 1,1,1,1 });

	MyModel model(layers);

	Inferno::MSELoss loss_fn;

	auto params = model.parameters();
	Inferno::OptimizerSGD optimizer(params, 0.001f);

	Inferno::Timer t1("matmul");

	int epochs = 1;
	int loopcount = 1;
	for (int e = 0; e < epochs; e++) {
		for (int i = 0; i < loopcount; i++) {

			t1.start();

			cudaDeviceSynchronize();


			Inferno::Tensor in(Inferno::DType::Float32, inputs[i], { 784 }, "input", device);
			Inferno::Tensor prediction = model.forward(in);

			Inferno::Tensor targ(Inferno::DType::Float32, targets[i], { 10 }, "target", device);
			Inferno::Tensor loss = loss_fn(prediction, targ);

			//std::cout << " **** Prediction ****" << std::endl;
			//std::cout << prediction.to(Inferno::Device::cpu()) << std::endl;


			//std::cout << " **** Input Grad before backward ****" << std::endl;
			//std::cout << input.to(Inferno::Device::cpu()) << std::endl;


			loss.backward();

			//std::cout << input.to(Inferno::Device::cpu()) << std::endl;

			//std::cout << " **** Input Grad after backward ****" << std::endl;
			//std::cout << input.to(Inferno::Device::cpu()) << std::endl;

			//std::cout << model.fc1.m_weights << std::endl;
			//std::cout << model.fc1.m_biases << std::endl;
			//std::cout << model.fc2.m_weights << std::endl;
			//std::cout << model.fc2.m_biases << std::endl;
			//std::cout << model.fc3.m_weights << std::endl;
			//std::cout << model.fc3.m_biases << std::endl;


			optimizer.step();
			optimizer.zero_grad();


			t1.stop();

			Inferno::Tensor lossp = loss.to(Inferno::Device::cpu());

			std::cout << std::fixed << "Epoch: " << e << " Iter: " << i << "  total took: " << std::setprecision(3) << t1.elapsed_ms() << " ms  Loss: " << std::setprecision(8) << lossp.item<float>() << std::endl;
			//std::cout << loss << std::endl;
			//std::cout << prediction.to(Inferno::Device::cpu()) << std::endl;

			//if (i % 1000 == 0 || i < 10) {
	//			std::cout << prediction.to(Inferno::Device::cpu()) << std::endl;
				//std::cout << target.to(Inferno::Device::cpu()) << std::endl;
			//}

			if (i >= loopcount - 1) {
				std::cout << prediction.to(Inferno::Device::cpu()) << std::endl;
			}
		}
	}
	

	Inferno::NodeTracker::dumpIDs();


	return 0;

}


