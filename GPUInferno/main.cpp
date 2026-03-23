#include "inferno.h"
#include "tests/tests.h"

//Inferno::Device device = Inferno::Device::cpu();
Inferno::Device device = Inferno::Device::cuda(0);



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
	

	std::cout << "break";

	
	
	Inferno::Tensor input(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(784, -1.0f, 1.0f), { 784 }, "input", device);
	
	//Inferno::Tensor input(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(10, -1.0f, 1.0f), { 10 }, "input", device);	
	//Inferno::Tensor input(Inferno::DType::Float32, std::vector<float>{0.5}, {1}, "input", Inferno::Device::cuda(0));
	
	//Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {1, 0, 1, 0, 1, 0, 1, 0, 1, 0 }, { 10 }, "target", Inferno::Device::cpu());
	Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, { 10 }, "target", device);
	//Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {1, 0, 1, 0, 1, 0, 1, 0, 1, 0 }, { 10 }, "target", Inferno::Device::cuda(0));
	//Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, { 10 }, "target", Inferno::Device::cuda(0));
	//Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {1}, { 1 }, "target", Inferno::Device::cuda(0));

	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, { 5,6 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5 }, { 5 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, { 5,6 }, "b", Inferno::Device::cuda(0));


	//Inferno::Tensor a2 = make_view(a, a.shape(), a.strides(), 0, "a2");
	//a2.shape() = { 1,2,3,4,5 };


	std::vector<std::vector<float>> inputs(50000, std::vector<float>(784));
	std::vector<std::vector<float>> targets(50000, std::vector<float>(10));

	//GenerateSampleData(inputs, targets);

	LoadSampleData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", inputs, targets);
	



	std::vector<int> layers({ 784,512,256,10 });
	//std::vector<int> layers({ 100,80,40,10});
	//std::vector<int> layers({ 1,1,1,1 });

	MyModel model(layers);

	Inferno::MSELoss loss_fn;

	auto params = model.parameters();
	Inferno::OptimizerSGD optimizer(params, 0.001f);

	Inferno::Timer t1("matmul");

	int epochs = 60;
	int loopcount = 60000;
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


