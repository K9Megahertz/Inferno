#include "inferno.h"
#include "tests/tests.h"




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


	MyModel(std::vector<int>& layers) : fc1(layers[0], layers[1], Inferno::Device::cuda(0)),
										fc2(layers[1], layers[2], Inferno::Device::cuda(0)),
										fc3(layers[2], layers[3], Inferno::Device::cuda(0)) {

	/*MyModel(std::vector<int>&layers) : fc1(layers[0], layers[1], Inferno::Device::cpu()),
									   fc2(layers[1], layers[2], Inferno::Device::cpu()),
										fc3(layers[2], layers[3], Inferno::Device::cpu()) {*/


		//TODO: add these to the constructors
		act1 = Inferno::Sigmoid(Inferno::Device::cuda(0));
		act2 = Inferno::Sigmoid(Inferno::Device::cuda(0));
		act3 = Inferno::Sigmoid(Inferno::Device::cuda(0));

		//act1 = Inferno::Sigmoid(Inferno::Device::cpu());
		//act2 = Inferno::Sigmoid(Inferno::Device::cpu());
		//act3 = Inferno::Sigmoid(Inferno::Device::cpu());
		this->register_module(&fc1);
		this->register_module(&fc2);
		this->register_module(&fc3);
		this->register_module(&act1);
		this->register_module(&act2);
		this->register_module(&act3);

	}

	Inferno::Tensor forward(Inferno::Tensor& input) {

		Inferno::Tensor out = fc1.forward(input);
		//std::cout << out.to(Inferno::Device::cpu());
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

	

	RunTests();
	



	//Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(10000000, -1.0f, 1.0f), { 10000000 }, "input", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(10000000, -1.0f, 1.0f), { 10000000 }, "input", Inferno::Device::cpu());
	//Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(1000000000, -1.0f, 1.0f), { 1000000000 }, "input", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(1000000000, -1.0f, 1.0f), { 1000000000 }, "input", Inferno::Device::cuda(0));
	//Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>({ 1,2,3,4 }), { 2,2 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>({ 4,3,2,1 }), { 2,2 }, "b", Inferno::Device::cuda(0));	
	/*Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "a", Inferno::Device::cpu());
	Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "b", Inferno::Device::cpu());
	Inferno::Tensor c(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "c", Inferno::Device::cpu());
	Inferno::Tensor d(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "d", Inferno::Device::cpu());
	*/
	/*Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "a", Inferno::Device::cuda(0));
	Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "b", Inferno::Device::cuda(0));
	Inferno::Tensor c(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "c", Inferno::Device::cuda(0));
	Inferno::Tensor d(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(20, -1.0f, 1.0f), { 4,5 }, "d", Inferno::Device::cuda(0));
	*/


	//Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(4194304, -1.0f, 1.0f), { 2048,2048 }, "a", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(4194304, -1.0f, 1.0f), { 2048,2048 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(67108864, -1.0f, 1.0f), { 8192,8192 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(67108864, -1.0f, 1.0f), { 8192,8192 }, "b", Inferno::Device::cuda(0));
	
	//Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(67108864, -1.0f, 1.0f), { 8192,8192 }, "a", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(67108864, -1.0f, 1.0f), { 8192,8192 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(15, -1.0f, 1.0f), {   3,5 }, "a", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(60, -1.0f, 1.0f), { 3,5,4 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(15, -1.0f, 1.0f), { 3,5 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(60, -1.0f, 1.0f), { 3,5,4 }, "b", Inferno::Device::cuda(0));

	//Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{ 1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8 }, { 2,2,3,4 }, "a", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{ 9,8,7,6,5,4,3,2,1,0,9,8,7,6,5,4,3,2,1,0 }, { 4,5 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{ 1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8 }, { 2,2,3,4 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{ 9,8,7,6,5,4,3,2,1,0,9,8,7,6,5,4,3,2,1,0 }, { 4,5 }, "b", Inferno::Device::cuda(0));

	//Inferno::Tensor a(Inferno::DType::Int32, std::vector<int> {1,2,3,1,2,3,1,2,3}, { 3,3 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Int32, std::vector<int> {1,2,3,1,2,3,1,2,3}, { 3,3 }, "b", Inferno::Device::cuda(0));

	//Inferno::Tensor a(Inferno::DType::Int32, std::vector<int> {1, 2, 3, 1, 2, 3, 1, 2, 3}, { 3,3 }, "a", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Int32, std::vector<int> {1, 2, 3, 1, 2, 3, 1, 2, 3}, { 3,3 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Int32, std::vector<int> {6}, { 1 }, "a", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Int32, std::vector<int> {3}, { 1 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 1, 2, 3, 1, 2, 3}, { 3,3 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 1, 2, 3, 1, 2, 3}, { 3,3 }, "b", Inferno::Device::cuda(0));

	//Inferno::Tensor a(Inferno::DType::Int32, std::vector<int> {6,2,1}, { 3 }, "a", Inferno::Device::cpu());
	//Inferno::Tensor b(Inferno::DType::Int32, std::vector<int> {3,2,1}, { 3,1 }, "b", Inferno::Device::cpu());



  	//Inferno::Tensor a(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5 }, { 5 }, "a", Inferno::Device::cuda(0));
	
	//Inferno::Tensor input(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(784, -1.0f, 1.0f), { 784 }, "input", Inferno::Device::cpu());
	Inferno::Tensor input(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(78400, -1.0f, 1.0f), { 784 }, "input", Inferno::Device::cuda(0));


	//Inferno::Tensor input(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(10, -1.0f, 1.0f), { 10 }, "input", Inferno::Device::cpu());
	//Inferno::Tensor input(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(10, -1.0f, 1.0f), { 10 }, "input", Inferno::Device::cuda(0));
	//Inferno::Tensor input(Inferno::DType::Float32, std::vector<float>{0.5}, {1}, "input", Inferno::Device::cuda(0));
	
	//Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {1, 0, 1, 0, 1, 0, 1, 0, 1, 0 }, { 10 }, "target", Inferno::Device::cpu());
	Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {1, 0, 1, 0, 1, 0, 1, 0, 1, 0 }, { 10 }, "target", Inferno::Device::cuda(0));
	//Inferno::Tensor target(Inferno::DType::Float32, std::vector<float> {1}, { 1 }, "target", Inferno::Device::cuda(0));

	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, { 5,6 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5 }, { 5 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, { 5,6 }, "b", Inferno::Device::cuda(0));


	//Inferno::Tensor a2 = make_view(a, a.shape(), a.strides(), 0, "a2");
	//a2.shape() = { 1,2,3,4,5 };


	std::vector<int> layers({ 784,512,256,10 });
	//std::vector<int> layers({ 10,10,10,10 });
	//std::vector<int> layers({ 1,1,1,1 });

	MyModel model(layers);

	Inferno::MSELoss loss_fn;

	auto params = model.parameters();
	Inferno::OptimizerSGD optimizer(params, 0.001f);

	Inferno::Timer t1("matmul");
	
	for (int i = 0; i < 30000; i++) {
		
		t1.start();

		cudaDeviceSynchronize();

		Inferno::Tensor prediction = model.forward(input);
		//prediction.backward();

		Inferno::Tensor loss = loss_fn(prediction, target);

		//std::cout << " **** Prediction ****" << std::endl;
		//std::cout << prediction.to(Inferno::Device::cpu()) << std::endl;


		//std::cout << " **** Input Grad before backward ****" << std::endl;
		//std::cout << input.to(Inferno::Device::cpu()) << std::endl;
		

		loss.backward();

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

		std::cout << std::fixed << "Iter: "  << i << "  total took: " << std::setprecision(3) << t1.elapsed_ms() << " ms  Loss: " << std::setprecision(8) << lossp.item<float>()	<< std::endl;
		//std::cout << loss << std::endl;

		if (i >= 29999) {
			std::cout << prediction.to(Inferno::Device::cpu()) << std::endl;
		}
	}

	
	//std::cout << a << std::endl;
	//std::cout << b << std::endl;

	//Inferno::Tensor e = a + b;
	//Inferno::Tensor f = c + d;


	//std::cout << g << std::endl;


	//Inferno::Tensor d = c.to(Inferno::Device::cpu());
	//b.backward();





	//Inferno::Tensor ac = c.to(Inferno::Device::cpu());
	//std::cout << ac << std::endl;
	//Inferno::Tensor ad = d.to(Inferno::Device::cpu());
	//std::cout << ad << std::endl;

	
	

	Inferno::NodeTracker::dumpIDs();


		return 0;

}