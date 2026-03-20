#include "inferno.h"





int main() {



	//Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_ERROR);
	//Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_DEBUG);
	Logger::SetLogLevel(Logger::LogLevel::LOGLEVEL_INFO);
	Logger::Start("logs/applicationlog.txt");

	Inferno::RandomGenerator::initializeWithSeed(42);

	

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
	
	Inferno::Tensor a(Inferno::DType::Float32, Inferno::RandomGenerator::generateRandomFloatVector(784, -1.0f, 1.0f), { 784 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, { 5,6 }, "b", Inferno::Device::cpu());

	//Inferno::Tensor a(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5 }, { 5 }, "a", Inferno::Device::cuda(0));
	//Inferno::Tensor b(Inferno::DType::Float32, std::vector<float> {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, { 5,6 }, "b", Inferno::Device::cuda(0));


	//Inferno::Tensor a2 = make_view(a, a.shape(), a.strides(), 0, "a2");
	//a2.shape() = { 1,2,3,4,5 };

	Inferno::Linear lin1 = Inferno::Linear(784, 512, Inferno::Device::cuda(0));
	Inferno::Linear lin2 = Inferno::Linear(512, 256, Inferno::Device::cuda(0));
	Inferno::Linear lin3 = Inferno::Linear(256, 64, Inferno::Device::cuda(0));
	Inferno::Linear lin4 = Inferno::Linear(64, 10, Inferno::Device::cuda(0));

	Inferno::Timer t1("matmul");
	Inferno::Timer t2("matmul");
	t1.start();
	for (int i = 0; i < 1; i++) {

		t2.start();

		Inferno::Tensor b = lin1.forward(a);
		b = lin2.forward(b);
		b = lin3.forward(b);
		b = lin4.forward(b);


		//Inferno::Tensor g = Inferno::matmul(a,b);	
		cudaDeviceSynchronize();

		t2.stop();

		b.backward();

		std::cout << "forward took: " << t2.elapsed_ms() << " ms\n";

	}

	t1.stop();
	std::cout << "total took: " << t1.elapsed_ms() << " ms\n";
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