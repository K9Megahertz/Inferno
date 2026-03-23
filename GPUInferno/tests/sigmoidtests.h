#pragma once

void RunSigmoidTests(Inferno::Device device) {


    TestStats stats;

    if (device == Inferno::Device::cuda(0))
        std::cout << "Using CUDA Device" << std::endl;
    if (device == Inferno::Device::cpu())
        std::cout << "Using CPU Device" << std::endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Int32 Tests
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 6.0f}, { 10 }, "a", device);        

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{0.017986f, 0.119203f, 0.268941f, 0.377541f, 0.500000f, 0.622459f, 0.731059f, 0.880797f, 0.982014f, 0.997527f }, { 10 }, "expected", device);
        
        Inferno::Sigmoid s = Inferno::Sigmoid(device, Inferno::DType::Float32);
        
        Inferno::Tensor actual = s.forward(a);

        ExpectTensorEq("Sigmoid Int32 simple", actual, expected, stats);
    }


    {
        Inferno::Tensor a(Inferno::DType::Float32,std::vector<float>{-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 6.0f},{ 10 },"a",device);

        Inferno::Tensor expected_data(Inferno::DType::Float32,std::vector<float>{-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 6.0f},{ 10 },"expected_data",device);

        Inferno::Tensor expected_grad(Inferno::DType::Float32,std::vector<float>{0.017662f, 0.104994f, 0.196612f, 0.235004f, 0.250000f,0.235004f, 0.196612f, 0.104994f, 0.017662f, 0.002466f},{ 10 },"expected_grad",device);

        Inferno::Sigmoid s(device, Inferno::DType::Float32);

        Inferno::Tensor actual = s.forward(a);
        actual.backward();

        ExpectTensorEq("Sigmoid backward data unchanged", a, expected_data, stats);

        auto agrad = GetImpl(a)->grad();
        if (!agrad) {
            stats.failed++;
            std::cout << "[FAIL] Sigmoid backward grad missing\n";
        }
        else {
            ExpectTensorEq("Sigmoid backward grad correct", *agrad, expected_grad, stats);
        }
    }
    

    
    std::cout << "Sigmoid tests: passed=" << stats.passed << " failed=" << stats.failed << "\n\n\n";
}

