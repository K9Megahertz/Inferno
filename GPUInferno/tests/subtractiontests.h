#pragma once
void RunSubtractionTests(Inferno::Device device) {


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
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 3, 1, 3, 1, 3, 1, 3, 1, 3}, { 10 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}, { 10 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, { 10 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Int32 simple", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1}, { 5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{0, 0, 0, 0, 0, 1, -1, 1, -1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Int32 broadcast A", actual, expected, stats);

    }

    {
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1}, { 5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{0, 0, 0, 0, 0, -1, 1, -1, 1, -1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Int32 broadcast B", actual, expected, stats);

    }

    {
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0}, { 2,1,2 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0}, { 2,2,1 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{0, -1, 1, 0, 0, -1, 1, 0}, { 2,2,2 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Int32 broadcast A and B", actual, expected, stats);
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Float32 Tests
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 3, 1, 3, 1, 3, 1, 3, 1, 3}, { 10 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}, { 10 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, { 10 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub float32 simple", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1}, { 5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{0, 0, 0, 0, 0, 1, -1, 1, -1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub float32 broadcast A", actual, expected, stats);

    }

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1}, { 5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{0, 0, 0, 0, 0, -1, 1, -1, 1, -1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub float32 broadcast B", actual, expected, stats);

    }

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0}, { 2,1,2 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0}, { 2,2,1 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{0, -1, 1, 0, 0, -1, 1, 0}, { 2,2,2 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub float32 broadcast A and B", actual, expected, stats);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Float64 Tests
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 3, 1, 3, 1, 3, 1, 3, 1, 3}, { 10 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}, { 10 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, { 10 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Float64 simple", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1}, { 5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{0, 0, 0, 0, 0, 1, -1, 1, -1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Float64 broadcast A", actual, expected, stats);

    }

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1}, { 5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{0, 0, 0, 0, 0, -1, 1, -1, 1, -1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Float64 broadcast B", actual, expected, stats);

    }

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0}, { 2,1,2 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0}, { 2,2,1 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{0, -1, 1, 0, 0, -1, 1, 0}, { 2,2,2 }, "expected", device);

        Inferno::Tensor actual = a - b;

        ExpectTensorEq("sub Float64 broadcast A and B", actual, expected, stats);
    }




    std::cout << "Subtraction tests: passed=" << stats.passed << " failed=" << stats.failed << "\n\n\n";


}

