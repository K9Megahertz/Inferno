#pragma once

void RunAdditionTests(Inferno::Device device) {


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
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 10 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 10 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{2, 0, 2, 0, 2, 0, 2, 0, 2, 0}, { 10 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Int32 simple", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1}, { 5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{2, 0, 2, 0, 2, 1, 1, 1, 1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Int32 broadcast A", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0, 1}, { 5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{2, 0, 2, 0, 2, 1, 1, 1, 1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Int32 broadcast B", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0}, { 2,1,2 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Int32, std::vector<int>{1, 0, 1, 0}, { 2,2,1 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Int32, std::vector<int>{2, 1, 1, 0, 2, 1, 1, 0}, { 2,2,2 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Int32 broadcast A and B", actual, expected, stats);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Float32 Tests
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 10 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 10 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{2, 0, 2, 0, 2, 0, 2, 0, 2, 0}, { 10 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add float32 simple", actual, expected, stats);

        actual.backward();

        Inferno::Tensor expected_a_grad(Inferno::DType::Float32, std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, { 10 }, "expected_a_grad", device);

        Inferno::Tensor expected_b_grad(Inferno::DType::Float32, std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, { 10 }, "expected_b_grad", device);

        ExpectTensorEq("Addition grad a", *GetImpl(a)->grad(), expected_a_grad, stats);
        ExpectTensorEq("Addition grad b", *GetImpl(b)->grad(), expected_b_grad, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1}, { 5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{2, 0, 2, 0, 2, 1, 1, 1, 1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add float32 broadcast A", actual, expected, stats);

        actual.backward();

        Inferno::Tensor expected_a_grad(Inferno::DType::Float32, std::vector<float>{2, 2, 2, 2, 2}, { 5 }, "expected_a_grad", device);

        Inferno::Tensor expected_b_grad(Inferno::DType::Float32, std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, { 2,5 }, "expected_b_grad", device);

        ExpectTensorEq("Addition grad a broadcast", *GetImpl(a)->grad(), expected_a_grad, stats);
        ExpectTensorEq("Addition grad b broadcast", *GetImpl(b)->grad(), expected_b_grad, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0, 1}, { 5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{2, 0, 2, 0, 2, 1, 1, 1, 1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add float32 broadcast B", actual, expected, stats);

        actual.backward();

        Inferno::Tensor expected_a_grad(Inferno::DType::Float32, std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, { 2,5 }, "expected_a_grad", device);

        Inferno::Tensor expected_b_grad(Inferno::DType::Float32, std::vector<float>{2, 2, 2, 2, 2}, { 5 }, "expected_b_grad", device);

        ExpectTensorEq("Addition grad a broadcast B", *GetImpl(a)->grad(), expected_a_grad, stats);
        ExpectTensorEq("Addition grad b broadcast B", *GetImpl(b)->grad(), expected_b_grad, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0}, { 2,1,2 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float32, std::vector<float>{1, 0, 1, 0}, { 2,2,1 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float32, std::vector<float>{2, 1, 1, 0, 2, 1, 1, 0}, { 2,2,2 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add float32 broadcast A and B", actual, expected, stats);

        actual.backward();

        Inferno::Tensor expected_a_grad(Inferno::DType::Float32, std::vector<float>{2, 2, 2, 2}, { 2,1,2 }, "expected_a_grad", device);

        Inferno::Tensor expected_b_grad(Inferno::DType::Float32, std::vector<float>{2, 2, 2, 2}, { 2,2,1 }, "expected_b_grad", device);

        ExpectTensorEq("Addition grad a", *GetImpl(a)->grad(), expected_a_grad, stats);
        ExpectTensorEq("Addition grad b", *GetImpl(b)->grad(), expected_b_grad, stats);
    }

  
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Float64 Tests
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 10 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 10 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{2, 0, 2, 0, 2, 0, 2, 0, 2, 0}, { 10 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Float64 simple", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1}, { 5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{2, 0, 2, 0, 2, 1, 1, 1, 1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Float64 broadcast A", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, { 2,5 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0, 1}, { 5 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{2, 0, 2, 0, 2, 1, 1, 1, 1, 1}, { 2,5 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Float64 broadcast B", actual, expected, stats);
    }

    {
        Inferno::Tensor a(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0}, { 2,1,2 }, "a", device);

        Inferno::Tensor b(Inferno::DType::Float64, std::vector<double>{1, 0, 1, 0}, { 2,2,1 }, "b", device);

        Inferno::Tensor expected(Inferno::DType::Float64, std::vector<double>{2, 1, 1, 0, 2, 1, 1, 0}, { 2,2,2 }, "expected", device);

        Inferno::Tensor actual = a + b;

        ExpectTensorEq("add Float64 broadcast A and B", actual, expected, stats);
    }



    std::cout << "Addition tests: passed=" << stats.passed << " failed=" << stats.failed << "\n\n\n";
}

