#pragma once

void RunMSELossTests(Inferno::Device device) {


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
        Inferno::Tensor prediction(
            Inferno::DType::Float32,
            std::vector<float>{2.0f, 3.0f},
            { 2 },
            "prediction",
            device
        );

        Inferno::Tensor target(
            Inferno::DType::Float32,
            std::vector<float>{1.0f, 5.0f},
            { 2 },
            "target",
            device
        );

        Inferno::Tensor expected_loss(
            Inferno::DType::Float32,
            std::vector<float>{2.5f},
            { 1 },
            "expected_loss",
            device
        );

        Inferno::Tensor expected_prediction_data(
            Inferno::DType::Float32,
            std::vector<float>{2.0f, 3.0f},
            { 2 },
            "expected_prediction_data",
            device
        );

        Inferno::Tensor expected_prediction_grad(
            Inferno::DType::Float32,
            std::vector<float>{1.0f, -2.0f},
            { 2 },
            "expected_prediction_grad",
            device
        );

        Inferno::MSELoss loss_fn;
        Inferno::Tensor loss = loss_fn(prediction, target);

        // forward check
        ExpectTensorEq("MSELoss forward simple", loss, expected_loss, stats);

        // backward
        loss.backward();

        // data should not change
        ExpectTensorEq("MSELoss backward prediction data unchanged", prediction, expected_prediction_data, stats);

        // grad should match expected
        auto pgrad = GetImpl(prediction)->grad();
        if (!pgrad)
        {
            stats.failed++;
            std::cout << "[FAIL] MSELoss backward prediction grad missing\n";
        }
        else
        {
            ExpectTensorEq("MSELoss backward prediction grad", *pgrad, expected_prediction_grad, stats);
        }

        // optional target grad check
        auto tgrad = GetImpl(target)->grad();
        if (tgrad)
        {
            Inferno::Tensor expected_target_grad(
                Inferno::DType::Float32,
                std::vector<float>{-1.0f, 2.0f},
                { 2 },
                "expected_target_grad",
                device
            );

            ExpectTensorEq("MSELoss backward target grad", *tgrad, expected_target_grad, stats);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Float32 Tests
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Float64 Tests
    //    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    

    std::cout << "Addition tests: passed=" << stats.passed << " failed=" << stats.failed << "\n\n\n";
}

