#pragma once
#include <inferno/inferno.h>
#include "logger.h"
#include "dtype_dispatch.h"

constexpr double EPSILON = 0.00001;

struct TestStats {
    int passed = 0;
    int failed = 0;
};

bool TensorEquals(const Inferno::Tensor& A, const Inferno::Tensor& B);

template <typename AT, typename BT>
bool are_vals_equal(AT vala, BT valb);

void ExpectTensorEq(const std::string& name, const Inferno::Tensor& actual, const Inferno::Tensor& expected, TestStats& stats);
void ExpectTrue(bool cond, const std::string& name, TestStats& stats);



void ExpectTrue(bool cond, const std::string& name, TestStats& stats)
{
    if (cond) {
        ++stats.passed;
        Logger::Append(Logger::LogLevel::LOGLEVEL_INFO, "[PASS] " + name);
        //std::cout << "[PASS] " << name << "\n";
    }
    else {
        ++stats.failed;
        Logger::Append(Logger::LogLevel::LOGLEVEL_INFO, "[FAIL] " + name);
        //std::cout << "[FAIL] " << name << "\n";
    }
}

void ExpectTensorEq(const std::string& name, const Inferno::Tensor& actual, const Inferno::Tensor& expected, TestStats& stats)
{
    if (TensorEquals(actual, expected)) {
        ++stats.passed;
        Logger::Append(Logger::LogLevel::LOGLEVEL_INFO, "[PASS] " + name);
        //std::cout << "[PASS] " << name << "\n";
    }
    else {
        ++stats.failed;
        std::cout << "[FAIL] " << name << "\n";
        std::cout << "  actual:   " << actual << "\n";
        std::cout << "  expected: " << expected << "\n";
    }
}



//Returns true if they're the same
template <typename AT, typename BT>
bool are_vals_equal(AT vala, BT valb) {

    if constexpr (std::is_floating_point_v<AT> || std::is_floating_point_v<BT>)
    {
        double a = static_cast<double>(vala);
        double b = static_cast<double>(valb);

        double diff = std::abs(a - b);
        if (diff > EPSILON)
            return false;
    }
    else
    {
        // exact compare for integers
        if (vala != valb)
            return false;
    }

    return true;


}

/*bool TensorEquals(const Inferno::Tensor& A, const Inferno::Tensor& B) {


	return dispatchTwo(A.dtype(), B.dtype(), [&](auto TA, auto TB) {

		using AT = typename decltype(TA)::type;
		using BT = typename decltype(TB)::type;

		Inferno::Tensor ta = A.to(Inferno::Device::cpu());
		Inferno::Tensor tb = B.to(Inferno::Device::cpu());


		if (ta.numel() != tb.numel())
			return false;


        if (ta.shape() != tb.shape())
            return false;

        if (ta.strides() != tb.strides())
            return false;



		auto a_dptr = GetImpl(ta)->data_as_ptr<AT>();
		auto b_dptr = GetImpl(tb)->data_as_ptr<BT>();
        for (size_t i = 0; i < ta.numel(); i++) {
            if (!are_vals_equal(a_dptr[i], b_dptr[i]))
                return false;
        }
		return true;

        //check the grad tensors, use or own functions
        std::shared_ptr<Inferno::Tensor> grada = GetImpl(ta)->grad();
        std::shared_ptr<Inferno::Tensor> gradb = GetImpl(tb)->grad();

        // one has grad, the other doesn't -> not equal
        if ((grada == nullptr) != (gradb == nullptr))
            return false;

        if (grada && gradb) {
            if (!TensorEquals(*GetImpl(ta)->grad(), *GetImpl(tb)->grad()))
                return false;
        }
        return true;

	});
  
}*/ 

