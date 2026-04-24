#pragma once
#include "node.h"
#include <inferno/core/tensor.h>





namespace Inferno {


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Class GeluBackward 
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class GeluBackward : public Node {

	public:

		GeluBackward(const Tensor& A, const Tensor& out);
		void backward() override;


	private:

		void get_inputs(std::vector<Tensor>& out) const override;
		void release() override;


		Tensor m_A;
		Tensor m_out;


	};



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function gelu_grad_tanh_approx
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T>
    static inline T gelu_grad_tanh_approx(T x) {
        const T kAlpha = static_cast<T>(0.7978845608028654); // sqrt(2/pi)
        const T kBeta = static_cast<T>(0.044715);
        const T kThreeBeta = static_cast<T>(3.0) * kBeta;
        const T kHalf = static_cast<T>(0.5);
        const T kOne = static_cast<T>(1.0);

        T x2 = x * x;
        T x3 = x2 * x;

        T u = kAlpha * (x + kBeta * x3);
        T t = std::tanh(u);
        T sech2 = kOne - t * t;

        return kHalf * (kOne + t)
            + kHalf * x * sech2 * kAlpha * (kOne + kThreeBeta * x2);
    }


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function cpu_gelu_backward
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    template<typename AT, typename GT, typename RT>
    void cpu_gelu_backward(const AT* aptr, const GT* gptr, RT* optr, size_t N, size_t off) {
        for (size_t i = 0; i < N; ++i) {
            RT x = static_cast<RT>(aptr[off + i]);
            RT g = static_cast<RT>(gptr[i]);

            RT dgelu = gelu_grad_tanh_approx<RT>(x);
            optr[i] = g * dgelu;
        }
    }



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Explicit Instantations
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template void cpu_gelu_backward<float, float, float>(const float*, const float*, float*, size_t, size_t);
    template void cpu_gelu_backward<double, double, double>(const double*, const double*, double*, size_t, size_t);
    template void cpu_gelu_backward<float, double, double>(const float*, const double*, double*, size_t, size_t);
    template void cpu_gelu_backward<double, float, double>(const double*, const float*, double*, size_t, size_t);
    template void cpu_gelu_backward<int, float, float>(const int*, const float*, float*, size_t, size_t);
    template void cpu_gelu_backward<int, double, double>(const int*, const double*, double*, size_t, size_t);


    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cpu_gelu_backward_strided
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename AT, typename GT, typename RT>
    void cpu_gelu_backward_strided(
        const AT* aptr,
        const GT* gptr,
        RT* optr,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& astrides,
        const std::vector<size_t>& gstrides,
        const std::vector<size_t>& ostrides,
        size_t aoffset,
        size_t goffset,
        size_t ooffset
    ) {
        const size_t ndim = shape.size();

        if (astrides.size() != ndim ||
            gstrides.size() != ndim ||
            ostrides.size() != ndim) {
            throw std::runtime_error("cpu_gelu_backward_strided: shape/stride rank mismatch");
        }

        // Total logical elements
        size_t N = 1;
        for (size_t d = 0; d < ndim; ++d) {
            N *= shape[d];
        }

        for (size_t linear = 0; linear < N; ++linear) {
            size_t tmp = linear;

            size_t aidx = aoffset;
            size_t gidx = goffset;
            size_t oidx = ooffset;

            // unravel linear index into multi-index, then apply strides
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                size_t coord = tmp % shape[d];
                tmp /= shape[d];

                aidx += coord * astrides[d];
                gidx += coord * gstrides[d];
                oidx += coord * ostrides[d];
            }

            RT x = static_cast<RT>(aptr[aidx]);
            RT g = static_cast<RT>(gptr[gidx]);

            RT dgelu = gelu_grad_tanh_approx<RT>(x);
            optr[oidx] = g * dgelu;
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Explicit Instantations
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template void cpu_gelu_backward_strided<float, float, float>(
        const float*, const float*, float*, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        size_t, size_t, size_t);

    template void cpu_gelu_backward_strided<double, double, double>(
        const double*, const double*, double*, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        size_t, size_t, size_t);

    template void cpu_gelu_backward_strided<float, double, double>(
        const float*, const double*, double*, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&,
        size_t, size_t, size_t);

    template void cpu_gelu_backward_strided<double, float, double>(
        const double*, const float*, double*, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&,
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        size_t, size_t, size_t);

    template void cpu_gelu_backward_strided<int, float, float>(
        const int*, const float*, float*, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        size_t, size_t, size_t);

    template void cpu_gelu_backward_strided<int, double, double>(
        const int*, const double*, double*, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        const std::vector<size_t>&, 
        size_t, size_t, size_t);

}