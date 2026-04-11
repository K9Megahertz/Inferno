#include <vector>

namespace Inferno {


    template<typename T>
    static inline T gelu_tanh_approx(T x) {
    const T kAlpha = static_cast<T>(0.7978845608028654); // sqrt(2/pi)
    const T kBeta = static_cast<T>(0.044715);

    T x3 = x * x * x;
    T inner = kAlpha * (x + kBeta * x3);
    return static_cast<T>(0.5) * x * (static_cast<T>(1) + std::tanh(inner));
    }

    template<typename AT, typename RT>
    void cpu_gelu(const AT* aptr, RT* optr, size_t N, size_t off) {
    for (size_t i = 0; i < N; ++i) {
        RT x = static_cast<RT>(aptr[off + i]);
        optr[i] = gelu_tanh_approx<RT>(x);
    }
    }

    template void cpu_gelu<float, float>(const float*, float*, size_t, size_t);
    template void cpu_gelu<double, double>(const double*, double*, size_t, size_t);
    template void cpu_gelu<int, float>(const int*, float*, size_t, size_t);
    template void cpu_gelu<int, double>(const int*, double*, size_t, size_t);



    template <typename AT, typename RT>
    void cpu_gelu_strided(
    const AT* aptr,
    RT* optr,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& astrides,
    const std::vector<size_t>& ostrides,
    size_t aoffset,
    size_t ooffset
    ) {
    const size_t ndim = shape.size();

    if (astrides.size() != ndim || ostrides.size() != ndim) {
        throw std::runtime_error("cpu_gelu_strided: shape/stride rank mismatch");
        exit(1);
    }

    // Total logical elements
    size_t N = 1;
    for (size_t d = 0; d < ndim; ++d) {
        N *= shape[d];
    }

    for (size_t linear = 0; linear < N; ++linear) {
        size_t tmp = linear;

        size_t aidx = aoffset;
        size_t oidx = ooffset;

        // unravel linear index into multi-index, then apply strides
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            size_t coord = tmp % shape[d];
            tmp /= shape[d];

            aidx += coord * astrides[d];
            oidx += coord * ostrides[d];
        }

        RT x = static_cast<RT>(aptr[aidx]);
        optr[oidx] = gelu_tanh_approx<RT>(x);
    }
    }

    template void cpu_gelu_strided<float, float>(const float*, float*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);
    template void cpu_gelu_strided<double, double>(const double*, double*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);
    template void cpu_gelu_strided<int, float>(const int*, float*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);
    template void cpu_gelu_strided<int, double>(const int*, double*, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t);


}