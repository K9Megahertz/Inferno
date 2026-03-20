#include "cudaops.h"

namespace Inferno {



    template<typename AT, typename BT, typename RT>
    __global__ void multiply_kernel(const AT* aptr, const BT* bptr, RT* outptr, size_t N) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) outptr[i] = static_cast<RT>(aptr[i]) * static_cast<RT>(bptr[i]);
    }

    template<typename AT, typename BT, typename RT>
    void cuda_multiply(const AT* aptr, const BT* bptr, RT* outptr, size_t N) {
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        multiply_kernel << <grid, block >> > (aptr, bptr, outptr, N);
    }

    template void cuda_multiply<int, int, int>(const int*, const int*, int*, size_t);
    template void cuda_multiply<int, float, float>(const int*, const float*, float*, size_t);
    template void cuda_multiply<float, int, float>(const float*, const int*, float*, size_t);

    template void cuda_multiply<float, float, float>(const float*, const float*, float*, size_t);
    template void cuda_multiply<float, double, double>(const float*, const double*, double*, size_t);
    template void cuda_multiply<double, float, double>(const double*, const float*, double*, size_t);

    template void cuda_multiply<double, double, double>(const double*, const double*, double*, size_t);
    template void cuda_multiply<double, int, double>(const double*, const int*, double*, size_t);
    template void cuda_multiply<int, double, double>(const int*, const double*, double*, size_t);

}