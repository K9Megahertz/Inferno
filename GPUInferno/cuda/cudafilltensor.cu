#include "cudaops.h"

namespace Inferno {

    

    template <typename AT>
    __global__ void cuda_fill_kernel(AT* data, const AT value, size_t n)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] = value;
    }

    template <typename AT>
    void cuda_fill(AT* aptr, const AT value, size_t N) {
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        cuda_fill_kernel << <grid, block >> > (aptr, value, N);
    }

    template void cuda_fill<int>(int*, const int, size_t);
    template void cuda_fill<float>(float*, const float, size_t);
    template void cuda_fill<double>(double*, const double, size_t);


}
