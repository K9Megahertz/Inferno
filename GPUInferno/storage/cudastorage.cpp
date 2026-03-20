#include "cudastorage.h"

namespace  Inferno {



    CUDAStorage::CUDAStorage(size_t bytes) {
        cudaMalloc(&ptr, bytes);
    }

    CUDAStorage::~CUDAStorage() {
        cudaFree(ptr);
    }
}