#include "cudastorage.h"

namespace  Inferno {



    CUDAStorage::CUDAStorage(size_t numbytes) {
        cudaMalloc(&ptr, numbytes);
        m_numbytes = numbytes;
    }

    CUDAStorage::~CUDAStorage() {
        cudaFree(ptr);
    }
}