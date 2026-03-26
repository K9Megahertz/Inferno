template<typename T>
__device__ inline void gpu_atomic_add(T* addr, T val);

template<>
__device__ inline void gpu_atomic_add<float>(float* addr, float val)
{
	atomicAdd(addr, val);
}

template<>
__device__ inline void gpu_atomic_add<double>(double* addr, double val)
{
	atomicAdd(addr, val);
}

template<>
__device__ inline void gpu_atomic_add<int>(int* addr, int val)
{
	atomicAdd(addr, val);
}