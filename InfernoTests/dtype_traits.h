#include "dtype.h"

namespace Inferno {

	// 1) DType <-> C++ type mapping
	template <DType> struct DTypeToCpp;
	template <> struct DTypeToCpp<DType::Int32> { using type = int; };
	template <> struct DTypeToCpp<DType::Float32> { using type = float; };
	template <> struct DTypeToCpp<DType::Float64> { using type = double; };

	template <typename T> struct CppToDType;
	template <> struct CppToDType<int> { static constexpr DType value = DType::Int32; };
	template <> struct CppToDType<float> { static constexpr DType value = DType::Float32; };
	template <> struct CppToDType<double> { static constexpr DType value = DType::Float64; };

	template <DType DT>
	using cpp_type_t = typename DTypeToCpp<DT>::type;

	template <typename T>
	constexpr DType dtype_of_v = CppToDType<T>::value;

}