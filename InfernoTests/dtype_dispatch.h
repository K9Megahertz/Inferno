#include <inferno/core/dtype.h>

namespace Inferno {



	// 2) Promotion rules 
	template <typename A, typename B> struct Promote;
	template <> struct Promote<int, int> { using type = int; };
	template <> struct Promote<int, float> { using type = float; };
	template <> struct Promote<float, int> { using type = float; };
	template <> struct Promote<float, float> { using type = float; };
	template <> struct Promote<double, float> { using type = double; };
	template <> struct Promote<float, double> { using type = double; };
	template <> struct Promote<double, int> { using type = double; };
	template <> struct Promote<int, double> { using type = double; };
	template <> struct Promote<double, double> { using type = double; };


	//template <typename A> struct PromoteSingle;
	//template <> struct PromoteSingle<int> { using type = float; };
	//template <> struct PromoteSingle<float> { using type = float; };
	//template <> struct PromoteSingle<double> { using type = double; };


	template <typename A, typename B>
	using promote_t = typename Promote<A, B>::type;

	// 3) Tiny tags to pass types from runtime switches into templates
	template <typename T> struct Tag { using type = T; };

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Dispatch Functions
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename F>
	auto dispatchOne(DType a_dt, F&& fn) {
		switch (a_dt) {
		case DType::Int32: return fn(Tag<int>{});
		case DType::Float32: return fn(Tag<float>{});
		case DType::Float64: return fn(Tag<double>{});
			break;
		}
		throw std::runtime_error("Unsupported dtype combination");
		//exit(1);
	}


	template <typename F>
	auto dispatchTwo(DType a_dt, DType b_dt, F&& fn) {
		switch (a_dt) {
		case DType::Int32:
			switch (b_dt) {
			case DType::Int32: return fn(Tag<int>{}, Tag<int>{});
			case DType::Float32: return fn(Tag<int>{}, Tag<float>{});
			case DType::Float64: return fn(Tag<int>{}, Tag<double>{});
			}
			break;
		case DType::Float32:
			switch (b_dt) {
			case DType::Int32: return fn(Tag<float>{}, Tag<int>{});
			case DType::Float32: return fn(Tag<float>{}, Tag<float>{});
			case DType::Float64: return fn(Tag<float>{}, Tag<double>{});
			}
			break;
		case DType::Float64:
			switch (b_dt) {
			case DType::Int32: return fn(Tag<double>{}, Tag<int>{});
			case DType::Float32: return fn(Tag<double>{}, Tag<float>{});
			case DType::Float64: return fn(Tag<double>{}, Tag<double>{});
			}
			break;
		}
		throw std::runtime_error("Unsupported dtype combination");
		//exit(1);
	}


	/*template <typename F>
	decltype(auto) dispatchThree(DType a, DType b, DType c, F&& fn) {
		return dispatchOne(a, [&](auto TA) -> decltype(auto) {
			return dispatchOne(b, [&](auto TB) -> decltype(auto) {
				return dispatchOne(c, [&](auto TC) -> decltype(auto) {
					return fn(TA, TB, TC);
					});
				});
			});
	}*/


}