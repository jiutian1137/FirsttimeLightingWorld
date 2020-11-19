#pragma once
#include <cmath>
#include "../numeric/real.h"

namespace clmagic {

	template<typename vec3_t, typename real_t> inline
	vec3_t compute_mass_center(const vec3_t* position_array, const real_t* mass_array, size_t array_size) {
		return weighted_mean(position_array, mass_array, array_size);
	}

	template<typename _Timefn, typename _RealTy> inline
	auto velocity(_Timefn _Distance_func, _RealTy t) {
		// velocity for _Distance_func at time t
		return derivative(_Distance_func, t);// or _Distance / t
	}

	template<typename _Timefn, typename _RealTy> inline
	auto velocity2(_Timefn _Acceleration_func, _RealTy dt) {
		// velocity for _Accleration_func during dt
		return integrate_n(_RealTy(0), dt, _Acceleration_func);// or _Accleration * t
	}

	template<typename _Timefn, typename _RealTy> inline
	auto acceleration(_Timefn _Velocity_func, _RealTy t) {
		return derivative(_Velocity_func, t);// or _Velocity / t
	}

	template<typename _Timefn, typename _RealTy> inline
	auto distance(_Timefn _Velocity_func, _RealTy t0, _RealTy t1) {
		return integrate_n(t0, t1, _Velocity_func);
	}

	template<typename _Timefn, typename _RealTy, typename _VecTy> inline
	_VecTy distance2(_Timefn _Acceleration_func, _RealTy t0, _VecTy t1) {
		return integrate_n(t0, t1,
				[_Acceleration_func, t0](_RealTy tX) {
					return integrate_n(t0, tX, _Acceleration_func);
				});
	}

}// namespace clmagic
