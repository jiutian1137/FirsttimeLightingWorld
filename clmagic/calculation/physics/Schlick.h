//--------------------------------------------------------------------------------------
// Copyright (c) 1994 Christophe-Schlick
// All Rights Reserved
//--------------------------------------------------------------------------------------
#pragma once
#ifndef clmagic_calculation_physic_SCHLICK_h_
#define clmagic_calculation_physic_SCHLICK_h_
#include "../lapack/vector.h"

namespace Schlick {

	template<typename _Ts, typename _Tb>
	clmagic::VECTOR3<_Ts, _Tb> Fresnel_reflectance(clmagic::VECTOR3<_Ts, _Tb> F0, 
		clmagic::unit_vector3<_Ts, _Tb> N, clmagic::unit_vector3<_Ts, _Tb> L) {
		using clmagic::pow;
		using clmagic::positive;
		return F0 + (((_Ts)1) - F0) * pow( ((_Ts)1) - positive(dot(N, L)), (_Ts)5 );
	}

	template<typename _Ts, typename _Tb>
	clmagic::VECTOR3<_Ts, _Tb> Fresnel_reflectance(clmagic::VECTOR3<_Ts, _Tb> F0, clmagic::VECTOR3<_Ts, _Tb> F90, _Ts p, 
		clmagic::unit_vector3<_Ts, _Tb> N, clmagic::unit_vector3<_Ts, _Tb> L) {
		using clmagic::pow;
		using clmagic::positive;
		return F0 + (F90 - F0) * pow(((_Ts)1) - positive(dot(N, L)), p);
	}

}// Christophe_Schlick

#endif