//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Free License
//
// Table of contents
// 1. define unit tag --------------------------------------------------------
// 2. define unit ------------------------------------------------------------
// 3. dynamics formulas ------------------------------------------------------
// 4. dynamics quantities ----------------------------------------------------
//--------------------------------------------------------------------------------------
#pragma once
#ifndef clmagic_calculation_physic_DYNAMICS_h_
#define clmagic_calculation_physic_DYNAMICS_h_
#include "fundamental.h"
#include "../lapack/vector.h"

namespace clmagic {

	// 1. define unit tag --------------------------------------------------------

	struct velocity_unit_tag {};
	struct acceleration_unit_tag {};
	struct force_unit_tag {};
	struct energy_unit_tag {};
	struct momentum_unit_tag {};

	// 2. define unit ------------------------------------------------------------

	template<typename _Ul, typename _Ut>
	struct velocity_unit {
		using unit_category = velocity_unit_tag;
		using length_unit = _Ul;
		using time_unit = _Ut;
	};

	template<typename _Ul, typename _Ut>
	struct acceleration_unit {
		using unit_category = acceleration_unit_tag;
		using length_unit = _Ul;
		using time_unit = _Ut;
	};

	template<typename _Um, typename _Ul, typename _Ut>
	struct force_unit {
		using unit_category = force_unit_tag;
		using mass_unit = _Um;
		using length_unit = _Ul;
		using time_unit = _Ut;
	};

	template<typename _Um, typename _Ul, typename _Ut>
	struct energy_unit {
		using unit_category = energy_unit_tag;
		using mass_unit = _Um;
		using length_unit = _Ul;
		using time_unit = _Ut;
	};

	template<typename _Um, typename _Ul, typename _Ut>
	struct momentum_unit {
		using unit_category = momentum_unit_tag;
		using mass_unit = _Um;
		using length_unit = _Ul;
		using time_unit = _Ut;
	};

	// { velocity unit }
	using meter_per_second = velocity_unit<meter, second>;
	using kilometer_per_hour = velocity_unit<kilometer, hour>;
	using knot = velocity_unit<nautical_mile, hour>;
	// { acceleration unit }
	using meter_per_square_second = acceleration_unit<meter, second>;
	using kilometer_per_square_hour = acceleration_unit<kilometer, hour>;
	// { force unit }
	using Newton = force_unit<kilogram, meter, second>;
	using pound_force = force_unit<pound_mass, foot, second>;
	// { energy unit }
	using Joule = energy_unit<kilogram, meter, second>;
	// { momentum unit }
	using kilogram_meter_per_second = momentum_unit<kilogram, meter, second>;

	using SI_velocity = meter_per_second;
	using SI_acceleration = meter_per_square_second;
	using SI_force = Newton;
	using SI_energy = Joule;
	using SI_momentum = kilogram_meter_per_second;

	// 3. dynamics formulas ------------------------------------------------------

	template<typename _Ty, typename _Ul, typename _Ut>
	using velocity_formula = formula_divides<length_formula<_Ty, _Ul>,
		time_formula<_Ty, _Ut>>;

	template<typename _Ty, typename _Ul, typename _Ut>
	using acceleration_formula = formula_divides<velocity_formula<_Ty,_Ul,_Ut>, time_formula<_Ty,_Ut>>;

	template<typename _Ty, typename _Um, typename _Ul, typename _Ut>
	using force_formula    = formula_multiplies<mass_formula<_Ty,_Um>, acceleration_formula<_Ty,_Ul,_Ut>>;

	template<typename _Ty, typename _Um, typename _Ul, typename _Ut>
	using momentum_formula = formula_multiplies<mass_formula<_Ty,_Um>, velocity_formula<_Ty,_Ul,_Ut>>;
	
	template<typename _Ty, typename _Um, typename _Ul, typename _Ut>
	using energy_formula   = formula<force_formula<_Ty,_Um,_Ul,_Ut>, length_formula<_Ty,_Ul>, scalar_product<>>;

	template<typename _Tv, typename _Ts, typename _Ul, typename _Ut>
	using directional_velocity_formula = formula_multiplies<_Tv, velocity_formula<_Ts, _Ul, _Ut>>;

	template<typename _Tv, typename _Ts, typename _Ul, typename _Ut>
	using directional_acceleration_formula = formula_multiplies<_Tv, acceleration_formula<_Ts, _Ul, _Ut>>;

	template<typename _Tv, typename _Ts, typename _Um, typename _Ul, typename _Ut>
	using directional_force_formula  = formula_multiplies<_Tv, force_formula<_Ts, _Um, _Ul, _Ut>>;

	template<typename _Tv, typename _Ts, typename _Um, typename _Ul, typename _Ut>
	using directional_energy_formula = formula<directional_force_formula<_Tv,_Ts,_Um,_Ul,_Ut>, length_formula<_Tv,_Ul>, scalar_product<>>;

	template<typename _Ts>
	formula_t<formula_result<energy_formula<_Ts, kilogram, meter, second>>> make_photon_energy( length_t<_Ts> lambda ) {
		const auto Planck_constant = static_cast<_Ts>(6.62607004 * 10E-34);// [J]/[s]
		const auto light_velocity  = static_cast<_Ts>(29'9792'458);// [m]/[s]
		return formula_t<formula_result<energy_formula<_Ts, kilogram, meter, second>>>(
			Planck_constant * light_velocity / lambda.get_result() 
		);
	}

// 4. dynamics quantities ----------------------------------------------------

	template<typename _Ty, typename _Uv = SI_velocity>
	using velocity_t    = formula_t<velocity_formula<_Ty, typename _Uv::length_unit, typename _Uv::time_unit> >;
	
	template<typename _Ty, typename _Ua = SI_acceleration>
	using acceleration_t = formula_t<acceleration_formula<_Ty, typename _Ua::length_unit, typename _Ua::time_unit> >;

	template<typename _Ty, typename _Uf = SI_force>
	using force_t        = formula_t<force_formula<_Ty, typename _Uf::mass_unit, typename _Uf::length_unit, typename _Uf::time_unit>>;

	template<typename _Ty, typename _Um = SI_momentum>
	using momentum_t     = formula_t<momentum_formula<_Ty, typename _Um::mass_unit, typename _Um::length_unit, typename _Um::time_unit> >;

	/*template<typename _Tv, typename _Uv = SI_velocity>
	using directional_velocity_t = formula_t<directional_velocity_formula<_Tv, 
		                                                                  scalar_t<_Tv>, 
																		  typename _Uv::length_unit, 
																		  typename _Uv::time_unit> >;

	template<typename _Ty, typename _Ul, typename _Ut>
	using weight_acceleration_formula = formula_minus<acceleration_formula<_Ty, _Ul, _Ut>,
													  formula_multiplies<length_formula<_Ty, _Ul>, formula_multiplies<std::pair<_Ty, _Ut>, std::pair<_Ty, _Ut>> > >;*/



	/*<notation>
		d: displacement 
		s: distance
		t: time
		v: velocity
		a: accelration
		M: mass
		F: force
		p: linear momentum
		r: radius
		Theta: angle
		Omega: angular velocity
		Alpha: angular accelration
	</notation>*/

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_distance(_Tv v0, _Tv a, _Ts t) {// s = v0*t + integral(integral(a, t), t)
		return v0*t + 0.5f*a*t*t;
	}

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_velocity(_Tv ds, _Ts dt) {// v = ds/dt
		return ds / dt;
	}

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_velocity(_Tv v0, _Tv a, _Ts t) {// v = v0 + a*t
		return v0 + a*t;
	}

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_acceleration(_Tv dv, _Ts dt) {// a = dv/dt
		return dv / dt;
	}

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_acceleration(_Tv ds, _Ts dt1, _Ts dt2) {// a = ds/ddt
		return ds / dt1 / dt2;
	}
	
	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_force(_Ts M, _Tv a) {// F = M*a
		return M * a;
	}

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_force(_Ts M, _Tv ds, _Ts dt1, _Ts dt2) {// F = M*(ds/ddt)
		return M * _Calc_acceleration(ds, dt1, dt2);
	}
	


	template<typename _Tv, typename _Ts> inline
	_Ts _Calc_kinetic_energy(_Ts M, _Tv v){// 1/2 * M * pow(v,2)
		return 0.5f * M * dot(v,v);
	}

	template<typename _Tv> inline
	auto _Calc_kinetic_energy(_Tv F, _Tv s) -> decltype(dot(F,s)) {// dot(F,s)
		return dot(F,s);
	}


	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_linear_momentum(_Ts M, _Tv v) {// p = M*v
		return M * v;
	}

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_linear_momentum(_Tv F, _Ts t) {// p = integral(F,t)
		return F * t;
	}

	template<typename _Tv, typename _Ts> inline
	_Tv _Calc_force2(_Tv dp, _Ts dt) {// F = dp/dt
		return dp / dt;
	}

	// { same axis }
	//template<typename _Ts>
	//_Ts _Calc_angle(_Ts Omega0, _Ts Alpha, _Ts t) {// Theta = Omega0*t + integral(integral(Alpha, t), t)
	//	return Omega0*t + 0.5f*Alpha*t*t;
	//}

	//template<typename _Ts> inline
	//_Ts _Calc_angular_velocity(_Ts dTheta, _Ts dt) {// dOmega = dTheta/dt
	//	return dTheta / dt;
	//}

	//template<typename _Ts> inline
	//_Ts _Calc_angular_accleration(_Ts dTheta, _Ts dt) {// dAlpha = dTheta/dt/dt
	//	return dTheta / dt / dt;
	//}

	template<typename _Ts>
	_Ts _Calc_distance2(_Ts Theta, _Ts r) {// s = Theta*r
		return Theta * r;
	}

	template<typename _Ts>
	_Ts _Calc_velocity2(_Ts Omega, _Ts r) {// (ds/dt) = (dTheta/dt)*r
		return Omega * r;
	}
	
	template<typename _Ts> inline
	_Ts _Calc_tangential_component(_Ts Alpha, _Ts r) {
		return Alpha * r;
	}

	template<typename _Ts> inline
	_Ts _Calc_radial_component(_Ts Omega, _Ts r) {
		return Omega*Omega * r;
	}

	template<typename _Tv, typename _Ts> inline
	_Ts _Calc_radial_component(_Tv v, _Ts r) {// pow(v,2)/r
		return dot(v, v) / r;
	}

	template<typename _Ts>
	_Ts _Calc_kinetic_energey2(_Ts M, _Ts Omega, _Ts r) {// (1/2)*M*pow(Omega*r,2)
		return 0.5f * M * pow(Omega*r, 2);
	}

	template<typename _Ts>
	_Ts _Calc_rotary_inertia(const _Ts* M, const _Ts* R, size_t count) {// sum( M[i] * pow(R[i],2) )
		_Ts I = M[0] * pow(R[0],2);
		for (size_t i = 1; i != count; ++i) {
			I += M[i] * pow(R[i], 2);
		}
		return I;
	}

	template<typename _Tv, typename _Ts>
	_Ts _Calc_rotary_inertia(const _Tv* P, const _Ts* M, size_t count, _Tv p0, _Tv v0) {
		_Ts I = M[0] * length_squared(P[0] - proj(P[0]-p0, v0));
		for (size_t i = 1; i != count; ++i) {
			I += M[i] * length_squared(P[i] - proj(P[i]-p0, v0));
		}
		return I;
	}

	template<typename _Tv, typename _Ts>
	_Ts _Calc_kinetic_energy2(const _Tv* P, const _Ts* M, size_t count, _Tv p0, _Tv v0, _Ts Omega) {// (1/2)*I*pow(Omega,2)
		const _Ts I = _Calc_rotary_inertia(P, M, count, p0, v0);
		return 0.5f * I * Omega*Omega;
	}

	//template<typename _Tv, typename _Ts = scalar_t<_Tv>,
	//	typename _Length_unit = SI_length, typename _Mass_unit = SI_mass>
	//struct mass_point {
	//	using scalar_type = _Ts;
	//	using vector_type = _Tv;

	//	using length_type = quantity<vector_type, _Length_unit>;
	//	using mass_type   = quantity<scalar_type, _Mass_unit>;

	//	length_type position;
	//	mass_type   mass;

	//	mass_point() = default;
	//	// construct by a number
	//	mass_point(vector_type _Pos_vector) : position(length_type(_Pos_vector)), mass(KILOGRAMS(1)) {}
	//	// construct by quantities
	//	mass_point(length_type _Position, mass_type _Mass) : position(_Position), mass(_Mass) {}

	//	mass_point& operator=(const mass_point&) = default;
	//	// assign by a number
	//	mass_point& operator=(vector_type _Pos_vector) {
	//		position = length_type(_Pos_vector);
	//		mass     = KILOGRAMS(1);
	//		return *this;
	//	}
	//};

	//template<typename _Tv, typename _Length_unit = SI_length>
	//struct axis {
	//	using vector_type = _Tv;
	//	using length_type = quantity<vector_type, _Length_unit>;

	//	length_type position;
	//	vector_type direction;

	//	axis() = default;
	//	// construct by a number
	//	axis(vector_type _Direction) : position(METERS(vector_type(0))), direction(_Direction) {}
	//	// construct by quantities
	//	axis(length_type _Position, vector_type _Direction) : position(_Position), direction(_Direction) {}

	//	axis& operator=(const axis&) = default;
	//	// assign by a number
	//	axis& operator=(vector_type _Direction) {
	//		position  = METERS(vector_type(0));
	//		direction = _Direction;
	//		return *this;
	//	}
	//};

	//template<typename _Tv, typename _Ts>
	//_Ts calc_rotary_inertia(const std::vector<mass_point<_Tv, _Ts>>& _Points, axis<_Tv> _Axis) {
	//	const _Tv p0 = _Axis.position.num;
	//	const _Tv v0 = _Axis.direction;
	//	_Ts       I  = 0;

	//	auto       _First = _Points.begin();
	//	const auto _Last  = _Points.end();
	//	for ( ; _First != _Last; ++_First) {
	//		_Tv p = _First->position.num;
	//		_Ts m = _First->mass.num;

	//		_Ts r2 = length_squared(p - proj(p-p0, v0));
	//		    I += m * r2;
	//	}
	//	return I; 
	//}

	

	/*<formula>
		       v = v0 + a * t
		                   1
		  x - x0 = v0*t + ---*a*pow(t,2)
		                   2
		pow(v,2) = pow(v0,2) + 2*a*(x - x0)
		            1
		  x - x0 = ---*(v0+v)*t
		            2
					      1
		  x - x0 = v*t - ---*a*pow(t,2)
		                  2
	</formula>*/

}// namespace clmagic

#endif