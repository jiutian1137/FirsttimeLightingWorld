//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Free License
//
// Table of contents
// 1. define unit tag --------------------------------------------------------
// 2. define unit ratio ------------------------------------------------------
// 3. define unit ------------------------------------------------------------
// 4. fundamental quantities -------------------------------------------------
//--------------------------------------------------------------------------------------
#pragma once
#define clmagic_calculation_physics_FUNDAMENTAL

#include "../numeric/formula_e.h"

namespace calculation {

#pragma region unit_tag

	struct length_unit_tag {};
	struct angle_unit_tag {};
	struct time_unit_tag {};
	struct mass_unit_tag {};
	struct temperature_unit_tag {};
	struct electric_current_unit_tag {};
	struct amount_of_light_unit_tag {};
	struct amount_of_matter_unit_tag {};

#pragma endregion

#pragma region unit_ratio

	// { length_unit ratio, SI }
	using femtometer_ratio    = std::ratio<1000'0000'0000'0000/*fm*/, 1/*m*/>;
	using picometer_ratio     = std::ratio<   1'0000'0000'0000/*pm*/, 1/*m*/>;
	using nanometer_ratio     = std::ratio<       10'0000'0000/*nm*/, 1/*m*/>;
	using micrometer_ratio    = std::ratio<           100'0000/*um*/, 1/*m*/>;
	using millimeter_ratio    = std::ratio<               1000/*mm*/, 1/*m*/>;
	using centimeter_ratio    = std::ratio<                100/*cm*/, 1/*m*/>;
	using meter_ratio         = std::ratio<                  1/*m*/,  1/*m*/>; //SI unit
	using kilometer_ratio     = std::ratio<                  1/*km*/, 1000/*m*/>;
	using light_year_ratio    = std::ratio<                  1/*ly*/, 9460'0000'0000'0000/*m*/>;
	// { angle_unit ratio, SI }
	using radian_ratio     = std::ratio<10/*rad*/, 573/*deg*/>;// SI_unit, 1[rad] = 57.3[deg]
	using degree_ratio     = std::ratio< 1/*deg*/,   1/*deg*/>;
	using revolution_ratio = std::ratio< 1/*rev*/, 360/*deg*/>;
	// { mass_unit ratio, SI }
	using gram_ratio     = std::ratio<1000/*g*/,      1/*kg*/>;
	using kilogram_ratio = std::ratio<   1/*kg*/,     1/*kg*/>; //SI unit
	using ton_ratio      = std::ratio<  10/*ton*/, 9072/*kg*/>;// 1[ton] = 907.2[kg]
	// { time_unit ratio, SI }
	using nanosecond_ratio  = std::ratio<1000'000'000/*ns*/, 1/*s*/>;
	using microsecond_ratio = std::ratio<1000'000/*us*/, 1/*s*/>;
	using millisecond_ratio = std::ratio<1000/*ms*/, 1/*s*/>;
	using second_ratio = std::ratio<1/*s*/,       1/*s*/>; //SI unit
	using minut_ratio  = std::ratio<1/*m*/,      60/*s*/>;
	using hour_ratio   = std::ratio<1/*h*/,    3600/*s*/>;
	using day_ratio    = std::ratio<1/*d*/, 24*3600/*s*/>;
	using year_ratio   = std::ratio<1/*y*/, static_cast<int>(365.25*24*3600)/*s*/>;
	// { temperature_unit ratio, SI  }
	using kelvin_ratio = std::ratio<1, 1>;
	// { electric_current ratio, SI }
	using ampere_ratio = std::ratio<1, 1>;
	// { amount_of_light ratio, SI }
	using candela_ratio = std::ratio<1, 1>;
	// { amount_of_matter ratio, SI }
	using mole_ratio = std::ratio<1, 1>;

	// { length_unit ratio, English }
	using inch_ratio          = std::ratio<3937/*in*/,     100/*m*/>;
	using foot_ratio          = std::ratio<3281/*ft*/,     1000/*m*/>;
	using mile_ratio          = std::ratio<   1/*mi*/,     1609/*m*/>;
	using nautical_mile_ratio = std::ratio<   1/*n mile*/, 1852/*m*/>;
	// { mass_unit ratio, English }
	using pound_mass_ratio = std::ratio<100000/*lbm*/, 45359/*kg*/>;// 1[lbm] = 0.45359[kg]

#pragma endregion

#pragma region unit

#define _DEFINE_UNIT(TYPE_TAG, UNIT_NAME, SYMBOL)  \
	struct UNIT_NAME {                                \
		using unit_category = TYPE_TAG;               \
		using ratio         = UNIT_NAME##_ratio;     \
		constexpr static const char* symbol = SYMBOL; \
	}

	// { length unit }
	_DEFINE_UNIT(length_unit_tag, micrometer, "um");
	_DEFINE_UNIT(length_unit_tag, millimeter, "mm");
	_DEFINE_UNIT(length_unit_tag, centimeter, "cm");
	_DEFINE_UNIT(length_unit_tag, meter,      "m");
	_DEFINE_UNIT(length_unit_tag, kilometer,  "km");
	_DEFINE_UNIT(length_unit_tag, foot,       "ft");
	_DEFINE_UNIT(length_unit_tag, inch,       "in");
	_DEFINE_UNIT(length_unit_tag, nautical_mile, "n mile");
	_DEFINE_UNIT(length_unit_tag, light_year, "ly");
	// { mass unit }
	_DEFINE_UNIT(mass_unit_tag, gram,     "g");
	_DEFINE_UNIT(mass_unit_tag, kilogram, "kg");
	_DEFINE_UNIT(mass_unit_tag, ton,      "ton");
	_DEFINE_UNIT(mass_unit_tag, pound_mass, "lbm");
	// { time unit }
	_DEFINE_UNIT(time_unit_tag, nanosecond,  "ns");
	_DEFINE_UNIT(time_unit_tag, microsecond, "us");
	_DEFINE_UNIT(time_unit_tag, millisecond, "ms");
	_DEFINE_UNIT(time_unit_tag, second, "s");
	_DEFINE_UNIT(time_unit_tag, minut,  "min");
	_DEFINE_UNIT(time_unit_tag, hour,   "h");
	_DEFINE_UNIT(time_unit_tag, day,    "d");
	_DEFINE_UNIT(time_unit_tag, year,   "y");
	// { angle unit }
	_DEFINE_UNIT(angle_unit_tag, radian,     "rad");
	_DEFINE_UNIT(angle_unit_tag, degree,     "deg");
	_DEFINE_UNIT(angle_unit_tag, revolution, "rev");
	// { temperature unit }
	_DEFINE_UNIT(temperature_unit_tag, kelvin, "K");
	// { electric_current unit }
	_DEFINE_UNIT(electric_current_unit_tag, ampere, "A");
	// { amount_of_light unit }
	_DEFINE_UNIT(amount_of_light_unit_tag, candela, "cd");
	// { amount_of_light unit }
	_DEFINE_UNIT(amount_of_matter_unit_tag, mole, "mol");

	// { 1954[y] }
	using SI_length = meter;
	using SI_angle  = radian;
	using SI_mass   = kilogram;
	using SI_time   = second;
	using SI_temperature      = kelvin;
	using SI_electric_current = ampere;
	using SI_amount_of_light  = candela;
	// { 1971[y] }
	using SI_amount_of_matter = mole;

#pragma endregion

#pragma region formula_simplify<_Fml>

	/*<idea>
		(...)/(a+b), ignore
		(...)/(a*b), (...)/a/b
		(...)/(a/b), (...)/a*b
	</idea>*/

	/*<verification and experiment>
		binary_tree_formula can't simplify formula, 
		but linear_formula can do
	</verification and experiment>*/

	template<typename _Fml>
	struct _Formula_reinterpret_divide {
		using type = _Fml;
	};

	template<typename _Fml1, typename _Fml21, typename _Fml22>
	struct _Formula_reinterpret_divide< formula<_Fml1, formula<_Fml21, _Fml22, std::multiplies<>>, std::divides<>> > {
		using type = formula< formula<_Fml1, _Fml21, std::divides<>>, _Fml22, std::divides<> >;
	};

	template<typename _Fml1, typename _Fml21, typename _Fml22>
	struct _Formula_reinterpret_divide< formula<_Fml1, formula<_Fml21, _Fml22, std::divides<>>, std::divides<>> > {
		using type = formula< formula<_Fml1, _Fml21, std::divides<>>, _Fml22, std::multiplies<> >;
	};

	template<typename _Fml>
	using formula_reinterpret_divide = typename _Formula_reinterpret_divide<_Fml>::type;

#pragma endregion

// fundamental quantities --------------------------------------------------

	template<typename _Ty, typename _Ul>
	using length_formula = formula<std::pair<_Ty, _Ul>, void, void>;
	template<typename _Ty, typename _Um>
	using mass_formula = formula<std::pair<_Ty, _Um>, void, void>;
	template<typename _Ty, typename _Ut>
	using time_formula = formula<std::pair<_Ty, _Ut>, void, void>;
	template<typename _Ty, typename _Ua>
	using angle_formula = formula<std::pair<_Ty, _Ua>, void, void>;

	template<typename _Ty, typename _Ul = SI_length>
	using length_t = formula_t<length_formula<_Ty, _Ul>>;
	template<typename _Ty, typename _Um = SI_mass>
	using mass_t   = formula_t<mass_formula<_Ty, _Um>>;
	template<typename _Ty, typename _Ut = SI_time>
	using time_t   = formula_t<time_formula<_Ty, _Ut>>;
	template<typename _Ty, typename _Ua = SI_angle>
	using angle_t  = formula_t<angle_formula<_Ty, _Ua>>;

	template<typename _Ty, typename _Ut>
	using squared_time_formula = formula_multiplies<std::pair<_Ty, _Ut>, std::pair<_Ty, _Ut>>;

#define RADIANS ::calculation::angle_t<_SclTy, ::calculation::radian>
#define DEGREES ::calculation::angle_t<_SclTy, ::calculation::degree>

}// namespace clmagic