//+------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache License 2.0
// ------------------------------------------------------------------------------------+
#pragma once
#include <cassert>
#include <concepts>

#include <cmath>
#include <stdint.h>

using boolean  = bool;

using real32_t  = float;
using real64_t  = double;
using realmax_t = real64_t;

namespace calculation {
	template<typename _Ty> constexpr
	bool is_real_v = std::is_same_v<_Ty, real32_t> | std::is_same_v<_Ty, real64_t>;

	template<typename _Ty> constexpr
	bool is_integer_v = std::is_same_v<_Ty, int8_t> | std::is_same_v<_Ty, int16_t> | std::is_same_v<_Ty, int32_t> | std::is_same_v<_Ty, int64_t>;

	template<typename _Ty> constexpr
	bool is_positive_integer_v = std::is_same_v<_Ty, uint8_t> | std::is_same_v<_Ty, uint16_t> | std::is_same_v<_Ty, uint32_t> | std::is_same_v<_Ty, uint64_t>;

	template<typename _Ty> constexpr
	bool is_number_v = is_real_v<_Ty> | is_integer_v<_Ty> | is_positive_integer_v<_Ty>;

	template<typename _Ty>
	concept real_typename = is_real_v<_Ty>;

	template<typename _Ty>
	concept integer_typename = is_integer_v<_Ty>;

	template<typename _Ty>
	concept positive_integer_typename = is_positive_integer_v<_Ty>;

	template<typename _Ty>
	concept number_typename = is_number_v<_Ty>;
}


/*<decision-tree>
	<!!! Only choose one, or make the same mistake>
	1. using template function
		<ture>
			1. less code
			2. can still overrided
		</true>
		<false>
			1. must replace all of it
			2. consider very more usage enviroment
			3. exist difficult and uncertain to optimize the special enviroment
				_Ty min(_Ty, _Ty)
				_Ty min(const _Ty&, const _Ty&)
				Which match?
		</false>
	2. using override function
		<true>
			1. more certain
			2. can replace only part of it, not all of it
		</true>
		<false>
			1. more code
			2. same meaning has to implement many times
		</false>
</decision-tree>*/

namespace calculation {
	// Basic_math:
	// +
	// -
	// *
	// /
	// %(integral)

#ifndef _REMOVE_CLMAGIC_REAL_NUMERIC
#undef min
	inline int32_t min(const int32_t number1, const int32_t number2) {
		return number1 < number2 ? number1 : number2;
	}
	inline int64_t min(const int64_t number1, const int64_t number2) {
		return number1 < number2 ? number1 : number2;
	}
	inline uint32_t min(const uint32_t number1, const uint32_t number2) {
		return number1 < number2 ? number1 : number2;
	}
	inline uint64_t min(const uint64_t number1, const uint64_t number2) {
		return number1 < number2 ? number1 : number2;
	}
	inline real32_t min(const real32_t number1, const real32_t number2) {
		return number1 < number2 ? number1 : number2;
	}
	inline real64_t min(const real64_t number1, const real64_t number2) {
		return number1 < number2 ? number1 : number2;
	}

#undef max
	inline int32_t max(const int32_t number1, const int32_t number2) {
		return number1 > number2 ? number1 : number2;
	}
	inline int64_t max(const int64_t number1, const int64_t number2) {
		return number1 > number2 ? number1 : number2;
	}
	inline uint32_t max(const uint32_t number1, const uint32_t number2) {
		return number1 > number2 ? number1 : number2;
	}
	inline uint64_t max(const uint64_t number1, const uint64_t number2) {
		return number1 > number2 ? number1 : number2;
	}
	inline real32_t max(const real32_t number1, const real32_t number2) {
		return number1 > number2 ? number1 : number2;
	}
	inline real64_t max(const real64_t number1, const real64_t number2) {
		return number1 > number2 ? number1 : number2;
	}

	inline int32_t clamp(const int32_t number, const int32_t lower, const int32_t upper) {
		return min(max(number, lower), upper);
	}
	inline int64_t clamp(const int64_t number, const int64_t lower, const int64_t upper) {
		return min(max(number, lower), upper);
	}
	inline uint32_t clamp(const uint32_t number, const uint32_t lower, const uint32_t upper) {
		return min(max(number, lower), upper);
	}
	inline uint64_t clamp(const uint64_t number, const uint64_t lower, const uint64_t upper) {
		return min(max(number, lower), upper);
	}
	inline real32_t clamp(const real32_t number, const real32_t lower, const real32_t upper) {
		return min(max(number, lower), upper);
	}
	inline real64_t clamp(const real64_t number, const real64_t lower, const real64_t upper) {
		return min(max(number, lower), upper);
	}

	inline real32_t saturate(const real32_t number) {
		return clamp(number, 0.0F, 1.0F);
	}
	inline real64_t saturate(const real64_t number) {
		return clamp(number, 0.0, 1.0);
	}

	inline int32_t sign(int32_t number) {
		return number < 0 ? -1 : (number & 1);
	}
	inline int64_t sign(int64_t number) {
		return number < 0 ? -1 : (number & 1);
	}
	inline real32_t sign(real32_t number) {
		return number == 0.0F ? 0.0F : (number > 0.0F ? 1.0F : -1.0F);
	}
	inline real64_t sign(real64_t number) {
		return number == 0.0 ? 0.0 : (number > 0.0 ? 1.0 : -1.0);
	}

	using _CSTD abs;
	//inline int32_t abs(int32_t)
	//inline int64_t abs(int64_t number)
	//inline real32_t abs(real32_t number)
	//inline real64_t abs(real64_t number)

	using _CSTD floor;
	//inline real32_t floor(real32_t number)
	//inline real64_t floor(real64_t number);

	using _CSTD ceil;
	//inline real32_t ceil(real32_t number)
	//inline real64_t ceil(real64_t number)

	using _CSTD trunc;
	//inline real32_t trunc(real32_t number)
	//inline real64_t trunc(real64_t number) 

	inline real32_t frac(real32_t number) {
		return number - floor(number);
	}
	inline real64_t frac(real64_t number) {
		return number - floor(number);
	}

	using _CSTD round;
	//inline real32_t round(real32_t number)
	//inline real64_t round(real64_t number)

	inline void remove_error(real32_t& x, real32_t epsilon = std::numeric_limits<real32_t>::epsilon()) {
		real32_t y = round(x);
		if (abs(y - x) < epsilon) {
			x = y;
		}
	}
	inline void remove_error(real64_t& x, real64_t epsilon = std::numeric_limits<real64_t>::epsilon()) {
		real64_t y = round(x);
		if (abs(y - x) < epsilon) {
			x = y;
		}
	}
	inline void remove_error(real32_t* x_array, size_t array_size, real32_t epsilon = std::numeric_limits<real32_t>::epsilon()) {
		real32_t* _First = x_array;
		real32_t* _Last  = x_array + array_size;
		for ( ; _First != _Last; ++_First) {
			remove_error(*_First, epsilon);
		}
	}

	inline int32_t even(int32_t number) {
		return number & 1 ? number + 1 : number;
	}
	inline int64_t even(int64_t number) {
		return number & 1 ? number + 1 : number;
	}

	inline int32_t mod(int32_t number, int32_t divisor) {
		return number % divisor;
	}
	inline int64_t mod(int64_t number, int64_t divisor) {
		return number % divisor;
	}
	inline real32_t mod(real32_t number, real32_t divisor) {
		return _CSTD fmodf(number, divisor);
	}
	inline real64_t mod(real64_t number, real64_t divisor) {
		return _CSTD fmod(number, divisor);
	}

	using _CSTD exp;
	//inline real32_t exp(real32_t number) 
	//inline real64_t exp(real64_t number)

	inline real32_t ln(real32_t number) {
		return _CSTD logf(number);
	}
	inline real64_t ln(real64_t number) {
		return _CSTD log(number);
	}

	using _CSTD pow;
	//inline real32_t pow(real32_t base, real32_t power)
	//inline real64_t pow(real64_t base, real64_t power)
	//inline real32_t pow(real32_t base, int32_t power)
	//inline real64_t pow(real64_t base, int32_t power)

	inline real32_t log(real32_t base, real32_t number) {
		/*<proof>
				pow(base, power) = number
		 log(C,pow(base, power)) = log(C,number)
			log(C,base) * power  = log(number) : Logarithm Power Rule
						  power  = log(C,number) / log(C,base)

				log(base,number) = INV(pow(base,power)) : power is only variable
		</proof>*/
		return _CSTD logf(number) / _CSTD logf(base);
	}
	inline real64_t log(real64_t base, real64_t number) {
		return _CSTD log(number) / _CSTD log(base);
	}

	using _CSTD sqrt;
	//inline real32_t sqrt(real32_t number) 
	//inline real64_t sqrt(real64_t number)

	using _CSTD cbrt;
	//inline real32_t cbrt(real32_t number)
	//inline real64_t cbrt(real64_t number)

	// { Reciprocal square root }
	inline real32_t rsqrt(const real32_t number) {
		return powf(number, -0.5F);
	}
	inline real64_t rsqrt(const real64_t number) {
		return pow(number, -0.5);
	}

	inline int32_t exp2(const int32_t integer) {
		return (1 << integer);
	}
	inline int64_t exp2(const int64_t integer) {
		return (1LL << integer);
	}
	inline uint32_t exp2(const uint32_t integer) {
		return (1U << integer);
	}
	inline uint64_t exp2(const uint64_t integer) {
		return (1ULL << integer);
	}

	// { Factorial number i*(i+1)*(i+2)*...*n, approximate: STIRLING, Stirling::factorial }
	inline int32_t fact(const int32_t n) {
		assert(n < 30);

		int32_t result = 1;
		for (int32_t i = 2; i <= n; result *= i, ++i) {}
		return result;
	}
	inline int64_t fact(const int64_t n) {
		assert(n < 50);
	
		int64_t result = 1;
		for (int64_t i = 2; i <= n; result *= i, ++i) {}
		return result;
	}
	inline real32_t fact(const real32_t n) {
		const uint32_t N = static_cast<uint32_t>(roundf(n));

		real32_t result = 1.0F;
		for (uint32_t i = 2U; i <= N; result *= static_cast<real32_t>(i), ++i) {}
		return result;
	}
	inline real64_t fact(const real64_t n) {
		const uint32_t N = static_cast<uint32_t>(round(n));

		real64_t result = 1.0;
		for (uint32_t i = 2U; i <= N; result *= static_cast<real64_t>(i), ++i) {}
		return result;
	}

	// { linear interpolate a Line }
	inline real32_t lerp(const real32_t left, const real32_t right, const real32_t t) {
		return left + (right - left) * t;
	}
	inline real64_t lerp(const real64_t left, const real64_t right, const real64_t t) {
		return left + (right - left) * t;
	}

	// { value at [lower, upper] remap to [new_lower, new_upper] }
	inline real32_t remap(const real32_t number, const real32_t lower, const real32_t upper, const real32_t new_lower, const real32_t new_upper) {
		return (number - lower)/(upper - lower) * (new_upper - new_lower) + new_lower;
	}
	inline real64_t remap(const real64_t number, const real64_t lower, const real64_t upper, const real64_t new_lower, const real64_t new_upper) {
		return (number - lower) / (upper - lower) * (new_upper - new_lower) + new_lower;
	}

	// { value at [lower, ...) rescale to [lower, upper] }
	real32_t rescale(const real32_t lower, const real32_t upper, const real32_t number) {
		return (number - lower) / (upper - lower);
	}
	real64_t rescale(const real64_t lower, const real64_t upper, const real64_t number) {
		return (number - lower) / (upper - lower);
	}
#endif

#ifndef _REMOVE_CLMAGIC_REAL_TRIGONOMETRIC
	using _CSTD hypot;
	//inline real32_t hypot(real32_t side1, real32_t side2)
	//inline real64_t hypot(real64_t side1, real64_t side2);

	using _CSTD cos;
	//inline real32_t cos(real32_t number)
	//inline real64_t cos(real64_t number) 
	
	using _CSTD sin;
	//inline real32_t sin(real32_t number)
	//inline real64_t sin(real64_t number) 

	using _CSTD tan;
	//inline real32_t tan(real32_t number)
	//inline real64_t tan(real64_t number) 

	using _CSTD acos;
	//inline real32_t acos(real32_t number)
	//inline real64_t acos(real64_t number) 
	
	using _CSTD asin;
	//inline real32_t asin(real32_t number)
	//inline real64_t asin(real64_t number)

	using _CSTD atan;
	//inline real32_t atan(real32_t number)
	//inline real64_t atan(real64_t number) 

	using _CSTD atan2;
	//inline real32_t atan2(real32_t y_num, real32_t x_num)
	//inline real64_t atan2(real64_t y_num, real64_t x_num)

	using _CSTD cosh;
	//inline real32_t cosh(real32_t number)
	//inline real64_t cosh(real64_t number) 

	using _CSTD sinh;
	//inline real32_t sinh(real32_t number)
	//inline real64_t sinh(real64_t number) 

	using _CSTD tanh;
	//inline real32_t tanh(real32_t number)
	//inline real64_t tanh(real64_t number) 

	using _CSTD acosh;
	//inline real32_t acosh(real32_t number)
	//inline real64_t acosh(real64_t number) 

	using _CSTD asinh;
	//inline real32_t asinh(real32_t number)
	//inline real64_t asinh(real64_t number)

	using _CSTD atanh;
	//inline real32_t atanh(real32_t number)
	//inline real64_t atanh(real64_t number) 

	/*<Theorem>
		_Ty a = any_value
		sin(a)*sin(a) + cos(a)*cos(a) = 1

		sin(a + 2Pi*k) = sin(a)
		cos(a + 2Pi*k) = cos(a)
		tan(a + 2Pi*k) = tan(a)
		cot(a + 2Pi*k) = cot(a)

		sin(a + Pi) = -sin(a)
		cos(a + Pi) = -cos(a)
		tan(a + Pi) =  tan(a)
		cot(a + Pi) =  cot(a)

		sin(-a) = -sin(a)
		cos(-a) =  cos(a)
		tan(-a) = -tan(a)
		cot(-a) = -cot(a)

		sin(a + Pi/2) =  cos(a)
		cos(a + Pi/2) = -sin(a)
		tan(a + Pi/2) = -cot(a)
		cot(a + Pi/2) = -tan(a)

		_Ty b = any_value
		sin(a + b) = sin(a)*cos(b) + cos(a)*sin(b)
		sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
		cos(a + b) = cos(a)*cos(b) - sin(a)*sin(b)
		cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
		tan(a + b) = ( tan(a)+tan(b) ) / ( 1 - tan(a)*tan(b) )
		tan(a - b) = ( tan(a)-tan(b) ) / ( 1 + tan(a)*tan(b) )
	</Theorem>*/
	/*<round-error>
		<example> cos(Pi/2) = -0.00000004F,
				  tan(angle / cos(Pi/2))
		</example>
		<avoid> tan2(angle, cos(Pi/2)), if(result < 0) result+=2Pi </avoid>
	<round-error>*/
#endif
}

// Check whether a number is "NAN", and returns TRUE or FALSE
inline boolean ISNAN(real32_t number) {
	return _CSTD isnan(number);
}
inline boolean ISNAN(real64_t number) {
	return _CSTD isnan(number);
}

// Check whether a number is not "INF", and returns TRUE or FALSE
inline boolean ISFINITE(real32_t number) {// number != "INF"
	return _CSTD isfinite(number);
}
inline boolean ISFINITE(real64_t number) {// number != "INF"
	return _CSTD isfinite(number);
}

// Check whether a number is "INF", and returns TRUE or FALSE
inline boolean ISINF(real32_t number) {
	return _CSTD isinf(number);
}
inline boolean ISINF(real64_t number) {
	return _CSTD isinf(number);
}

// Return the sign of a number
inline int32_t SIGN(int32_t number) {
	return number < 0 ? -1 : (number & 1);
}
inline int64_t SIGN(int64_t number) {
	return number < 0 ? -1 : (number & 1);
}
inline real32_t SIGN(real32_t number) {
	return number == 0.0F ? 0.0F : (number > 0.0F ? 1.0F : -1.0F);
}
inline real64_t SIGN(real64_t number) {
	return number == 0.0 ? 0.0 : (number > 0.0 ? 1.0 : -1.0);
}

inline int32_t ABS(int32_t number) {
	return _CSTD abs(number);
}
inline int64_t ABS(int64_t number) {
	return _CSTD llabs(number);
}
inline real32_t ABS(real32_t number) {
	return _CSTD fabsf(number);
}
inline real64_t ABS(real64_t number) {
	return _CSTD fabs(number);
}

inline real32_t FLOOR(real32_t number) {
	return _CSTD floorf(number);
}
inline real64_t FLOOR(real64_t number) {
	return _CSTD floor(number);
}

inline real32_t CEIL(real32_t number) {
	return _CSTD ceilf(number);
}
inline real64_t CEIL(real64_t number) {
	return _CSTD ceil(number);
}

inline real32_t TRUNC(real32_t number) {
	return _CSTD truncf(number);
}
inline real64_t TRUNC(real64_t number) {
	return _CSTD trunc(number);
}

template<typename _RealTy>
inline _RealTy FRAC(_RealTy number) {
	return number - FLOOR(number);
}

inline real32_t ROUND(real32_t number) {
	return _CSTD roundf(number);
}
inline real64_t ROUND(real64_t number) {
	return _CSTD round(number);
}

inline int32_t EVEN(int32_t number) {
	return number & 1 ? number + 1 : number;
}
inline int64_t EVEN(int64_t number) {
	return number & 1 ? number + 1 : number;
}

inline int32_t MOD(int32_t number, int32_t divisor) {
	return number % divisor;
}
inline int64_t MOD(int64_t number, int64_t divisor) {
	return number % divisor;
}
inline real32_t MOD(real32_t number, real32_t divisor) {
	return _CSTD fmodf(number, divisor);
}
inline real64_t MOD(real64_t number, real64_t divisor) {
	return _CSTD fmod(number, divisor);
}

// Large number:

inline real32_t POWER(real32_t base, real32_t power) {
	return static_cast<real32_t>(_CSTD pow(base, power));
}
inline real64_t POWER(real64_t base, real64_t power) {
	return _CSTD pow(base, power);
}
inline real32_t POWER(real32_t number, int32_t power) {// !!!MAYBE OPTIMIZE
	return power == 2 ? (number * number) : POWER(number, static_cast<real32_t>(power));
}
inline real64_t POWER(real64_t number, int32_t power) {
	return power == 2 ? (number * number) : POWER(number, static_cast<real64_t>(power));
}

inline real32_t LOG(real32_t base, real32_t number) {
	/*<proof>
		    POWER(base, power) = number
	 LOG(C,POWER(base, power)) = LOG(C,number)
		  LOG(C,base) * power  = LOG(number) : Logarithm Power Rule
			            power  = LOG(C,number) / LOG(C,base)
					           = INV(POWER(base, power)) : power is only variable 
	</proof>*/
	return _CSTD logf(number) / _CSTD logf(base);
}
inline real64_t LOG(real64_t base, real64_t number) {
	return _CSTD log(number) / _CSTD log(base);
}

inline real32_t EXP(real32_t number) {
	return _CSTD expf(number);
}
inline real64_t EXP(real64_t number) {
	return _CSTD exp(number);
}

inline real32_t LN(real32_t number) {
	return _CSTD logf(number);
}
inline real64_t LN(real64_t number) {
	return _CSTD log(number);
}

inline real32_t SQRT(real32_t number) {
	return _CSTD sqrtf(number);
}
inline real64_t SQRT(real64_t number) {
	return _CSTD sqrt(number);
}

inline real32_t CBRT(real32_t number) {
	return _CSTD cbrtf(number);
}
inline real64_t CBRT(real64_t number) {
	return _CSTD cbrt(number);
}

inline real32_t RSQRT(real32_t number) {
	return powf(number, -0.5F);
}
inline real64_t RSQRT(real64_t number) {
	return pow(number, -0.5);
}

inline int32_t EXP2(int32_t integer) { 
	return (1 << integer);
}
inline int64_t EXP2(int64_t integer) {
	return (1LL << integer);
}
inline uint32_t EXP2(uint32_t integer) {
	return (1U << integer);
}
inline uint64_t EXP2(uint64_t integer) {
	return (1ULL << integer);
}


// Trigonometric:

inline real32_t HYPOT(real32_t side1, real32_t side2) {
	return _CSTD hypotf(side1, side2);
}
inline real64_t HYPOT(real64_t side1, real64_t side2) {
	return _CSTD hypot(side1, side2);
}

inline real32_t SIN(real32_t number) {
	return _CSTD sinf(number);
}
inline real64_t SIN(real64_t number) {
	return _CSTD sin(number);
}

inline real32_t COS(real32_t number) {
	return _CSTD cosf(number);
}
inline real64_t COS(real64_t number) {
	return _CSTD cos(number);
}

inline real32_t TAN(real32_t number) {
	return _CSTD tanf(number);
}
inline real64_t TAN(real64_t number) {
	return _CSTD tan(number);
}

inline real32_t ASIN(real32_t number) {
	return _CSTD asinf(number);
}
inline real64_t ASIN(real64_t number) {
	return _CSTD asin(number);
}

inline real32_t ACOS(real32_t number) {
	return _CSTD acosf(number);
}
inline real64_t ACOS(real64_t number) {
	return _CSTD acos(number);
}

inline real32_t ATAN(real32_t number) {
	return _CSTD atanf(number);
}
inline real64_t ATAN(real64_t number) {
	return _CSTD atan(number);
}

inline real32_t ATAN2(real32_t y_num, real32_t x_num) {
	return _CSTD atan2f(y_num, x_num);
}
inline real64_t ATAN2(real64_t y_num, real64_t x_num) {
	return _CSTD atan2(y_num, x_num);
}


//template<typename _RealTy> inline 
//_RealTy MIN(_RealTy number1, _RealTy number2) {
//	return number1 < number2 ? number1 : number2;
//}
//
//template<typename _RealTy> inline 
//_RealTy MAX(_RealTy number1, _RealTy number2) {
//	return number1 > number2 ? number1 : number2;
//}

template<typename _RealTy, typename _RealTy2> inline 
_RealTy CLAMP(_RealTy number, _RealTy2 lower, _RealTy2 upper) {
	return MIN(MAX(number, static_cast<_RealTy>(lower)), static_cast<_RealTy>(upper));
}

template<typename _RealTy> inline 
_RealTy SATURATE(_RealTy number) {
	return CLAMP(number, static_cast<_RealTy>(0), static_cast<_RealTy>(1));
}

template<typename _RealTy> inline
_RealTy POSITIVE(_RealTy _X) {// max(x, 0)
	return MAX(_X, static_cast<_RealTy>(0));
}

template<typename _RealTy> inline
_RealTy NEGATIVE(_RealTy _X) {// max(x, 0)
	return MIN(_X, static_cast<_RealTy>(0));
}

// { factorial number }
template<typename _Ty>
_Ty FACT(_Ty n, _Ty i = static_cast<_Ty>(1)) {
	_Ty result = i++;
	for ( ; i <= n; ++i) {
		result *= i;
	}

	return result;
}


/*<notation>
	p00  = p{v[0],v[1]} = p{x,y}
	p000 = p{v[0],v[1],v[2]} = p{x,y,z}
	p010 = p{x,y+1,z}
	p001 = p{x,y,z+1}
	and more...
</notation>*/

template<typename _Ty>
_Ty LERP(_Ty p0, _Ty p1, bool is_p1) {
	return (is_p1 ? p1 : p0);
}

// { linear interpolate a Line }
template<typename _Ty1, typename _Ty2>
_Ty1 LERP(_Ty1 p0, _Ty1 p1, _Ty2 t) {
	return p0 + (p1 - p0) * t;
}

// { bilinear interpolate a Plane }
template<typename _Ty1, typename _Ty2> inline
_Ty1 BILERP(_Ty1 p00, _Ty1 p10, _Ty1 p01, _Ty1 p11, _Ty2 x, _Ty2 y) {
	return LERP(LERP(p00, p10, x), LERP(p01, p11, x), y);
}

// { trilinear interpolate a Spatial }
template<typename _Ty1, typename _Ty2> inline
_Ty1 TRILERP(_Ty1 p000, _Ty1 p100, _Ty1 p010, _Ty1 p110, _Ty1 p001, _Ty1 p101, _Ty1 p011, _Ty1 p111, _Ty2 x, _Ty2 y, _Ty2 z) {
	return LERP(BILERP(p000, p100, p010, p110, x, y), BILERP(p001, p101, p011, p111, x, y), z);
}

// { value at [lower, upper] remap to [new_lower, new_upper] }
template<typename _Ty> inline
_Ty REMAP(_Ty value, _Ty lower, _Ty upper, _Ty new_lower, _Ty new_upper) {
	return (value - lower)/(upper - lower) * (new_upper - new_lower) + new_lower;
}

// { value at [lower, ...) rescale to [lower, upper] }
template<typename _Ty> inline
_Ty RESCALE(_Ty lower, _Ty upper, _Ty value) {
	// _Val rescale to [0, 1]
	return (value - lower) / (upper - lower);
}

template<typename _RealTy> inline
_RealTy S_CURVE(_RealTy t) {
	const _RealTy tt  = t * t;
	const _RealTy ttt = tt * t;
	return tt * 3 - ttt * 2;
	/*<No general>
		return ( t * t * (3 - t*2) );
		<Error> (3-t*2): operator-(int,vector) <Error>
	</No general>*/
}

template<typename _RealTy> inline
_RealTy FADE(_RealTy t) {
	const _RealTy ttt = t * t * t;
	const _RealTy tttt = ttt * t;
	const _RealTy ttttt = tttt * t;
	return ttttt * 6 - tttt * 15 + ttt * 10;
}

namespace calculation {

	template<typename _Ty, typename _Ty2>
	bool quadratic(_Ty a, _Ty b, _Ty c, _Ty2& t0, _Ty2& t1) {
		_Ty2 A    = static_cast<_Ty2>(a);
		_Ty2 B    = static_cast<_Ty2>(b);
		_Ty2 C    = static_cast<_Ty2>(c);
		_Ty2 Zero = static_cast<_Ty2>(0);
		_Ty2 Two  = static_cast<_Ty2>(2);
		_Ty2 Four = static_cast<_Ty2>(4);
		_Ty2 discrim      = B*B - Four*A*C;
		_Ty2 sqrt_discrim = sqrt(max(discrim, Zero));
		_Ty2 A2           = A * Two;
		
		t0 = (-B - sqrt_discrim) / A2;
		t1 = (-B + sqrt_discrim) / A2;
		return discrim >= Zero;
		/*<theorem> t0 = [-b - sqrt(b*b-4ac)] / 2a
					t1 = [-b + sqrt(b*b-4ac)] / 2a
		</theorem>*/
	}

	template<typename _Ty, typename _Ty2>
	bool quadratic_accurate(_Ty a, _Ty b, _Ty c, _Ty2& t0, _Ty2& t1) {
		_Ty2 A    = static_cast<_Ty2>(a);
		_Ty2 B    = static_cast<_Ty2>(b);
		_Ty2 C    = static_cast<_Ty2>(c);
		_Ty2 Zero = static_cast<_Ty2>(0);
		_Ty2 Two  = static_cast<_Ty2>(2);
		_Ty2 Four = static_cast<_Ty2>(4);
		_Ty2 discrim	  = B * B - Four * A * C;
		_Ty2 sqrt_discrim = sqrt(max(discrim, Zero));

		if (B > Zero) {
			t0 = (-B - sqrt_discrim) / (Two * A);
			t1 = Two * C / (-B - sqrt_discrim);
		} else {
			t0 = Two * C / (-B + sqrt_discrim);
			t1 = (-B + sqrt_discrim) / (Two * A);
		}

		return discrim >= Zero;
		/*<theorem>
			<reference type="book">
				{ Numerical Analysis, Timothy-Sauer }
			</reference>

			<Exercise 0.4.4> when b < 0
				t0 = [abs(b) - sqrt(b*b-4*a*c)] / (2*a)

				      [-b - sqrt(b*b-4*a*c)] * [-b + sqrt(b*b-4*a*c)]
				   = --------------------------------------------------------       : multiple [-b + sqrt(b*b-4*a*c)]
				                         2*a * [-b + sqrt(b*b-4*a*c)]

					  b*b - (b*b - 4*a*c)	          4*a*c
	               = ----------------------------- = -----------------------------  : eliminate
				      2*a * [-b + sqrt(b*b-4*a*c)]    2*a * [-b + sqrt(b*b-4*a*c)]

				   = 2*c/[-b + sqrt(b*b-4*a*c)]

				t1 = [-b + sqrt(b*b-4*a*c)] / 2a
			<Exercise 25>
		</theorem>*/
	}

	template<typename _Fn, typename _IdxTy>
	auto sum(_Fn _Func, _IdxTy xinit, _IdxTy xend) -> decltype(_Func(xinit)) {
		// accumulate _Func(i) from index region: [xinit, xend]
		assert( xinit <= xend );
		auto _Result = _Func(xinit);
		for (_IdxTy i = xinit+1; i <= xend; ++i) {
			_Result += _Func(i);
		}
		return _Result;
	}

	template<typename _Iter, typename _Fn>
	auto sum_for(_Iter _First, const _Iter _Last, _Fn _Func) {
		// accumulate for [_First, _Last) using transform_op _Func
		assert( _First != _Last);
		auto _Result = _Func(*_First++);
		while (_First != _Last) {
			_Result += _Func(*_First++);
		}
		return _Result;
	}

	template<typename _Container, typename _Fn> inline
	auto sum_of(const _Container& _Cont, _Fn _Func) {
		assert( !_Cont.empty() );
		return sum_for(_Cont.begin(), _Cont.end(), _Func);
	}

	template<typename _Ty, typename _Fn> inline
	auto sum_of(std::initializer_list<_Ty> _Ilist, _Fn _Func) {
		assert( _Ilist.size() != 0 );
		return sum_for(_Ilist.begin(), _Ilist.end(), _Func);
	}

	/*<Theorem>
		sum([](double k){ return k; }, 1, n);
			= n*(n+1)/2;

		sum(pow(n,2), 1, n);
			= n*(n+1)(2*n+1)/6;

		sum(pow(n,3), 1, n);
			= pow( n*(n+1)/2, 2 );

		<Reference>Thomas-Calculus.Section5.2</Reference>
	</Theorem>*/

	template<typename _UnaryOp, typename _RealTy>
	_RealTy derivative(_UnaryOp _Fx, _RealTy t0, _RealTy dt = static_cast<_RealTy>(0.001F)) {
		auto x0 = _Fx(t0);
		auto x1 = _Fx(t0 + dt);
		return (x1 - x0) / dt;
	}

	template<typename _RealTy, typename _UnaryOp>
	_RealTy integrate(_RealTy xinit, _RealTy xend, _UnaryOp _Fx, _RealTy dx = static_cast<_RealTy>(0.001F)) {
		//assert(dx != 0);
		if (dx == 0) {
			return _RealTy(0);
		}
		
		_RealTy x      = xinit;
		_RealTy result = _Fx(x) * dx; x += dx;
		for ( ; x + dx < xend; ) {
			result += _Fx(x) * dx;
			x      += dx;
		}

		if (x < xend) {
			result += _Fx(xend) * (xend - x);
		}
		
		return result;
	}

	template<typename _RealTy, typename _UnaryOp>
	_RealTy integrate_n(_RealTy xinit, _RealTy xend, _UnaryOp _Fx, size_t nstep = 1000) {
		return integrate(xinit, xend, _Fx, (xend - xinit)/static_cast<_RealTy>(nstep));
	}

	//template<typename _RealTy, typename _RealTy2, typename _UnaryOp>
	//std::map<_RealTy,_RealTy> numeric(_RealTy xinit, _RealTy xend, _RealTy2 dx, _UnaryOp _Fx) {
	//	// integrate: [xinit, xend] with _Fx into _Results
	//	assert(dx != 0);
	//	
	//	std::map<_RealTy,_RealTy> _Results;
	//	
	//	_RealTy x = xinit;
	//	for (; x < xend; x += dx) {
	//		_Results.insert_or_assign(x, _Fx(x));
	//	}
	//	_Results.insert_or_assign(xend, _Fx(xend));

	//	return std::move(_Results);
	//}

	// { Newton }
	template<typename _Ty, typename _Fn1, typename _Fn2>
	_Ty tangent_iterate(_Ty x0, _Fn1 F, _Fn2 dF, size_t step) {
		// f: f(x) = 0, cos(x) - x = 0
		_Ty x = x0;
		for (size_t i = 0; i != step; ++i) {
			x = x - F(x) / dF(x);
		}

		return x;
	}

	template<typename _Ty, typename _Fn>
	_Ty fixed_point_iterate(_Ty x0, _Fn G, size_t step) {
		// g: g(x) = x, cos(x) = x
		_Ty x = x0;
		for (size_t i = 0; i != step; ++i) {
			x = G(x);
		}

		return x;
	}

	template<typename _Ty, typename _Fn, typename _Pr>
	_Ty fixed_point_iterate(_Ty x0, _Fn G, _Pr terminate, size_t max_step) {
		assert(max_step != 0);
		_Ty xi      = x0;
		_Ty xi_next = G(xi);
		for (size_t i = 1; !terminate(xi, xi_next) && i != max_step; ++i) {
			xi      = xi_next;
			xi_next = G(xi);
		}

		return xi_next;
	}

	template<typename _Ty, typename _Fn>
	_Ty fixed_point_iterate_convergence_speed(_Ty solved_value, _Fn dG) {
		return abs(dG(solved_value));
	}

}// namespace calculation

namespace Stirling {
	template<typename _RealTy>
	_RealTy factorial(_RealTy n) {
		_RealTy pi = static_cast<_RealTy>(3.141592653589793);
		_RealTy e  = static_cast<_RealTy>(2.718281828459045);
		return sqrt(2*pi * n) * pow(n/e, n);
	}
}// namespace Stirling

namespace Newton {
	template<typename _RealTy = realmax_t, typename _UintTy = uint32_t>
	_RealTy binomial_coefficient(_UintTy n, _UintTy i) {
		// n in [0,infinite), i in [0,n]
		realmax_t numerator   = fact(realmax_t(n));
		realmax_t denominator = fact(realmax_t(n - i)) * fact(realmax_t(i));
		return static_cast<_RealTy>(numerator / denominator);

		//static auto _Lookup_table = std::vector<uintmax_t>{
		//	1,
		//	1, 1,
		//	1, 2, 1,
		//	1, 3, 3, 1,
		//	1, 4, 6, 4, 1,
		//	1, 5, 10, 10, 5, 1
		//};
		//static auto _Lookup_index = std::vector<uint16_t>{
		//	0,  /* 1 elements */
		//	1,  /* 2 elements */
		//	3,  /* 3 elements */
		//	6,  /* 4 elements */
		//	10, /* 5 elements */
		//	15  /* n + 1 elements */
		//};

		//if ( n < _Lookup_index.size() ) {
		//	return static_cast<_IntegralTy>(_Lookup_table[_Lookup_index[n] + i]);
		//} else {	// compute lookup table
		//	for (size_t k = _Lookup_index.size(); k <= n; ++k) {
		//		_Lookup_index.push_back( static_cast<uint16_t>(_Lookup_table.size()) ); /* insert index */
		//		_Lookup_table.insert(_Lookup_table.end(), k + 1, 1);/* insert n+1 elemtns into end */
		//		/*<table>
		//		line   graph: 
		//		Prev   [1, 3, 3, 1]     
		//		Curt   [1, 1, 1, 1, 1]
		//		</table>*/

		//		uintmax_t* _Prev_ptr = &_Lookup_table[_Lookup_index[k - 1]];
		//		uintmax_t* _Curt_ptr = &_Lookup_table[_Lookup_index[k]];
		//		_Curt_ptr += 1; /* first element is 1 */
		//		/*<table>
		//		line graph 
		//			_Prev_ptr
		//				|
		//		Prev [1, 3, 3, 1]
		//				_Curt_ptr
		//					|
		//		Curt [1, 1, 1, 1, 1]
		//		</table>*/

		//		do {
		//			*_Curt_ptr = *_Prev_ptr + *(_Prev_ptr + 1);
		//			++_Prev_ptr; ++_Curt_ptr;
		//		/*<table>
		//		line graph 
		//					_Prev_ptr
		//					|
		//		Prev [1- - -3, 3, 1]
		//			   \	|
  //        				 \  | _Curt_ptr
		//				   \|  |
		//		Curt [1,    4, 1, 1, 1]
		//		</table>*/
		//		} while (*_Prev_ptr != 1);
		//		
		//		/*<table>
		//		line graph
		//				   _Prev_ptr
		//					   |
		//		Prev [1, 3, 3, 1]
		//					  _Curt_ptr
		//						  |
		//		Curt [1, 4, 6, 4, 1]
		//		</table>*/
		//	}
		//	return static_cast<_IntegralTy>(_Lookup_table[_Lookup_index[n] + i]);
		//}
	}

	template<typename _RealTy> inline
	_RealTy binomial(_RealTy A, _RealTy B, size_t n, size_t i) {
		if (i == 0) {
			return pow(A, n);
		} else if (i != n) {
			return pow(A, n-i) * pow(B, i) * binomial_coefficient<_RealTy>(n, i);
		} else {
			return pow(B, n);
		}
	}

	template<typename _RealTy> inline
	_RealTy binomial(_RealTy A, _RealTy B, size_t n) {
		return pow(A + B, n);
	}
}// namespace Newton

namespace Riemann {
	template<typename _RealTy, typename _RealTy2, typename _UnaryOp>
	auto sumL(_RealTy start, _RealTy end, _RealTy2 dx, _UnaryOp _Fx) {
		// sum of [ start, end )
		assert( dx != 0 );
	
		auto _Result = _Fx(start) * dx;
		for (_RealTy i = start+dx; i < end; i += dx) {
			_Result += _Fx(i) * dx;
		}
		return _Result;
	}

	template<typename _RealTy, typename _RealTy2, typename _UnaryOp>
	auto sumR(_RealTy start, _RealTy end, _RealTy2 dx, _UnaryOp _Fx) {
		// sum of ( start, end ]
		assert( dx != 0 );
	
		auto _Result = _Fx(end) * dx;
		for (_RealTy i = end-dx; i > start; i -=dx) {
			_Result += _Fx(i) * dx;
		}
		return _Result;
	}

	template<typename _RealTy, typename _RealTy2, typename _UnaryOp>
	auto sumM(_RealTy start, _RealTy end, _RealTy2 dx, _UnaryOp _Fx) {
		// sum of ( start, end )
		assert( dx != 0 );
		return sumL(_Fx, start + dx/2, end, dx);
	}
}// namespace Riemann

namespace Euler {
	/*template<typename _BinOp, typename _Ty>
	std::map<_Ty, _Ty> numeric(_Ty x0, _Ty y0, _Ty dx, _BinOp _Dy, size_t N) {
		assert(N != 0);
		std::map<_Ty, _Ty> results;
		results.insert_or_assign(x0, y0);
		
		_Ty x = x0;
		_Ty y = y0;
		for (size_t i = 1; i != N; ++i) {
			y += _Dy(x, y) * dx;
			x += dx;
			results.insert_or_assign(x, y);
		}

		return std::move(results);
	}*/
}// namespace Euler

namespace Gaussian {
	template<typename _RealTy>
	_RealTy probability_density(_RealTy x, _RealTy mu, _RealTy sigma) {
		return (        1
			/*-----------------------*/ * exp( (x-mu)/(2*sigma*sigma) )
			/ ( sigma * sqrt(6.283F) ) );
	}

	template<typename _Vec2Ty, typename _RealTy>
	_RealTy probability_density2D(_Vec2Ty _X, _Vec2Ty _Mu, _RealTy _Sigma) {
		return 1 / (6.283F * _Sigma * _Sigma) * exp( -(pow(_X[0]-_Mu[0], 2) + pow(_X[1]-_Mu[1], 2)) / (2*_Sigma*_Sigma) );
	}
}// namespace Gaussian

namespace Bernstein {
	template<typename _RealTy, typename _UintTy> inline
	_RealTy polynomial(_UintTy n, _UintTy i, _RealTy t) {
		return Newton::binomial_coefficient<_RealTy>(n, i) * POWER(t, i) * POWER(1-t, n-i);
	}
}// namespace Bernstein

// { approximate factorial, SQRT(2pi * n) * POWER(n/e, n) }
template<typename _RealTy> inline
_RealTy STIRLING(_RealTy n) {
	return Stirling::factorial(n);
}

// { C(n.i) * POWER(t,i) * POWER(1-t,n-i) }
template<typename _RealTy, typename _UintTy> inline
_RealTy BERNSTEIN(_UintTy n, _UintTy i, _RealTy t) {
	return Bernstein::polynomial(n, i, t);
}