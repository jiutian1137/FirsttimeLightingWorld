//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache License 2.0
//--------------------------------------------------------------------------------------
#pragma once
#include <bitset>
#include <assert.h>

template<size_t _Bits>
std::bitset<_Bits>& operator+=(std::bitset<_Bits>& _Left, std::bitset<_Bits> _Right) {
	auto _Carry = _Left & _Right;
	for (; _Carry.any(); _Carry = _Left & _Right) {
		_Left    = _Left ^ _Right;
		_Right   = _Carry << 1;
	}

	return (_Left ^= _Right);
	/*
		00110101                      
	+ 00000001                    
	= 00110110                     
	forward_bit = A & B
	forward_bit <<= 1
	result = A | forward_bit

		10101111 A
	+ 00110110 B
	= 11100101 Result
		 
		10011001 A ^ B        = A2
	+ 01001100 (A & B) << 1 = B2
	= 11100101
		  
		11010101 A2 ^ B2        = A3
	+ 00010000 (A2 & B2) << 1 = B3
	= 11100101

		11000101 A3 ^ B3        = A4
	+ 00100000 (A3 & B3) << 1 = B4
	= 11100101

		00000000 A4 & B4
		11100101 A4 ^ B4 = Result
	*/
}

template<size_t _Bits>
std::bitset<_Bits> operator+(const std::bitset<_Bits>& _Left, const std::bitset<_Bits>& _Right) noexcept {
	auto _Left_copy  = std::bitset<_Bits>(_Left);
	     _Left_copy += _Right;
	return _Left_copy;
}

template<size_t _Bits>
std::bitset<_Bits>& operator-=(std::bitset<_Bits>& _Left, std::bitset<_Bits> _Right) {
	for ( ; ; ) {
		_Left  = _Left ^ _Right;
		_Right = _Left & _Right;// (_Left^_Right)&_Right
		if ( _Right.none() ) {
			break;
		}
		_Right = _Right << 1;
	}

	return (_Left);
	/*
		10100111 A
	- 10001111 B
	= 00011000 Result
		
		00101000 A ^ B         = A1
	- 00010000 (A1 & B) << 1 = B1
	= 00011000

		00111000 A1 ^ B1        = A2
	- 00100000 (A2 & B1) << 1 = B2
	= 00011000

		00011000 A2 ^ B2 = A3 = Result
		00000000 A3 & B2
	*/
}

template<size_t _Bits>
std::bitset<_Bits> operator-(const std::bitset<_Bits>& _Left, const std::bitset<_Bits>& _Right) {
	auto _Left_copy  = std::bitset<_Bits>(_Left);
		    _Left_copy -= _Right;
	return _Left_copy;
}

namespace calculation {

	// Memory-Operation
	constexpr size_t floor(size_t _Val, size_t _Bound) {
		return _Val & (~(_Bound - 1));
	}

	// Memory-Operation
	constexpr size_t ceil(size_t _Val, size_t _Bound) {
		return (_Val + (_Bound - 1)) & (~(_Bound - 1));
	}

	// Memory-Operation
	constexpr size_t trunc(size_t _Val, size_t _Bound) {
		return _Val & (_Bound - 1);
	}


	template<typename _BitTy> inline
	constexpr _BitTy mask_at(size_t _Nx) {// pow(2, _Nx)
		return (static_cast<_BitTy>(1) << _Nx);
	}

	template<typename _BitTy> inline
	constexpr _BitTy mask_until(size_t _Nx) {// [0, _Nx]
		auto _Tmp = mask_at<_BitTy>(_Nx);
		return ( _Tmp | (_Tmp - static_cast<_BitTy>(1)) );
	}

	template<typename _BitTy> inline
	constexpr _BitTy mask_n(size_t _Nx) {// _Nx ones in the mask, or [0, _Nx)
		return ( _Nx == 0 ? _BitTy(0) : mask_until<_BitTy>(_Nx - 1) );
	}

	template<typename _BitTy> inline
	constexpr _BitTy mask_for(size_t _Off, size_t _Nx) {// [_Off, _Nx)
		return ( _Off >= _Nx ? _BitTy(0) : (mask_n<_BitTy>(_Nx - _Off) << _Off) );
		// not use assert(...), because can happen (_Off < _Nx)
	}

	template<typename _BitTy> inline
	size_t count_1(_BitTy _X) {// number of equal 1 
		const auto _One   = static_cast<_BitTy>(_X);
		auto       _Count = _BitTy(0);
		while ( _X ) {
			_Count += _X & _One;
			_X >>= 1;
		}
		return static_cast<size_t>(_Count);
	}

	template<typename _BitTy> inline
	size_t count_0(_BitTy _X) {// number of equal 0
		return (sizeof(_BitTy) * 8 - count_1(_X));
	}

	template<typename _Ty, typename _BitTy> inline
	_BitTy mask_use(const _Ty& _Val, _BitTy _Mask) {// _Val & _Mask
		assert(sizeof(_Ty) == sizeof(_BitTy));
		const auto _Valbit = *reinterpret_cast<const _BitTy*>(&_Val);
		return (_Valbit & _Mask);
	}

	template<typename _Ty, typename _BitTy> inline
	_BitTy mask_use(const _Ty& _Val, _BitTy _Mask, size_t _Off) {// (_Val & _Mask) >> _Off
		assert(sizeof(_Ty) == sizeof(_BitTy));
		const auto _Valbit = *reinterpret_cast<const _BitTy*>(&_Val);
		return ((_Valbit & _Mask) >> _Off);
	}


	template<size_t _Bits> inline
	int32_t make_int32(const std::bitset<_Bits>& _Bitval) {
		static_assert(_Bits >= 32, "calculation::Error");
		return *( reinterpret_cast<const int32_t*>(&_Bitval) );
	}
	
	template<size_t _Bits> inline
	intmax_t make_intmax(const std::bitset<_Bits>& _Bitval) {
		if _CONSTEXPR_IF(_Bits >= 64) {
			return *( reinterpret_cast<const intmax_t*>(&_Bitval) );
		} else {
			return static_cast<intmax_t>(make_int32(_Bitval));
		}
	}

	template<size_t _Bits> inline
	uint32_t make_uint32(const std::bitset<_Bits>& _Bitval) {
		return (_Bitval.to_ulong());
	}

	template<size_t _Bits> inline
	uint64_t make_uint64(const std::bitset<_Bits>& _Bitval) {
		return (_Bitval.to_ullong());
	}

	template<size_t _Bits> inline
	uintmax_t make_uintmax(const std::bitset<_Bits>& _Bitval) {
		return (_Bitval.to_ullong());
	}


	template<typename _Ty, size_t _En, size_t _Mn>
	class floating_point {
		static_assert((1 + _En + _Mn) <= sizeof(_Ty) * 8, "error floating type");

	public:
		constexpr static size_t _Bits                = sizeof(_Ty) * 8;
		constexpr static size_t _Off_significant_bit = 0;
		constexpr static size_t _Off_exponent_bit    = _Off_significant_bit + _Mn;
		constexpr static size_t _Off_sign_bit        = _Off_exponent_bit + _En;

		static std::bitset<_Bits> mask_sign() {// Get sign bit: mask_use(_Float, ???::mask_sign()) >> (_Bits - 1)
			return mask_for<std::bitset<_Bits>>(_Off_sign_bit, _Bits);
		}

		static std::bitset<_Bits> mask_exponent() {// Get sign bit: mask_use(_Float, ???::mask_exponent()) >> (_Bits - _En)
			return mask_for<std::bitset<_Bits>>(_Off_exponent_bit, _Off_sign_bit);
		}

		static std::bitset<_Bits> mask_significant() {// Get sign bit: mask_use(_Float, ???::mask_significant())
			return mask_for<std::bitset<_Bits>>(0, _Off_exponent_bit);
		}

		constexpr floating_point() : _Myval{ 0 } { }

		constexpr floating_point(_Ty _Val) : _Myval(_Val) { }

		constexpr floating_point(std::bitset<_Bits> _Bitval) : _Myval(*reinterpret_cast<_Ty*>(&_Bitval)){ }

		constexpr operator _Ty() const { return (_Myval); }

		template<size_t _Ubits = _Bits>
		std::bitset<_Ubits> sign() const {
			static_assert(_Ubits <= _Bits, "error _Bits");
			const auto _Sign = mask_use(*this, mask_sign(), _Off_sign_bit);
			return ( *reinterpret_cast<const std::bitset<_Ubits>*>(&_Sign) );
		}
		
		template<size_t _Ubits = _Bits>
		std::bitset<_Ubits> exponent() const {
			static_assert(_Ubits <= _Bits, "error _Bits");
			const auto _Exp = mask_use(*this, mask_exponent(), _Off_exponent_bit);
			return ( *reinterpret_cast<const std::bitset<_Ubits>*>(&_Exp) );
		}
		
		template<size_t _Ubits = _Bits>
		std::bitset<_Ubits> significant() const {
			static_assert(_Ubits <= _Bits, "error _Bits");
			const auto _Sign = mask_use(*this, mask_significant(), _Off_significant_bit);
			return ( *reinterpret_cast<const std::bitset<_Ubits>*>(&_Sign) );
		}

		template<size_t _Ubits = _Bits>
		static std::bitset<_Ubits> exponent_bias() {// pow(2, _En-1) - 1
			return mask_n<std::bitset<_Ubits>>(_En - 1);
		}

		std::bitset<_Bits> floor() const {
			auto _Src  = *reinterpret_cast<const std::bitset<_Bits>*>(&_Myval);
			auto _Bias = mask_at<std::bitset<_Bits>>(_En - 1) - std::bitset<_Bits>(1);
			auto _Exp  = this->exponent<_Bits>() -_Bias;
			auto _Tmp  = _Exp.to_ulong();
			auto _Expi = *((long*)(&_Tmp));
			if (_Expi < 0) {
				return (0); 
			} else {
				auto _Mask = mask_n<std::bitset<_Bits>>( _Expi > _Mn ? 0 : _Mn - _Expi );
				return ( (_Src) & (~_Mask) );
			}
		}

	private:
		_Ty _Myval;
	};

	using IEEE754_float  = floating_point<float, 8, 23>;// (-1)^S * (1+Fraction) * 2^(Exponent-Bias), Bias = 127
	using IEEE754_double = floating_point<double, 11, 52>;// (-1)^S * (1+Fraction) * 2^(Exponent-Bias), Bias = 1023

}// namespace clmagic