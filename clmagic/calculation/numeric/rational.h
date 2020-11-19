//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache Licene 2.0
//--------------------------------------------------------------------------------------
#pragma once
#ifndef clmagic_calculation_numeric_RATIONAL_h_
#define clmagic_calculation_numeric_RATIONAL_h_
#include "bitset.h"
#include <ratio>
#include <string>

namespace calculation {

	template<typename _Ti>
	_Ti _Gcdx(_Ti _Ax, _Ti _Bx) {// computes greatest common divisor of _Ax and _Bx
		if (_Bx == 0) {
			return (_Ax);
		} else {
			return _Gcdx(_Bx, _Ax % _Bx);
		}
	}

	template<typename _Ti> inline
	_Ti gcd(_Ti _Ax, _Ti _Bx) {// computes greatest common divisor of _Ax and _Bx
		if (_Ax == 0 && _Bx == 0) {
			return (1);
		}

		if (_Bx > _Ax) {
			std::swap(_Ax, _Bx);
		}
		return _Gcdx(_Ax < 0 ? -_Ax : _Ax, _Bx < 0 ? -_Bx : _Bx);
	}


	template<size_t _Exp, typename _Ti>
	_Ti _Gcdx_approx(_Ti _Ax, _Ti _Bx) {// computes greatest common divisor of _Ax and _Bx
		if ((_Ax >> _Exp) >= _Bx) {
			return (_Ax);
		} else {
			return _Gcdx_approx<_Exp>(_Bx, _Ax % _Bx);
		}
		/*
		Ax     Bx
		16667 10000
		10000 6667
		6667  3333
		3333  1

		16666 10000
		10000 6666
		6666  3334
		3334  3332
		3332  2
		This is simple test
		Result is _Ax when (_Ax modulus _Bx) is very small
		But _Bx must be large than (_Ax mudulus _Bx)
		And not error, because final-divides is <trunc (_Ax modulus _Bx)>
		*/
	}

	template<size_t _Exp, typename _Ti> inline 
	_Ti gcd_approx(_Ti _Ax, _Ti _Bx) {
		if (_Ax == 0 && _Bx == 0) {
			return (1);
		}

		if (_Bx > _Ax) {
			std::swap(_Ax, _Bx);
		}
		return _Gcdx_approx<_Exp>(_Ax < 0 ? -_Ax : _Ax, _Bx < 0 ? -_Bx : _Bx);
	}
	

	template<typename _Ti>
	struct rational {
		static_assert(std::_Is_any_of_v<_Ti, int32_t, int64_t, intmax_t>, "-> rational<_Ti>");

		using integral_type = _Ti;

		constexpr rational()
			: _Num(0), _Den(0) {}
	
		explicit constexpr rational(integral_type _Integral)
			: _Num(_Integral), _Den(1) {}
		
		constexpr rational(integral_type _Nx, integral_type _Dx)
			: _Num(_Nx), _Den(_Dx) {}
		
		constexpr rational(std::initializer_list<integral_type> _Ilist) {
			assert(_Ilist.size() == 2);
			*this = _Ilist;
		}
		
		template <intmax_t _Nx, intmax_t _Dx>
		constexpr rational(std::ratio<_Nx, _Dx> _Ratio)
			: _Num(static_cast<_Ti>(_Nx)), _Den(static_cast<_Ti>(_Dx)) {}

		rational& operator=(const rational& _Right) = default;
		
		rational& operator=(integral_type _Integral) {
			_Num = _Integral;
			_Den = 1; 
			return *this;
		}
		
		rational& operator=(std::initializer_list<integral_type> _Ilist) {
			assert(_Ilist.size() == 2);
			auto _First = _Ilist.begin();
			this->_Num  = *_First++;
			this->_Den  = *_First;
			return *this;
		}
		
		template<typename _Fp = float>
		constexpr _Fp to_floating() const {
			return static_cast<_Fp>(_Num) / static_cast<_Fp>(_Den);
		}

		std::string to_string() const {
			return (std::to_string(_Num) + "/" + std::to_string(_Den));
		}

		bool operator==(const rational& _Right) const {
			return this->_Num * _Right._Den == _Right._Num * this->_Den;
		}
		bool operator!=(const rational& _Right) const {
			return !((*this) == _Right);
		}
		bool operator< (const rational& _Right) const {
			return std::_Ratio_less(this->_Num, this->_Den, _Right._Num, _Right._Den);
		}
		bool operator> (const rational& _Right) const {
			return (_Right < (*this));
		}
		bool operator>=(const rational& _Right) const {
			return !((*this) < _Right);
		}
		bool operator<=(const rational& _Right) const {
			return !((*this) > _Right);
		}
		
		rational& operator++() {
			_Num += _Den;
			return *this;
		}
		rational  operator++(int) {
			auto _Prev = *this;
			++(*this);
			return _Prev;
		}
		
		rational operator-() const {
			return rational(-this->_Num, this->_Den);
		}
		rational  operator+ (const rational& _Right) const {
			if (this->_Den == 0) {
				return _Right;
			} else if(_Right._Den == 0){
				return *this;
			} else {
				const auto _N1 = this->_Num;
				const auto _D1 = this->_Den;
				const auto _N2 = _Right._Num;
				const auto _D2 = _Right._Den;
				const auto _Gx = gcd(_D1, _D2);
				return rational(_N1 * (_D2/_Gx) + _N2 * (_D1/_Gx), 
								_D1 * (_D2/_Gx));
				/*
				 N1*D2 + N2*D1
				--------------
				   D1*D2
				*/
			}
		}
		rational  operator- (const rational& _Right) const {
			const auto _N1 = this->_Num;
			const auto _D1 = this->_Den;
			const auto _N2 = _Right._Num;
			const auto _D2 = _Right._Den;
			const auto _Gx = gcd(_D1, _D2);
			return rational(_N1 * (_D2/_Gx) - _N2 * (_D1 /_Gx), 
							_D1 * (_D2/_Gx));
			/*
			 N1*D2 - N2*D1
			---------------
			   D1*D2
			*/
		}
		rational  operator* (const rational& _Right) const {
			const auto _N1 = this->_Num;
			const auto _D1 = this->_Den;
			const auto _N2 = _Right._Num;
			const auto _D2 = _Right._Den;
			const auto _Gx = gcd(_N1, _D2);
			const auto _Gy = gcd(_N2, _D1);
			return rational((_N1 / _Gx) * (_N2 / _Gy),
							(_D1 / _Gy) * (_D2 / _Gx));
			/*
			 N1*N2
			-------
			 D1*D2
			*/
		}
		rational  operator/ (const rational& _Right) const {
			return (*this) * rational(_Right._Den, _Right._Num);
			/*
			 N1*D2
			-------
			 D1*N2
			 */
		}
		rational  operator% (const rational& _Right) const {
			const auto _N1 = this->_Num;
			const auto _D1 = this->_Den;
			const auto _N2 = _Right._Num;
			const auto _D2 = _Right._Den;
			/*const auto _Gx = gcd(_N1, _D2);
			const auto _Gy = gcd(_N2, _D1);*/
			return rational((_N1*_D2) % (_N2*_D1),
							_D1 * _D2);
			/*
			 N1*D2 % N2*D1 
			--------------
			    D1*D2
			*/
		}
		rational& operator+=(const rational& _Right) {
			(*this) = (*this) + _Right;
			return *this;
		}
		rational& operator-=(const rational& _Right) {
			(*this) = (*this) - _Right;
			return *this;
		}
		rational& operator*=(const rational& _Right) {
			(*this) = (*this) * _Right;
			return *this;
		}
		rational& operator/=(const rational& _Right) {
			(*this) = (*this) / _Right;
			return *this;
		}
		rational& operator%=(const rational& _Right) {
			(*this) = (*this) % _Right;
			return *this;
		}
		
		rational  operator+ (integral_type _Integral) const {
			const auto _Nx = this->_Num;
			const auto _Dx = this->_Den;
			return rational(_Nx + _Dx * _Integral, _Dx);
		}
		rational  operator- (integral_type _Integral) const {
			const auto _Nx = this->_Num;
			const auto _Dx = this->_Den;
			return rational(_Nx - _Dx * _Integral, _Dx);
		}
		rational  operator* (integral_type _Integral) const {
			const auto _Nx = this->_Num;
			const auto _Dx = this->_Den;
			const auto _Gy = gcd(_Integral, _Dx);
			return rational(_Nx * (_Integral / _Gy),
							_Dx / _Gy);
		}
		rational  operator/ (integral_type _Integral) const {
			const auto _Nx = this->_Num;
			const auto _Dx = this->_Den;
			const auto _Gx = gcd(_Nx, _Integral);
			return rational(_Nx / _Gx,
							_Dx * (_Integral / _Gx));
		}
		rational  operator% (integral_type _Integral) const {
			const auto _Nx = this->_Num;
			const auto _Dx = this->_Den;
			return rational(_Nx % (_Integral*_Dx), _Dx);
			/*
			 _Nx % (_Int*_Dx)
			----------------
				_Dx
			*/
		}
		rational& operator+=(integral_type _Integral) {
			(*this) = (*this) + _Integral;
			return *this;
		}
		rational& operator-=(integral_type _Integral) {
			(*this) = (*this) - _Integral;
			return *this;
		}
		rational& operator*=(integral_type _Integral) {
			(*this) = (*this) * _Integral;
			return *this;
		}
		rational& operator/=(integral_type _Integral) {
			(*this) = (*this) / _Integral;
			return *this;
		}
		rational& operator%=(integral_type _Integral) {
			(*this) = (*this) % _Integral;
			return *this;
		}
		friend rational operator+(_Ti _Integral, const rational& _R1) {
			return _R1 + _Integral;
		}
		friend rational operator-(_Ti _Integral, const rational& _R1) {
			return (-_R1 + _Integral);
		}
		friend rational operator*(_Ti _Integral, const rational& _R1) {
			return _R1 * _Integral;
		}
		friend rational operator/(_Ti _Integral, const rational& _R1) {
			return _Integral * rational(_R1._Den, _R1._Num);
		}

		friend std::ostream& operator<<(std::ostream& _Ostr, const rational& _R1) {
			return (_Ostr << _R1.to_string());
		}

		void reduce() {
			const auto _Gx = gcd(this->_Num, this->_Den);
			this->_Num /= _Gx;
			this->_Den /= _Gx;
		}

		_Ti _Num;
		_Ti _Den;
	};

	template<typename _Ti> inline
	rational<_Ti> inverse(const rational<_Ti>& _R1) {
		return rational<_Ti>(_R1._Den, _R1._Num);
	}

	template<typename _Ti> inline
	std::string to_string(const rational<_Ti>& _R1) {
		return _R1.to_string();
	}

	template<typename _Ti, typename _Ty, size_t _En, size_t _Mn>
	rational<_Ti> cvtfloating_rational(floating_point<_Ty, _En, _Mn> _Fp) {
		/*<describ>
			<formula>
				<source>0 00000001 10101101011010110101000</source>
				<first>
					<tips> IEEE754-floating-formula: (-1)^S * (1+Fraction) * 2^(Exponent-Bias) </tips>
					
					(-1)^0 * (1 + 0.10101101011010110101000) * 2^(00000001 - Bias)
					= 1 * 1.10101101011010110101000 * pow(2, _Exp)
					= 1 * 0.110101101011010110101000 * pow(2, _Exp)
					= 1 * 110101101011010110101000/pow(2,_Mn) * pow(2, _Exp)
				</first>				
				<second>
					<tips> pow(2, X) = (1 << X) </tips>

					_Nx     110101101011010110101000
					---- = -------------------------- * ( 1 << _Exp )
					_Dx           1 << _Mn

					   110101101011010110101000 << 1
					----------------------------------- * ( 1 << (_Exp - 1) )
							 1 << _Mn

					   110101101011010110101000
					----------------------------------- * ( 1 << (_Exp - 1) )
							 1 << (_Mn-1)

					   110101101011010110101000
					----------------------------------- * (1 << 0)
							 1 << (_Mn - _Exp)
				</second>
			</formula>
		</describ>*/
		using _BitTy = decltype(_Fp.exponent());
		auto       _Exp      = make_int32(_Fp.exponent() - _Fp.exponent_bias());
		const auto _Hide     = mask_at<_BitTy>(_Mn);
		auto       _Mantissa = _Fp.significant() | _Hide;
		intmax_t _Num = 0;
		intmax_t _Den = 0;
		if (int32_t(_Mn) >= _Exp ) {// _Mn - _Exp >= 0
			_Num = make_uintmax(_Mantissa);
			_Den = intmax_t(1) << (int32_t(_Mn) - _Exp);
		} else {
			_Num = make_uintmax(_Mantissa) << (_Exp - int32_t(_Mn));
			_Den = 1;
		}

		if (make_int32(_Fp.sign()) == 1) {
			_Num = -_Num;
		}

		const auto _Gx = gcd_approx<_En*2>(_Num, _Den);
		//const auto _Gx = gcd(_Num, _Den);
		return rational<_Ti>(static_cast<_Ti>(_Num / _Gx), static_cast<_Ti>(_Den / _Gx));
	}

	template<typename _Ti> inline
	rational<_Ti> cvtfloating_rational(float _Fp) {
		return cvtfloating_rational<_Ti>(IEEE754_float(_Fp));
	}
	
	template<typename _Ti> inline
	rational<_Ti> cvtfloating_rational(double _Fp) {
		return cvtfloating_rational<_Ti>(IEEE754_double(_Fp));
	}
}// namespace clmagic

#endif