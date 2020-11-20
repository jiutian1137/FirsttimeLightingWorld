//+------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache License 2.0
// ------------------------------------------------------------------------------------+
#pragma once
#ifndef clmagic_calculation_lapack_VECTOR_h_
#define clmagic_calculation_lapack_VECTOR_h_
#include "block.h"
#include "../numeric/real.h"

namespace calculation{
	// { vector, static size }
	template<typename _Ty, size_t _Size, typename _Traits = block_traits<_Ty>>
	struct vector {
		static_assert(_Size != 0, 
			"calculation::static_assert::empty_vector");
		//static_assert(std::is_scalar_v<_Ty>, 
		//	"clmagic::static_assert::strict_math_vector");
		
		static constexpr size_t scalar_size = _Size;
		using vector_category = static_aligned_vector_tag;

		using traits_type = _Traits;
		using scalar_type = typename _Traits::scalar_type;
		using block_type  = typename _Traits::block_type;

		using iterator       = scalar_type*;
		using const_iterator = const scalar_type*;

	private:
		using real = scalar_type;

	public:
		constexpr size_t size() const { 
			return scalar_size;
		}
		
		constexpr bool aligned() const { 
			return (scalar_size % traits_type::size()) == 0;
		}
		
		constexpr bool empty() const { 
			return false; 
		}
		
		const scalar_type* data() const {
			return _Mydata; 
		}
		
		scalar_type* data() {
			return _Mydata;
		}

		const scalar_type* begin() const {
			return _Mydata; 
		}
		
		scalar_type* begin() {
			return _Mydata; 
		}
		
		const scalar_type* end() const {
			return _Mydata + size(); 
		}
		
		scalar_type* end() {
			return _Mydata + size(); 
		}
		
		const scalar_type& at(size_t _Pos) const {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}
		
		scalar_type& at(size_t _Pos) {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}
		
		const scalar_type& operator[](size_t _Pos) const {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}
		
		scalar_type& operator[](size_t _Pos) {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}

		void fill(const scalar_type& _Val) {
			std::fill(begin(), end(), _Val);
		}
		
		template<typename _Iter>
		void assign(_Iter _First, _Iter _Last) {
			assert(std::distance(_First, _Last) <= 
				std::_Iter_diff_t<_Iter>(this->size()));
			std::copy(_First, _Last, _Mydata);
		}

		template<typename _UnaryOp>
		void assign(vector v1, _UnaryOp f) {
			if constexpr (vector::scalar_size == 2) 
				{
				_Mydata[0] = f(v1[0]);
				_Mydata[1] = f(v1[1]);
				}
			else if constexpr (vector::scalar_size == 3) 
				{
				_Mydata[0] = f(v1[0]);
				_Mydata[1] = f(v1[1]);
				_Mydata[2] = f(v1[2]);
				}
			else if constexpr (vector::scalar_size == 4) 
				{
				_Mydata[0] = f(v1[0]);
				_Mydata[1] = f(v1[1]);
				_Mydata[2] = f(v1[2]);
				_Mydata[3] = f(v1[3]);
				} 
			else 
				{
				std::transform(v1.begin(), v1.end(), _Mydata, f);
				}
		}

		template<typename _BinOp>
		void assign(vector v1, vector v2, _BinOp f) {
			if constexpr (vector::scalar_size == 2) 
				{
				_Mydata[0] = f(v1[0], v2[0]);
				_Mydata[1] = f(v1[1], v2[1]);
				}
			else if constexpr (vector::scalar_size == 3) 
				{
				_Mydata[0] = f(v1[0], v2[0]);
				_Mydata[1] = f(v1[1], v2[1]);
				_Mydata[2] = f(v1[2], v2[2]);
				}
			else if constexpr (vector::scalar_size == 4) 
				{
				_Mydata[0] = f(v1[0], v2[0]);
				_Mydata[1] = f(v1[1], v2[1]);
				_Mydata[2] = f(v1[2], v2[2]);
				_Mydata[3] = f(v1[3], v2[3]);
				} 
			else 
				{
				std::transform(v1.begin(), v1.end(), v2.begin(),
					_Mydata, f);
				}
		}

		template<typename _UnaryOp> static 
		vector calculate(vector v1, _UnaryOp f) {
			// make vector from v1 with f(x)
			if constexpr (vector::scalar_size == 2) 
				{
				return vector{ f(v1[0]), f(v1[1]) };
				}
			else if constexpr (vector::scalar_size == 3) 
				{
				return vector{ f(v1[0]), f(v1[1]), f(v1[2]) };
				}
			else if constexpr (vector::scalar_size == 4) 
				{
				return vector{ f(v1[0]), f(v1[1]), f(v1[2]), f(v1[3]) };
				} 
			else 
				{
				vector v2;
				std::transform(v1.begin(), v1.end(), v2.begin(), f);
				return std::move(v2);
				}
		}
		
		template<typename _BinOp> static
		vector calculate(vector v1, vector v2, _BinOp f) {
			// make vector from v1 and v2 with f(x,y)
			if constexpr (vector::scalar_size == 2)
				{
				return vector{ f(v1[0],v2[0]), f(v1[1],v2[1]) };
				} 
			else if constexpr (vector::scalar_size == 3)
				{
				return vector{ f(v1[0],v2[0]), f(v1[1],v2[1]), f(v1[2],v2[2]) };
				} 
			else if constexpr (vector::scalar_size == 4)
				{
				return vector{ f(v1[0],v2[0]), f(v1[1],v2[1]), f(v1[2],v2[2]), f(v1[3],v2[3]) };
				} 
			else 
				{
				vector v3;
				std::transform(v1.begin(), v1.end(), v2.begin(), v3.begin(), f);
				return std::move(v3);
				}
		}

		vector operator-() const {
			return calculate(*this, std::negate<>());
		}

		vector operator+(vector _Right) const {
			return calculate(*this, _Right, std::plus<>());
		}

		vector operator-(vector _Right) const {
			return calculate(*this, _Right, std::minus<>());
		}

		vector operator*(vector _Right) const {
			return calculate(*this, _Right, std::multiplies<>());
		}

		vector operator/(vector _Right) const {
			return calculate(*this, _Right, std::divides<>());
		}

		vector operator%(vector _Right) const {
			return calculate(*this, _Right, std::modulus<>());
		}

		vector operator==(vector _Right) const {
			return calculate(*this, _Right, std::equal_to<>());
		}

		vector operator!=(vector _Right) const {
			return calculate(*this, _Right, std::not_equal_to<>());
		}

		vector operator>(vector _Right) const {
			return calculate(*this, _Right, std::greater<>());
		}

		vector operator>=(vector _Right) const {
			return calculate(*this, _Right, std::greater_equal<>());
		}

		vector operator<(vector _Right) const {
			return calculate(*this, _Right, std::less<>());
		}

		vector operator<=(vector _Right) const {
			return calculate(*this, _Right, std::less_equal<>());
		}

		vector operator+(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return _Left + _Right; });
		}

		vector operator-(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return _Left - _Right; });
		}

		vector operator*(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return _Left * _Right; });
		}

		vector operator/(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return _Left / _Right; });
		}

		vector operator%(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return _Left % _Right; });
		}

		vector operator==(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return static_cast<real>(_Left == _Right); });
		}

		vector operator!=(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return static_cast<real>(_Left != _Right); });
		}

		vector operator>(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return static_cast<real>(_Left > _Right); });
		}

		vector operator>=(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return static_cast<real>(_Left >= _Right); });
		}

		vector operator<(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return static_cast<real>(_Left < _Right); });
		}

		vector operator<=(real _Right) const {
			return calculate(*this, [_Right](real _Left) {
				return static_cast<real>(_Left <= _Right); });
		}

		vector& operator+=(vector _Right) {
			this->assign(*this, _Right, std::plus<>());
			return *this;
		}

		vector& operator-=(vector _Right) {
			this->assign(*this, _Right, std::minus<>());
			return *this;
		}

		vector& operator*=(vector _Right) {
			this->assign(*this, _Right, std::multiplies<>());
			return *this;
		}

		vector& operator/=(vector _Right) {
			this->assign(*this, _Right, std::divides<>());
			return *this;
		}

		vector& operator%=(vector _Right) {
			this->assign(*this, _Right, std::modulus<>());
			return *this;
		}

		vector& operator+=(real _Right) {
			this->assign(*this, [_Right](real _Left) {
				return _Left + _Right; });
			return *this;
		}

		vector& operator-=(real _Right) {
			this->assign(*this, [_Right](real _Left) {
				return _Left - _Right; });
			return *this;
		}

		vector& operator*=(real _Right) {
			this->assign(*this, [_Right](real _Left) {
				return _Left * _Right; });
			return *this;
		}

		vector& operator/=(real _Right) {
			this->assign(*this, [_Right](real _Left) {
				return _Left / _Right; });
			return *this;
		}

		vector& operator%=(real _Right) {
			this->assign(*this, [_Right](real _Left) {
				return _Left % _Right; });
			return *this;
		}

		friend vector operator/(real _Left, vector _This) {
			return calculate(_This, [_Left](real _Right) {
				return _Left / _Right; });
		}

		scalar_type _Mydata[_Size];
	};

	template<typename _Ty, typename _Traits = block_traits<_Ty>>
	using vector2 = vector<_Ty, 2, _Traits>;
	template<typename _Ty, typename _Traits = block_traits<_Ty>>
	using vector3 = vector<_Ty, 3, _Traits>;
	template<typename _Ty, typename _Traits = block_traits<_Ty>>
	using vector4 = vector<_Ty, 4, _Traits>;

	template<typename T, size_t N> inline
	vector<T,N> isnan(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return scalarA != scalarA; });
	}

	template<typename T, size_t N> inline
	vector<T,N> isinf(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return isinf(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T, N> cast_nan(vector<T,N> vectorA, vector<T,N> cast_vector) {
		return vector<T,N>::calculate(vectorA, cast_vector, [](T scalarA, T cast_scalar) {
			return isnan(scalarA) ? cast_scalar : scalarA; });
	}

	template<typename T, size_t N> inline
	vector<T, N> cast_nan(vector<T,N> vectorA, T cast_scalar) {
		return vector<T,N>::calculate(vectorA, [cast_scalar](T scalarA) {
			return isnan(scalarA) ? cast_scalar : scalarA; });
	}

	template<typename T, size_t N> inline
	vector<T,N> min(vector<T,N> vectorA, vector<T,N> vectorB) {
		return vector<T,N>::calculate(vectorA, vectorB, [](T scalarA, T scalarB) { 
			return min(scalarA, scalarB); });
	}

	template<typename T, size_t N> inline
	vector<T,N> max(vector<T,N> vectorA, vector<T,N> vectorB) {
		return vector<T,N>::calculate(vectorA, vectorB, [](T scalarA, T scalarB) { 
			return max(scalarA, scalarB); });
	}

	template<typename T, size_t N> inline
	vector<T,N> clamp(vector<T,N> numbers, vector<T,N> lowers, vector<T,N> uppers) {
		return min(max(numbers, lowers), uppers);
	}
	
	template<typename T, size_t N> inline
	vector<T,N> remap(vector<T,N> numbers, vector<T,N> lowers, vector<T,N> uppers, vector<T,N> new_lowers, vector<T,N> new_uppers) {
		return (numbers - lowers) / (uppers - lowers) * (new_uppers - new_lowers) + new_lowers;
	}
	
	template<typename T, size_t N> inline
	vector<T,N> rescale(vector<T, N> numbers, vector<T,N> lowers, vector<T,N> uppers) {
		return (numbers - lowers) / (uppers - lowers);
	}

	template<typename T, size_t N> inline
	vector<T,N> min(vector<T,N> vectorA, T scalarB) {
		return vector<T,N>::calculate(vectorA, [scalarB](T scalarA) { 
			return min(scalarA, scalarB); });
	}
	
	template<typename T, size_t N> inline
	vector<T,N> max(vector<T,N> vectorA, T scalarB) {
		return vector<T,N>::calculate(vectorA, [scalarB](T scalarA) { 
			return max(scalarA, scalarB); });
	}

	template<typename T, size_t N> inline
	vector<T,N> clamp(vector<T,N> numbers, T lower, T upper) {
		return min(max(numbers, lower), upper);
	}
	
	template<typename T, size_t N> inline
	vector<T,N> lerp(vector<T,N> start, vector<T,N> end, T t) {
		return start + (end - start) * t;
	}

	template<typename T, size_t N> inline
	vector<T,N> remap(vector<T,N> numbers, T lower, T upper, T new_lower, T new_upper) {
		return (numbers - lower) * ((new_upper - new_lower) / (upper - lower)) + new_lower;
	}
	
	template<typename T, size_t N> inline
	vector<T,N> rescale(vector<T, N> numbers, T lower, T upper) {
		return (numbers - lower) / (upper - lower);
	}

	template<typename T, size_t N> inline
	vector<T,N> saturate(vector<T,N> vectorA) {
		return clamp(vectorA, static_cast<T>(0), static_cast<T>(1));
	}


	template<typename T, size_t N> inline
	vector<T,N> sign(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return sign(scalarA); });
	}
	
	template<typename T, size_t N> inline
	vector<T,N> abs(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return abs(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> floor(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return floor(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> ceil(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return ceil(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> trunc(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return trunc(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> frac(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return frac(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> round(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return round(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> mod(vector<T,N> vectorA, vector<T,N> vectorB) {
		return vector<T,N>::calculate(vectorA, vectorB, [](T scalarA, T scalarB) {
			return mod(scalarA, scalarB); });
	}

	template<typename T, size_t N> inline
	vector<T,N> mod(vector<T,N> vectorA, T scalarB) {
		return vector<T,N>::calculate(vectorA, [scalarB](T scalarA) {
			return mod(scalarA, scalarB); });
	}
	
	template<typename T, size_t N> inline
	vector<T,N> exp(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) {
			return exp(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> ln(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) {
			return ln(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> pow(vector<T,N> bases, vector<T,N> powers) {
		return vector<T,N>::calculate(bases, powers, [](T base, T power) { 
			return pow(base, power); });
	}

	template<typename T, size_t N> inline
	vector<T,N> pow(vector<T,N> bases, T power) {
		return vector<T,N>::calculate(bases, [power](T base) { 
			return pow(base, power); });
	}

	template<typename T, size_t N> inline
	vector<T,N> log(vector<T,N> bases, T number) {
		return vector<T,N>::calculate(bases, [number](T base) {
			return ln(number) / ln(base); });
	}

	template<typename T, size_t N> inline
	vector<T,N> log(vector<T,N> bases, vector<T,N> numbers) {
		return ln(numbers) / ln(bases);
	}

	template<typename T, size_t N> inline
	vector<T,N> sqrt(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return sqrt(scalarA); });
	}
	
	template<typename T, size_t N> inline
	vector<T,N> cbrt(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return cbrt(scalarA); });
	}
	
	template<typename T, size_t N> inline
	vector<T,N> rsqrt(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return rsqrt(scalarA); });
	}


	template<typename T, size_t N> inline
	vector<T,N> cos(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return cos(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> sin(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return sin(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> tan(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return tan(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> acos(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return acos(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> asin(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return asin(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> atan(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return atan(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> atan2(vector<T,N> vectorY, vector<T,N> vectorX) {
		return vector<T,N>::calculate(vectorY, vectorX, [](T scalarY, T scalarX) { 
			return atan2(scalarY, scalarX); });
	}

	template<typename T, size_t N> inline
	vector<T,N> cosh(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return cosh(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> sinh(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return sinh(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> tanh(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return tanh(scalarA); });
	}
	
	template<typename T, size_t N> inline
	vector<T,N> acosh(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return acosh(scalarA); });
	}

	template<typename T, size_t N> inline
	vector<T,N> asinh(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return asinh(scalarA); });
	}
	
	template<typename T, size_t N> inline
	vector<T,N> atanh(vector<T,N> vectorA) {
		return vector<T,N>::calculate(vectorA, [](T scalarA) { 
			return atanh(scalarA); });
	}


	template<typename T, size_t N>
	T dot(vector<T,N> vectorA, vector<T,N> vectorB) {
		if constexpr (N == 2) {
			return vectorA[0] * vectorB[0] +
				   vectorA[1] * vectorB[1];
		} else if constexpr (N == 3) {
			return vectorA[0] * vectorB[0] + 
				   vectorA[1] * vectorB[1] +
				   vectorA[2] * vectorB[2];
		} else if constexpr (N == 4) {
			return vectorA[0] * vectorB[0] + 
				   vectorA[1] * vectorB[1] + 
				   vectorA[2] * vectorB[2] +
				   vectorA[3] * vectorB[3];
		} else {
			T total = vectorA[0] * vectorB[0];
			for (size_t i = 1; i != N; ++i) {
				total += vectorA[i] * vectorA[i];
			}

			return total;
		}
	}

	template<typename T, size_t N>
	T norm(vector<T, N> vectorA, int level) {
		T total = pow(vectorA[0], level);
		for (size_t i = 1; i != N; ++i) {
			total += pow(vectorA[i], level);
		}

		return pow(total, static_cast<T>(1.0/level));
	}
	
	template<typename T, size_t N> inline
	T length(vector<T,N> vectorA) {
		return sqrt(dot(vectorA, vectorA));
	}
	
	template<typename T, size_t N> inline
	vector<T,N> cross3(vector<T,N> vectorA, vector<T,N> vectorB) {
		/*<idea>
			[ i  j  k  ]
			[ lx ly lz ] = i*1*det(minor(0,0)) + j*-1*det(minor(0,1)) + k*1*det(minor(0,2)), 1.determinat expand
			[ rx ry rz ]
						 = vector{ det(minor(0,0)), -det(minor(0,1)), det(minor(0,2)) }      2.cast to vector
		</idea>*/
		return calculation::vector<T,N>{
			vectorA[1] * vectorB[2] - vectorA[2] * vectorB[1],
			vectorA[2] * vectorB[0] - vectorA[0] * vectorB[2],
			vectorA[0] * vectorB[1] - vectorA[1] * vectorB[0]
		};
	}
	
	template<typename T, size_t N> inline
	vector<T,N> cross4(vector<T,N> vectorA, vector<T,N> vectorB, vector<T,N> vectorC) {
		/*<idea>
			[   i       j      k     u  ]
			[ v1.x    v1.y   v1.z  v1.w ]
			[ v2.x    v2.y   v2.z  v2.w ] = i*1*det(minor(0,0)) + j*-1*det(minor(0,1)) + k*1*det(minor(0,2)) + u*-1*det(minor(0,3)), 1.determinat expand
			[ v3.x    v3.y   v3.z  v3.w ]
				|      | |    |      |    = vector{ +(v1.y*detC - v1.z*detE + v1.w*detB),
				+-detA-+-detB-+-detC-+              -(v1.x*detC - v1.z*detF + v1.w*detD),
				|        |    |      |              +(v1.x*detE - v1.y*detF + v1.w*detA),
				+---detD-+----+      |              -(v1.x*detB - v1.y*detD + v1.z*detA) }
				|        |           |
				|   	 +----detE---+
				|                    |
				+-----detF-----------+
		</idea>*/
		T detA = vectorB[0] * vectorC[1] - vectorB[1] * vectorC[0];
		T detB = vectorB[1] * vectorC[2] - vectorB[2] * vectorC[1];
		T detC = vectorB[2] * vectorC[3] - vectorB[3] * vectorC[2];
		T detD = vectorB[0] * vectorC[2] - vectorB[2] * vectorC[0];
		T detE = vectorB[1] * vectorC[3] - vectorB[3] * vectorC[1];
		T detF = vectorB[0] * vectorC[3] - vectorB[3] * vectorC[0];
		return calculation::vector<T,N>{
			  vectorA[1]*detC - vectorA[2]*detE + vectorA[3]*detB,
			-(vectorA[0]*detC - vectorA[2]*detF + vectorA[3]*detD),
			  vectorA[0]*detE - vectorA[1]*detF + vectorA[3]*detA,
			-(vectorA[0]*detB - vectorA[1]*detD + vectorA[2]*detA) 
		};
	}
	
	template<typename T, size_t N> inline
	vector<T,N> normalize(vector<T,N> vectorA) {
		return vectorA * rsqrt(dot(vectorA, vectorA));
	}
}// namespace calculation



#include <memory>// vector_any<>
#include <vector>// vector_any<>

namespace calculation{
	// { vector, dynamic size }
	template<typename _Ty, typename _BlkTy = _Ty>
	class dynamic_vector {
	public:
		using vector_category = aligned_vector_tag;

		using scalar_type    = _Ty;
		using block_type     = _BlkTy;
		//using subvector_type = subvector<scalar_type, block_type>;
		using iterator       = _Ty*;
		using const_iterator = const _Ty*;
		
		using _My_block_traits = block_traits<block_type>;
		
#pragma region vector_any_data
		std::vector<_BlkTy> _My_blocks;
		size_t              _My_size;

		size_t _Real_size()  const { return _My_blocks.size() * _My_block_traits::size(); }
		size_t _Tail_size()  const { return size() % _My_block_traits::size(); }
		size_t _Block_size() const { return size() / _My_block_traits::size(); }
		size_t  size()    const { return _My_size; }
		bool    empty()   const { return _My_size == 0; }
		bool    aligned() const { return _Tail_size() == 0; }
		
		iterator begin() {
			return reinterpret_cast<iterator>(_My_blocks.data());
		}
		const_iterator begin() const {
			return reinterpret_cast<const_iterator>(_My_blocks.data());
		}
		iterator end() {
			return begin() + size();
		}
		const_iterator end() const {
			return begin() + size();
		}
		
		scalar_type* data() { return begin(); }
		const scalar_type* data() const { return begin(); }

		template<typename _Ty>
		_Ty& at(size_t _Pos) {
			assert( _Pos < size() );
			return *( data() + _Pos );
		}
		template<typename _Ty>
		const _Ty& at(size_t _Pos) const {
			assert( _Pos < size() );
			return *( data() + _Pos );
		}
		
		void _Correct_tail_elements() {
			if ( _Real_size() > size() ) {
				std::fill( ptr(size()), ptr(_Real_size()), static_cast<scalar_type>(0) );
			}
		}
		void release() noexcept {
			if (_My_size != 0) {
				std::swap(std::vector<block_type>(), std::move(_My_blocks));// noexcept
				_My_size = 0;
			}
		}
		void swap(dynamic_vector& _Right) noexcept {
			std::swap(_My_blocks, _Right._My_blocks);
			std::swap(_My_size,   _Right._My_size);
		}
		void resize(size_t _Newsize) {
			size_t _Real_newsize = (_Newsize + (_My_block_traits::size()-1)) & (~(block_traits::size() - 1));
			_My_blocks.resize(_Real_newsize / block_traits::size());
			_My_size = _Newsize;
		}
		void fill(const scalar_type& _Val) {
			std::fill(begin(), end(), _Val);
		}
		template<typename _Iter>
		void assign(_Iter _First, _Iter _Last) {
			size_t _Diff = std::distance(_First, _Last);
			if (_Diff <= size()) {
				std::fill( std::copy(_First, _Last, this->begin()), 
					       this->end(), 
					       static_cast<_Ty>(0) );
			} else {
				dynamic_vector _New_data;
				_New_data.resize(_Diff);
				_New_data.assign(_First, _Last);
				_New_data.swap(*this);
			}
		}
#pragma endregion

		dynamic_vector() = default;
		dynamic_vector(const dynamic_vector&) = default;
		dynamic_vector(dynamic_vector&& _Right) noexcept {// move-constructor
			_Right.swap(*this);
			_Right.release();
		}
		dynamic_vector(size_t _Count, scalar_type _Val) {
			this->resize(_Count);
			this->fill(_Val);
			this->_Correct_tail_elements();
		}
		dynamic_vector(std::initializer_list<scalar_type> _Ilist) {
			this->assign(_Ilist.begin(), _Ilist.end());
			this->_Correct_tail_elements();
		}
		template<typename _Iter>
		dynamic_vector(_Iter _First, _Iter _Last) {
			this->assign(_First, _Last);
			this->_Correct_tail_elements();
		}
		
		dynamic_vector& operator=(const dynamic_vector&) = default;
		dynamic_vector& operator=(dynamic_vector&& _Right) {
			_Right.swap(*this);
			_Right.release();
		}
		
		scalar_type& operator[](size_t _Pos) { return at<scalar_type>(_Pos); }
		const scalar_type& operator[](size_t _Pos) const { return at<scalar_type>(_Pos); }

	/*	subvector_type operator()(size_t i, size_t f) {
			return subvector_type(data() + i, data() + f);
		}
		const subvector_type operator()(size_t i, size_t f) const {
			return const_cast<dynamic_vector&>(*this).operator()(i, f);
		}*/
		
		dynamic_vector  operator-() const;
		dynamic_vector  operator+ (const dynamic_vector& _Right) const;
		dynamic_vector  operator- (const dynamic_vector& _Right) const;
		dynamic_vector  operator* (const dynamic_vector& _Right) const;
		dynamic_vector  operator/ (const dynamic_vector& _Right) const;
		dynamic_vector  operator% (const dynamic_vector& _Right) const;
		dynamic_vector& operator+=(const dynamic_vector& _Right);
		dynamic_vector& operator-=(const dynamic_vector& _Right);
		dynamic_vector& operator*=(const dynamic_vector& _Right);
		dynamic_vector& operator/=(const dynamic_vector& _Right);
		dynamic_vector& operator%=(const dynamic_vector& _Right);
		/*template<scalar _Ty> vector_any  operator+(_Ty s) const;
		template<scalar _Ty> vector_any  operator-(_Ty s) const;
		template<scalar _Ty> vector_any  operator*(_Ty s) const;
		template<scalar _Ty> vector_any  operator/(_Ty s) const;
		template<scalar _Ty> vector_any  operator%(_Ty s) const;
		template<scalar _Ty> vector_any& operator+=(_Ty s) const;
		template<scalar _Ty> vector_any& operator-=(_Ty s) const;
		template<scalar _Ty> vector_any& operator*=(_Ty s) const;
		template<scalar _Ty> vector_any& operator/=(_Ty s) const;
		template<scalar _Ty> vector_any& operator%=(_Ty s) const;*/
		scalar_type dot(const dynamic_vector&) const;
		scalar_type norm(int L) const;
		scalar_type length() const { return sqrt(this->dot(*this)); }
		void        normalize();
	};

	using point2_size_t = vector<size_t, 2>;

}// namespace clmagic


#ifdef _INCLUDED_MM2
const union {  int i[4]; __m128 m; } __f32vec4_abs_mask_cheat
	= { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

/* SSE default-true: 1|11111111|11111111111111111111111
               1.0F: 0|01111111|00000000000000000000000 */
const union { float f[4]; __m128 m; } __f32vec4_eq_mask_cheat
	= { 1.0F, 1.0F, 1.0F, 1.0F };

#define _mm_abs_ps(x) \
	_mm_and_ps(x, __f32vec4_abs_mask_cheat.m)

#define _MM_SHUFFLER(fp0, fp1, fp2, fp3) \
	_MM_SHUFFLE(fp3, fp2, fp1, fp0)

#define _mm_shuffler_ps(v0, v1, i) \
	_mm_shuffle_ps(v1, v0, (i))

inline __m128 _mm_frac_ps(__m128 x) {
    return _mm_sub_ps(x, _mm_floor_ps(x));
}

inline __m128 _mm_lerp_ps(__m128 start, __m128 end, __m128 t) {
    return _mm_add_ps(start, _mm_mul_ps(_mm_sub_ps(end, start), t));
}

inline __m128 _mm_lerp_ss(__m128 start, __m128 end, __m128 t) {
    return _mm_add_ss(start, _mm_mul_ss(_mm_sub_ss(end, start), t));
}

inline __m128 _mm_scurve_ps(__m128 x) {
    __m128 xx = _mm_mul_ps(x, x);
    __m128 xxx = _mm_mul_ps(xx, x);
    return _mm_sub_ps(_mm_mul_ps(xxx, _mm_set1_ps(2.0F)), _mm_mul_ps(xx, _mm_set1_ps(3.0F)));
}

inline __m128 _mm_fade_ps(__m128 t) {
    __m128 ttt = _mm_mul_ps(_mm_mul_ps(t, t), t);
    __m128 tttt = _mm_mul_ps(ttt, t);
    __m128 ttttt = _mm_mul_ps(tttt, t);
    return _mm_add_ps(_mm_sub_ps(_mm_mul_ps(ttttt, _mm_set1_ps(6.0F)),
        _mm_mul_ps(tttt, _mm_set1_ps(15.0F))),
        _mm_mul_ps(ttt, _mm_set1_ps(10.0F)));
}

namespace calculation {
    template<size_t _Size>
	using m128vector = vector<float, _Size, block_traits<__m128>>;

	template<size_t _Size>
	struct __declspec(align(16)) vector<float, _Size, block_traits<__m128>> {
		static constexpr size_t scalar_size = _Size;
		using vector_category = static_aligned_vector_tag;
		using alignment_type  = static_alignment_traits<float, _Size, __m128>;

		using traits_type = block_traits<__m128>;
		using scalar_type = float;
		using block_type  = __m128;

		using iterator       = scalar_type*;
		using const_iterator = const scalar_type*;

	private:
		using real = scalar_type;

	public:
		constexpr size_t size() const { 
			return scalar_size;
		}
		
		constexpr bool aligned() const { 
			return (scalar_size % traits_type::size()) == 0;
		}
		
		constexpr bool empty() const { 
			return false; 
		}
		
		const scalar_type* data() const {
			return _Mydata; 
		}
		
		scalar_type* data() {
			return _Mydata;
		}

		const scalar_type* begin() const {
			return _Mydata; 
		}
		
		scalar_type* begin() {
			return _Mydata; 
		}
		
		const scalar_type* end() const {
			return _Mydata + size(); 
		}
		
		scalar_type* end() {
			return _Mydata + size(); 
		}
		
		const scalar_type& at(size_t _Pos) const {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}
		
		scalar_type& at(size_t _Pos) {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}
		
		const scalar_type& operator[](size_t _Pos) const {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}
		
		scalar_type& operator[](size_t _Pos) {
			assert( _Pos < size() );
			return _Mydata[_Pos];
		}

		vector operator+(vector _Right) const {
			if constexpr (scalar_size == 4) {
				vector _Result;
				_mm_store_ps(_Result.data(), 
					_mm_add_ps(_mm_load_ps(this->data()), 
							   _mm_load_ps(_Right.data())));
				return std::move(_Result);
			} else if constexpr (scalar_size == 3) {
				vector _Result;
				__m128 result = _mm_add_ps(_mm_load_ps(this->data()),
										   _mm_load_ps(_Right.data()));
				return vector{ result.m128_f32[0], result.m128_f32[1], result.m128_f32[2] };
			} else {
				vector _Result;
				transform_static_aligned_vector<__m128, _Size>(this->data(), _Right.data(), 
					[](__m128 a, __m128 b) { return _mm_add_ps(a, b); });
				return std::move(_Result);
			}
		}

		float _Mydata[_Size];
	};

	template<size_t N, typename _mm_unaryop> inline
	m128vector<N> generate_vector(const m128vector<N>& _Source, _mm_unaryop _Func) {
		m128vector<N> _Result;
		if constexpr (N <= 4) {
			_mm_store_ps(_Result.data(), _Func( _mm_load_ps(_Source.data()) ));
			return std::move(_Result);
		} else {
			const float* _First = _Source.data();
			const float* _Last  = _Source.data() + _Source.aligned_size();
			float*       _Dest  = _Result.data();
			for (; _First != _Last; _First += 4, _Dest += 4) {
				_mm_store_ps( _Dest, _Func(_mm_load_ps(_First)) );
			}
		}

		return std::move(_Result);
	}

	template<size_t N, typename _mm_binaryop> inline
	m128vector<N> generate_vector(const m128vector<N>& _Left, const m128vector<N>& _Right, _mm_binaryop _Func) {
		m128vector<N> _Result;
		if constexpr (N <= 4) {
			_mm_store_ps( _Result.data(), _Func(_mm_load_ps(_Left.data()), _mm_load_ps(_Right.data())) );
		} else {
			const float* _First1 = _Left.data();
			const float* _Last1  = _Left.data() + _Left.aligned_size();
			const float* _First2 = _Right.data();
			float*       _Dest   = _Result.data();
			for (; _First1 != _Last1; _First1 += 4, _First2 += 4, _Dest += 4) {
				_mm_store_ps( _Dest, _Func(_mm_load_ps(_First1), _mm_load_ps(_First2)) );
			}
		}

		return std::move(_Result);
	}
}

//template<size_t N> inline
//calculation::m128vector<N> operator-(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_mul_ps(a, _mm_set1_ps(-1.0f)); });
//}
//
//template<size_t N> inline
//calculation::m128vector<N> operator+(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_add_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator-(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_sub_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator*(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_mul_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator/(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_div_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator%(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_fmod_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator==(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, 
//		[](__m128 a, __m128 b) { return _mm_and_ps(_mm_cmpeq_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator!=(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2,
//		[](__m128 a, __m128 b) { return _mm_and_ps(_mm_cmpneq_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator<(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2,
//		[](__m128 a, __m128 b) { return _mm_and_ps(_mm_cmplt_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator>(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2,
//		[](__m128 a, __m128 b) { return _mm_and_ps(_mm_cmpgt_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator<=(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2,
//		[](__m128 a, __m128 b) { return _mm_and_ps(_mm_cmple_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator>=(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2,
//		[](__m128 a, __m128 b) { return _mm_and_ps(_mm_cmpge_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//
//template<size_t N> inline
//calculation::m128vector<N>& operator+=(calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	if constexpr (N <= 4) {
//		_mm_store_ps( vector1.data(), _mm_add_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data())) );
//	} else {
//		float*       _First1 = vector1.data();
//		const float* _Last1  = vector1.data() + vector1.aligned_size();
//		const float* _First2 = vector2.data();
//		for (; _First1 != _Last1; _First1 += 4, _First2 += 4) {
//			_mm_store_ps( _First1, _mm_add_ps(_mm_load_ps(_First1), _mm_load_ps(_First2)) );
//		}
//	}
//	return vector1;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator-=(calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//    if constexpr (N <= 4) {
//		_mm_store_ps( vector1.data(), _mm_sub_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data())) );
//	} else {
//		float*       _First1 = vector1.data();
//		const float* _Last1  = vector1.data() + vector1.aligned_size();
//		const float* _First2 = vector2.data();
//		for (; _First1 != _Last1; _First1 += 4, _First2 += 4) {
//			_mm_store_ps( _First1, _mm_sub_ps(_mm_load_ps(_First1), _mm_load_ps(_First2)) );
//		}
//	}
//	return vector1;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator*=(calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//    if constexpr (N <= 4) {
//		_mm_store_ps( vector1.data(), _mm_mul_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data())) );
//	} else {
//		float*       _First1 = vector1.data();
//		const float* _Last1  = vector1.data() + vector1.aligned_size();
//		const float* _First2 = vector2.data();
//		for (; _First1 != _Last1; _First1 += 4, _First2 += 4) {
//			_mm_store_ps( _First1, _mm_mul_ps(_mm_load_ps(_First1), _mm_load_ps(_First2)) );
//		}
//	}
//	return vector1;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator/=(calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//    if constexpr (N <= 4) {
//		_mm_store_ps( vector1.data(), _mm_div_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data())) );
//	} else {
//		float*       _First1 = vector1.data();
//		const float* _Last1  = vector1.data() + vector1.aligned_size();
//		const float* _First2 = vector2.data();
//		for (; _First1 != _Last1; _First1 += 4, _First2 += 4) {
//			_mm_store_ps( _First1, _mm_div_ps(_mm_load_ps(_First1), _mm_load_ps(_First2)) );
//		}
//	}
//	return vector1;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator%=(calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//    if constexpr (N <= 4) {
//		_mm_store_ps( vector1.data(), _mm_fmod_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data())) );
//	} else {
//		float*       _First1 = vector1.data();
//		const float* _Last1  = vector1.data() + vector1.aligned_size();
//		const float* _First2 = vector2.data();
//		for (; _First1 != _Last1; _First1 += 4, _First2 += 4) {
//			_mm_store_ps( _First1, _mm_fmod_ps(_mm_load_ps(_First1), _mm_load_ps(_First2)) );
//		}
//	}
//	return vector1;
//}
//
//template<size_t N> inline
//calculation::m128vector<N> operator+(const calculation::m128vector<N>& vector, const float scalar) {
//    __m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_add_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator-(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_sub_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator*(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_mul_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator/(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_div_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator%(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_fmod_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator==(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, 
//		[b](__m128 a) { return _mm_and_ps(_mm_cmpeq_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator!=(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector,
//		[b](__m128 a) { return _mm_and_ps(_mm_cmpneq_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator<(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector,
//		[b](__m128 a) { return _mm_and_ps(_mm_cmplt_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator>(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector,
//		[b](__m128 a) { return _mm_and_ps(_mm_cmpgt_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator<=(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector,
//		[b](__m128 a) { return _mm_and_ps(_mm_cmple_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> operator>=(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector,
//		[b](__m128 a) { return _mm_and_ps(_mm_cmpge_ps(a, b), __f32vec4_eq_mask_cheat.m); });
//}
//
//template<size_t N> inline
//calculation::m128vector<N>& operator+=(calculation::m128vector<N>& vector, const float scalar) {
//    __m128 temp = _mm_set1_ps(scalar);
//	if constexpr (N <= 4) {
//		_mm_store_ps( vector.data(), _mm_add_ps(_mm_load_ps(vector.data()), temp) );
//	} else {
//		float*       _First = vector.data();
//		const float* _Last  = vector.data() + vector.aligned_size();
//		for (; _First != _Last; _First += 4) {
//			_mm_store_ps( _First, _mm_add_ps(_mm_load_ps(_First), temp) );
//		}
//	}
//	return vector;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator-=(calculation::m128vector<N>& vector, const float scalar) {
//	__m128 temp = _mm_set1_ps(scalar);
//	if constexpr (N <= 4) {
//		_mm_store_ps( vector.data(), _mm_sub_ps(_mm_load_ps(vector.data()), temp) );
//	} else {
//		float*       _First = vector.data();
//		const float* _Last  = vector.data() + vector.aligned_size();
//		for (; _First != _Last; _First += 4) {
//			_mm_store_ps( _First, _mm_sub_ps(_mm_load_ps(_First), temp) );
//		}
//	}
//	return vector;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator*=(calculation::m128vector<N>& vector, const float scalar) {
//	__m128 temp = _mm_set1_ps(scalar);
//	if constexpr (N <= 4) {
//		_mm_store_ps( vector.data(), _mm_mul_ps(_mm_load_ps(vector.data()), temp) );
//	} else {
//		float*       _First = vector.data();
//		const float* _Last  = vector.data() + vector.aligned_size();
//		for (; _First != _Last; _First += 4) {
//			_mm_store_ps( _First, _mm_mul_ps(_mm_load_ps(_First), temp) );
//		}
//	}
//	return vector;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator/=(calculation::m128vector<N>& vector, const float scalar) {
//	__m128 temp = _mm_set1_ps(scalar);
//	if constexpr (N <= 4) {
//		_mm_store_ps( vector.data(), _mm_div_ps(_mm_load_ps(vector.data()), temp) );
//	} else {
//		float*       _First = vector.data();
//		const float* _Last  = vector.data() + vector.aligned_size();
//		for (; _First != _Last; _First += 4) {
//			_mm_store_ps( _First, _mm_div_ps(_mm_load_ps(_First), temp) );
//		}
//	}
//	return vector;
//}
//template<size_t N> inline
//calculation::m128vector<N>& operator%=(calculation::m128vector<N>& vector, const float scalar) {
//	__m128 temp = _mm_set1_ps(scalar);
//	if constexpr (N <= 4) {
//		_mm_store_ps( vector.data(), _mm_fmod_ps(_mm_load_ps(vector.data()), temp) );
//	} else {
//		float*       _First = vector.data();
//		const float* _Last  = vector.data() + vector.aligned_size();
//		for (; _First != _Last; _First += 4) {
//			_mm_store_ps( _First, _mm_fmod_ps(_mm_load_ps(_First), temp) );
//		}
//	}
//	return vector;
//}
//
//template<size_t N> inline
//calculation::m128vector<N> min(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_min_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> max(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_max_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> clamp(const calculation::m128vector<N>& vector, const calculation::m128vector<N>& lower, const calculation::m128vector<N>& upper) {
//    return min(max(vector, lower), upper);
//}
//template<size_t N> inline
//calculation::m128vector<N> remap(const calculation::m128vector<N>& numbers, const calculation::m128vector<N>& lowers, const calculation::m128vector<N>& uppers, const calculation::m128vector<N>& new_lowers, const calculation::m128vector<N>& new_uppers) {
//	return (numbers - lowers) / (uppers - lowers) * (new_uppers - new_lowers) + new_lowers;
//}
//template<size_t N> inline
//calculation::m128vector<N> rescale(const calculation::m128vector<N>& lowers, const calculation::m128vector<N>& uppers, const calculation::m128vector<N>& numbers) {
//    return (numbers - lowers) / (uppers - lowers);
//}
//template<size_t N> inline
//calculation::m128vector<N> min(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_min_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> max(const calculation::m128vector<N>& vector, const float scalar) {
//	__m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_max_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> clamp(const calculation::m128vector<N>& vector, const float lower, const float upper) {
//	return min(max(vector, lower), upper);
//}
//template<size_t N> inline
//calculation::m128vector<N> remap(const calculation::m128vector<N>& numbers, const float lower, const float upper, const float new_lower, const float new_upper) {
//	// less precision, numbers in [lower,upper], (upper-lower) > (numbers-lower), if (new_upper - new_lower) less many than (upper-lower)
//	return (numbers - lower) * ((new_upper - new_lower) / (upper - lower)) + new_lower;
//}
//template<size_t N> inline
//calculation::m128vector<N> rescale(const calculation::m128vector<N>& numbers, const float lower, const float upper) {
//	return (numbers - lower) / (upper - lower);
//}
//template<size_t N> inline
//calculation::m128vector<N> saturate(const calculation::m128vector<N>& vector) {
//    return clamp(vector, 0.0F, 1.0F);
//}
//template<size_t N> inline
//calculation::m128vector<N> lerp(const calculation::m128vector<N>& start, const calculation::m128vector<N>& end, const float t) {
//	return start + (end - start) * t;
//}
//
//template<size_t N> inline
//calculation::m128vector<N> sign(const calculation::m128vector<N>& vector) {
//	return (vector > 0.0f) - (vector < 0.0f);
//}
//template<size_t N> inline
//calculation::m128vector<N> abs(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_and_ps(a, __f32vec4_abs_mask_cheat.m); });
//}
//template<size_t N> inline
//calculation::m128vector<N> floor(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_floor_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> ceil(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_ceil_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> trunc(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_trunc_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> frac(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_frac_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> round(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_round_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> mod(const calculation::m128vector<N>& vector, const float scalar) {
//    return vector % scalar;
//}
//template<size_t N> inline
//calculation::m128vector<N> mod(const calculation::m128vector<N>& vector, const calculation::m128vector<N>& divisors) {
//    return vector % divisors;
//}
//template<size_t N> inline
//calculation::m128vector<N> exp(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_exp_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> ln(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_log_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> pow(const calculation::m128vector<N>& vector, const float scalar) {
//    __m128 b = _mm_set1_ps(scalar);
//	return calculation::generate_vector(vector, [b](__m128 a) { return _mm_pow_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> pow(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	return calculation::generate_vector(vector1, vector2, [](__m128 a, __m128 b) { return _mm_pow_ps(a, b); });
//}
//template<size_t N> inline
//calculation::m128vector<N> log(const calculation::m128vector<N>& bases, const calculation::m128vector<N>& numbers) {
//	return ln(numbers) / ln(bases);
//}
//template<size_t N> inline
//calculation::m128vector<N> sqrt(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_sqrt_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> cbrt(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_cbrt_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> rsqrt(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_rsqrt_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> rcbrt(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_invcbrt_ps(a); });
//}
//
//template<size_t N> inline
//calculation::m128vector<N> sin(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_sin_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> cos(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_cos_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> tan(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_tan_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> asin(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_asin_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> acos(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_acos_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> atan(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_atan_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> atan2(const calculation::m128vector<N>& y_vector, const calculation::m128vector<N>& x_vector) {
//	return calculation::generate_vector(y_vector, x_vector, [](__m128 y, __m128 x) { return _mm_atan2_ps(y,x); });
//}
//template<size_t N> inline
//calculation::m128vector<N> sinh(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_sinh_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> cosh(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_cosh_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> tanh(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_tanh_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> asinh(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_asinh_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> acosh(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_acosh_ps(a); });
//}
//template<size_t N> inline
//calculation::m128vector<N> atanh(const calculation::m128vector<N>& vector) {
//	return calculation::generate_vector(vector, [](__m128 a) { return _mm_atanh_ps(a); });
//}
//
//template<size_t N> inline
//float dot(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//	if constexpr (N == 4) {
//		return _mm_cvtss_f32(_mm_dp_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data())));
//	} else if constexpr (N == 3) {
//		__m128 temp = _mm_mul_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data()));
//		temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0, 2)));
//		temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0, 1)));
//		return _mm_cvtss_f32(temp);
//	} else if constexpr (N == 2) {
//		__m128 temp = _mm_mul_ps(_mm_load_ps(vector1.data()), _mm_load_ps(vector2.data()));
//		temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0, 1)));
//		return _mm_cvtss_f32(temp);
//	} else {
//		return calculation::transform_reduce_static_aligned_vector<__m128, N>(vector1.data(), vector2.data(), 0.0F, 
//			[](__m128 a, __m128 b) { return _mm_add_ps(a, b); }, std::plus<>(), 
//			[](__m128 a, __m128 b) { return _mm_mul_ps(a, b); });
//	}
//}
//template<size_t N> inline
//float norm(const calculation::m128vector<N>& vector, int level) {
//	if constexpr (N == 4) {
//		__m128 temp = _mm_pow_ps( _mm_load_ps(vector.data()), _mm_set_ps(level) );
//		temp = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0, 3,2)));
//		temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0, 1)));
//		return pow(_mm_cvtss_f32(temp), 1.0f/level);
//	} else if constexpr (N == 3) {
//		__m128 temp = _mm_pow_ps( _mm_load_ps(vector.data()), _mm_set_ps(level) );
//		temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0, 2)));
//		temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0, 1)));
//		return pow(_mm_cvtss_f32(temp), 1.0f/level);
//	}  else if constexpr (N == 2) {
//		__m128 temp = _mm_pow_ps( _mm_load_ps(vector.data()), _mm_set_ps(level) );
//		temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(0,0,0, 1)));
//		return pow(_mm_cvtss_f32(temp), 1.0f/level);
//	} else {
//		float sum_pow_level = calculation::transform_reduce_static_aligned_vector<__m128, N>(vector.data(), 0.0f,
//			[](__m128 a, __m128 b) { return _mm_add_ps(a.b); }, std::plus<>(),
//			[level](__m128 a) { return _mm_pow_ps(a, _mm_set1_ps(level)); });
//		return pow(sum_pow_level, 1.0f/level);
//	}
//}
//template<size_t N> inline
//float length(const calculation::m128vector<N>& vector) {
//	return sqrt(dot(vector, vector));
//}
//template<size_t N> inline
//calculation::m128vector<N> cross3(const calculation::m128vector<N>& vector1, const calculation::m128vector<N>& vector2) {
//    __m128 a = _mm_load_ps(vector1.data());
//    __m128 b = _mm_load_ps(vector2.data());
//    __m128 c = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(a,a, _MM_SHUFFLE(0, 0,2,1)), _mm_shuffle_ps(b,b, _MM_SHUFFLE(0, 1,0,2))),
//                          _mm_mul_ps(_mm_shuffle_ps(a,a, _MM_SHUFFLE(0, 1,0,2)), _mm_shuffle_ps(b,b, _MM_SHUFFLE(0, 0,2,1))));
//    calculation::m128vector<N> result;
//    _mm_store_ps(result.data(), c);
//    return std::move(result);
//}
//template<size_t N> inline
//calculation::m128vector<N> normalize(const calculation::m128vector<N>& vector) {
//    float length_square = dot(vector, vector);
//    float error         = length_square - 1.0f;
//    return abs(error) < std::numeric_limits<float>::epsilon() ? vector : vector/sqrt(length_square);
//}
//template<size_t N> inline
//calculation::m128vector<N> proj(const calculation::m128vector<N>& _X, const calculation::m128vector<N>& _Proj) {// _X proj to _Proj
//    return _Proj * (dot(_X, _Proj) / dot(_Proj, _Proj));
//}
//template<size_t N> inline
//calculation::m128vector<N> reflect(const calculation::m128vector<N>& vector, const calculation::m128vector<N>& normal) {
//    return (vector - proj(vector, normal) * 2.0F);
//}
#endif // _INCLUDED_MM2

namespace calculation {
	// { unit_vector, static size }
	template<typename _Ty, size_t _Size, typename _Traits = block_traits<_Ty>>
	class unit_vector : public vector<_Ty, _Size, _Traits> {
		using _Mybase = vector<_Ty, _Size, _Traits>;

	public:
		using scalar_type = typename _Traits::scalar_type;
		using block_type  = typename _Traits::block_type;

		unit_vector() = default;
		unit_vector(scalar_type xi, scalar_type yj, scalar_type zk, bool _Unitized = false) : _Mybase{ xi, yj, zk } {
			if (!_Unitized) normalize(static_cast<_Mybase&>(*this));
		}
		unit_vector(const _Mybase& _Vector, bool _Unitized = false) : _Mybase(_Vector) {
			if (!_Unitized) normalize(static_cast<_Mybase&>(*this));
		}

		unit_vector operator-() const {
			return unit_vector(-static_cast<const _Mybase&>(*this), true);
		}
	};
	
	template<typename _Ty, typename _Traits = block_traits<_Ty>>
	using unit_vector3 = unit_vector<_Ty, 3, _Traits>;


	#pragma region vector cast
	// template definition
	template<typename _InVec, typename _OutVec>
	struct _Vector_cast {
		_OutVec operator()(const _InVec& _Source) const {
			return _OutVec( _Source.begin(), _Source.end() );
		}
	};
	
	// template trunc_size
	template<typename _InVec, typename _Ty, size_t _Size, typename _Traits>
	struct _Vector_cast<_InVec, vector<_Ty,_Size,_Traits> > {
		using source_vector_t     = _InVec;
		using detination_vector_t = vector<_Ty, _Size, _Traits>;

		detination_vector_t operator()(const source_vector_t& _Source) const {
			detination_vector_t _Result;
			size_t N = std::min<size_t>(_Source.size(), _Size);
			std::copy(_Source.begin(), std::next(_Source.begin(), N), _Result.begin());
			return std::move(_Result);
		}
	};
	
	template<typename _InVec, typename _OutVec> inline
	void vector_cast(const _InVec& src_vector, _OutVec& dst_vector) {
		// Convert _InVec to _OutVec, from src_vector to dst_vector
		dst_vector = _Vector_cast<_InVec, _OutVec>()(src_vector);
	}
	
	template<typename _InVstream, typename _OutVstream>
	void vector_stream_cast(const _InVstream& src_vstream, _OutVstream& dst_vstream) {
		// Convert _InVec -> _OutVec, from [src_vstream.begin(), src_vstream.end()) to [dst_vstream.begin(), ...)
		auto        _First = src_vstream.begin();
		const auto  _Last  = src_vstream.end();
		auto        _Dest  = dst_vstream.begin();
		for ( ; _First != _Last; ++_First, ++_Dest) {
			vector_cast(*_First, *_Dest);
		}
	}
	
	template<size_t i0, size_t i1, size_t i2, typename _InVec3, typename _OutVec3> inline
	void vector3_cast(const _InVec3& _Source, _OutVec3& _Destination) {
		_Destination[0] = _Source[i0];
		_Destination[1] = _Source[i1];
		_Destination[2] = _Source[i2];
	}
	
	template<size_t i0, size_t i1, size_t i2, typename _InVstream, typename _OutVstream>
	void vector3_stream_cast(const _InVstream& _Source, _OutVstream& _Destination) {
		auto       _First = _Source.begin();
		const auto _Last  = _Source.end();
		auto       _Dest  = _Destination.begin();
		for ( ; _First != _Last; ++_First, ++_Dest) {
			vector3_cast<i0, i1, i2>(*_First, *_Dest);
		}
	}
	
	template<typename _OutVec, typename _InVec> inline
	_OutVec vector_cast(const _InVec& src_vector) {
		// Convert _InVec to _OutVec, return _OutVec(src_vector)
		return _Vector_cast<_InVec,_OutVec>()(src_vector);
	}

	template<typename _OutVstream, typename _InVstream>
	_OutVstream vector_stream_cast(const _InVstream& src_vstream) {
		_OutVstream dst_vstream = _OutVstream( src_vstream.size() );// invoke _OutVstream::_OutVstream(size_t _Count)
		vector_stream_cast<_InVstream, _OutVstream>( src_vstream, dst_vstream );
		return std::move( dst_vstream );
	}

	template<typename _OutVec3, size_t i0, size_t i1, size_t i2, typename _InVec3> inline
	_OutVec3 vector3_cast(const _InVec3& _Source) {
		return _OutVec3{ _Source[i0],  _Source[i1],  _Source[i2] };
	}

	template<typename _OutVstream, size_t i0, size_t i1, size_t i2, typename _InVstream>
	_OutVstream vector3_stream_cast(const _InVstream& _Source) {
		_OutVstream _Destination = _OutVstream( _Source.size() );
		vector3_stream_cast<i0, i1, i2, _InVstream, _OutVstream>( _Source, _Destination );
		return std::move( _Destination );
	}
#pragma endregion
}

/*<version1>
	<idea>
		<block-traits>
			size_t      size()
			block_type  construct0()
			block_type  construct1(scalar_type)
			scalar_type at0(block_type)
		</block-traits>

		<vector-operation>
			_Vector_operation_level{ _FASTEST, _FAST, _NORM, _AUTO }
			_NORM: scalar_operation
			other: scalar_block_mix_operation in accumulate
		</vector-operation>

		<subvector>
			invoke [_AUTO]_vector_operation
		</subvector>
		
		<vector>
			invoke [_FASTEST | _FAST]_vector_operation
		</vector>
		
		<vector_any>
			invoke [_FASTEST | _FAST]_vector_operation
		</vector_any>
	</idea>
	
	<summary>
		1. Can accelerate any vector
		2. The idea seperate vector-algorithm and vector-struct, and unified some kinds of algorithm
	</summary>

	<false>
		1. Dependent on [](auto&&,auto&&){ ... }, compile is [very difficute]
		2. vector implementation codes are [very bloated]
	</false>
</version1>*/

/*<version2>
	<idea>
		<block-traits>
			size_t size()
			block_type set(...)
			_InIt  load(_InIt, block_type&)
			_InIt  load(_InIt, block_type&, size_t)
			_OutIt store(_OutIt, const block_type&)
			_OutIt store(_OutIt, const block_type&, size_t)
			const_iterator begin(const block_type&)
			const_iterator end(const block_type&)
			iterator begin(block_type&)
			iterator end(block_type&)
			
			<summary>
				The new implicit-interface <more flexibility>
				The iterator-interface used to <reduce>
			</summary>
		</block-traits>

		<vector-operation>
			_Vector_operation_level{ _DICONTINUOUS, _CONTINUOUS, _CONTINUOUS_ALIGNED, _AUTO }
			<summary>
				The new _Vector_operation_level <more physical>
			</summary>
		</vector-operation>
	</idea>

	<table-of-content>
		1.   block traits *------------------------------------------------------ L39
		2.   vector iterator ---------------------------------------------------- L97
		3.   vector size -------------------------------------------------------- L243
		4.   vector operation --------------------------------------------------- L365
		5.   shuffle<_OutTy, ...>( _InVec ) ------------------------------------- L1076
		6.   subvector ---------------------------------------------------------- L1143
		7.   vector *------------------------------------------------------------ L1310
		8.   vector_any --------------------------------------------------------- L1476
		9.   vector algorithm --------------------------------------------------- L1637
		10.  vector cast *------------------------------------------------------- L1886
		11.  unit_vector *------------------------------------------------------- L1999
		12.  undetermined ------------------------------------------------------- L
	</table-of-content>
</version2>*/

/*<version3>
	1. more certain than [version2]
	2. compile-complexity [decrease many times]
	3. code-maintenance cost [more lower]
</version3>*/

/*<template-compile>
	sin(_Ty x) : least 1 times convertion
	sin(float x)
	sin(double x)
	sin(complex<_Ty> x)

	Not-convert = MAX_INT

	input float : { 1 times, 0 times, 1 times, Not-convert }
	input double : { 1 times, 1 times, 0 times, Not-convert }
	input Myclass : { 1 times, Not-convert, Not-convert, Not-convert }
	input complex<_Ty> : { 1 times, 0 times, 0 times, 0 times!!!!!!!!!!! }

<idea>
	Container<_Ty>: This is Container idea, Container can direct matched
	So if definite:
		template<typename _Cont>
		XXX foo(_Cont<xxx>){ ... }
	input complex<_Ty>, need least 1 times conversion, such situation not correct match Container<_Ty>
</idea>
</template-compile>*/


#endif