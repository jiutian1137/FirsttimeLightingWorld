//--------------------------------------------------------------------------------------
//  Copyright (c) William-Rowan-Hamilton(1805?865)[16.October.1843 ]
//  All Rights Reserved
//  <Biographie>http://mathshistory.st-andrews.ac.uk/Biographies/Hamilton.html</Biographie>
//--------------------------------------------------------------------------------------
//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//--------------------------------------------------------------------------------------
#pragma once
#ifndef clmagic_calculation_complex_QUATERNION_h_
#define clmagic_calculation_complex_QUATERNION_h_
#include "../lapack/vector.h"
#include <complex>

namespace calculation {
	template<typename _Ty, typename _Traits = calculation::block_traits<_Ty>>
	struct quaternion {
		using traits_type = _Traits;
		using scalar_type = typename _Traits::scalar_type;
		using block_type  = typename _Traits::block_type;

	private:
		using real = scalar_type;

	public:
		quaternion() = default;

		quaternion(const quaternion&) = default;
		
		explicit quaternion(scalar_type _Re)
			: a(_Re),
			b(static_cast<scalar_type>(0)), c(static_cast<scalar_type>(0)), d(static_cast<scalar_type>(0)) {}
		
		quaternion(scalar_type _Re, scalar_type _Im0, scalar_type _Im1, scalar_type _Im2)
			: a(_Re),
			b(_Im0), c(_Im1), d(_Im2) {}

		template<typename VecT>
		quaternion(scalar_type _Re, VecT _Im)
			: a(_Re),
			b(_Im[0]), c(_Im[1]), d(_Im[2]) {}

		quaternion& operator=(const quaternion&) = default;
		
		quaternion& operator=(scalar_type _Re) {
			a = _Re;
			b = c = d = static_cast<scalar_type>(0);
			return *this;
		}
		
		friend quaternion operator+(quaternion q, quaternion r) {
			return quaternion{ q.a + r.a, q.b + r.b, q.c + r.c, q.d + r.d };
		}
		
		friend quaternion operator-(quaternion q, quaternion r) {
			return quaternion{ q.a - r.a, q.b - r.b, q.c - r.c, q.d - r.d };
		}
		
		friend quaternion operator*(quaternion q, quaternion r) {
			// { q.real * r.real - dot(q.imag, r.imag), cross3(q.imag, r.imag) + q.real * r.imag + q.imag * r.real }
			return quaternion{ q.a*r.a - (q.b*r.b + q.c*r.c + q.d*r.d),
							   q.a*r.b + q.b*r.a + (q.c*r.d - q.d*r.c),
							   q.a*r.c + q.c*r.a + (q.d*r.b - q.b*r.d),
							   q.a*r.d + q.d*r.a + (q.b*r.c - q.c*r.b) };
		}
		
		friend quaternion operator/(quaternion q, quaternion r) {
			// { dot(q, r), cross3(r, q) - q.real * r.imag + q.imag * r.real } / dot(r, r)
			scalar_type denominator = r.a*r.a + r.b*r.b + r.c*r.c + r.d*r.d;
			return quaternion{ (+q.a*r.a + q.b*r.b + q.c*r.c + q.d*r.d) / denominator, 
							   (-q.a*r.b + (q.d*r.c - q.c*r.d) + q.b*r.a) / denominator,
							   (-q.a*r.c + (q.b*r.d - q.d*r.b) + q.c*r.a) / denominator,
							   (-q.a*r.d + (q.c*r.b - q.b*r.c) + q.d*r.a) / denominator };
		}
		
		friend quaternion operator+(quaternion q, real r) {
			return quaternion{ q.a + r, q.b, q.c, q.d };
		}
		
		friend quaternion operator-(quaternion q, real r) {
			return quaternion{ q.a - r, q.b, q.c, q.d };
		}
		
		friend quaternion operator*(quaternion q, real r) {
			return quaternion{ q.a * r, q.b * r, q.c * r, q.d * r };
		}
		
		friend quaternion operator/(quaternion q, real r) {
			return quaternion{ q.a / r, q.b / r, q.c / r, q.d / r };
		}

		// a + b*i + c*j + d*k
		scalar_type a;
		scalar_type b, c, d;
	};

	template<typename T>
    quaternion<T> pow(const quaternion<T>& q, int n) {
        if (n > 1) {
            int m = n>>1;
                
            quaternion<T> result = pow(q, m);
                
            result *= result;
                
            if (n != (m<<1)) {
                result *= q; // n odd
            }
                
            return(result);
        } else if (n == 1) {
            return(q);
        } else if (n == 0) {
            return(quaternion<T>(static_cast<T>(1)));
        } else  /* n < 0 */ {
            return(pow(quaternion<T>(static_cast<T>(1))/q, -n));
        }
    }

	template<typename T> inline
	quaternion<T> conj(quaternion<T> q) {
		return quaternion<T>{ q.a, 
						     -q.b, -q.c, -q.d };
	}

	template<typename T> inline
	T sup(quaternion<T> q) {
		return max( max(abs(q.a), abs(q.b)), max(abs(q.c), abs(q.d)) );
	}

	template<typename T> inline
	T abs(quaternion<T> q) {
		T maxim = sup(q);
		T mixam = static_cast<T>(1) / maxim;
		return maxim * sqrt(pow(q.a * mixam, 2) + 
							pow(q.b * mixam, 2) + 
							pow(q.c * mixam, 2) + 
							pow(q.d * mixam, 2));
	}

	template<typename T> inline
	T real(quaternion<T> q) {
		return q.a;
	}

	template<typename T> inline
	T norm(quaternion<T> q) {
		return real(q * conj(q));
	}
	
	template<typename QuatT/*noAuto*/, typename VecT, typename T> inline
	QuatT polar(VecT axis, T angle) {
		return QuatT{ cos(angle / static_cast<T>(2)),
					  axis * sin(angle / static_cast<T>(2)) };
	}
	
	template<typename T> inline
	quaternion<T> inv(quaternion<T> q) {
		return conj(q) * (static_cast<T>(1) / norm(q));
	}

	template<typename T> inline
	quaternion<T> normalize(quaternion<T> q) {
		T discrim = sqrt(q.a*q.a + q.b*q.b + q.c*q.c + q.d*q.d);
		  discrim = (discrim <= std::numeric_limits<T>::epsilon() ? static_cast<T>(1) : discrim);
		return q / discrim;
	}

	template<typename T> inline
	quaternion<T> lerp(quaternion<T> q1, quaternion<T> q2, T t) {
		return q1 + (q2 - q1) * t;
	}

	template<typename T> inline
	quaternion<T> slerp(quaternion<T> q1, quaternion<T> q2, T t) {
		// <Reference type="book"> { pbrt, Section[2.9.2][Quaternion Interpolation] } </Reference>
		T cosTheta = q1.a*q2.a + q1.b*q2.b + q1.c*q2.c + q1.d*q2.d;
		if (cosTheta > static_cast<T>(0.99995)) {// q1 q2 parallel
			return normalize(lerp(q1, q2, t));
		} else {
			T theta  = acos( clamp(cosTheta, static_cast<T>(-1), static_cast<T>(1)) );
			T thetap = theta * t;
			quaternion<T> qperp = normalize(q2 - q1 * cosTheta);
			return ( q1 * cos(thetap) + qperp * sin(thetap) );
		}
	}

	template<typename T> inline
	quaternion<T> rotate(quaternion<T> q, quaternion<T> p) {
		return q * p * inv(q);
	}

	template<typename T> inline
	quaternion<T> normalized_rotate(quaternion<T> q, quaternion<T> p) {
		//assert( abs(norm(q) - static_cast<T>(1)) <= std::numeric_limits<T>::epsilon() );// error very large...
		return q * p * conj(q);
	}





	template<typename T> inline
	vector3<T> rotate(quaternion<T> q, vector3<T> p) {
		quaternion<T> qpq = rotate(q, quaternion<T>{ static_cast<T>(0), p[0], p[1], p[2] });
		return vector3<T>{ qpq[0], qpq[1], qpq[2] };
	}

	template<typename T> inline
	vector3<T> normalized_rotate(quaternion<T> q, vector3<T> p) {
		quaternion<T> qpq = normalized_rotate(q, quaternion<T>{ static_cast<T>(0), p[0], p[1], p[2] });
		return vector3<T>{ qpq[0], qpq[1], qpq[2] };
	}



#ifdef _INCLUDED_MM2
	template<>
	struct __declspec(align(16)) quaternion<float, calculation::block_traits<__m128>> {
		using traits_type = calculation::block_traits<__m128>;
		using scalar_type = typename traits_type::scalar_type;
		using block_type  = typename traits_type::block_type;

		quaternion() = default;
		
		explicit quaternion(__m128 _a_b_c_d) : a_b_c_d(_a_b_c_d) {}

		explicit quaternion(float _Re)
			: a_b_c_d( _mm_setr_ps(_Re, 0.0f, 0.0f, 0.0f) ) {}
		
		quaternion(float _Re, float _Im0, float _Im1, float _Im2)
			: a_b_c_d( _mm_setr_ps(_Re, _Im0, _Im1, _Im2) ) {}

		template<typename VecT>
		quaternion(float _Re, VecT _Im)
			: a_b_c_d( _mm_setr_ps(_Re, _Im[0], _Im[1], _Im[2]) ) {}


		friend quaternion operator+(quaternion q, quaternion r) {
			return quaternion(_mm_add_ps(q.a_b_c_d, r.a_b_c_d));
		}
		friend quaternion operator-(quaternion q, quaternion r) {
			return quaternion(_mm_sub_ps(q.a_b_c_d, r.a_b_c_d));
		}
		friend quaternion operator*(quaternion q, quaternion r) {
			__m128 qa = _mm_permute_ps(q.a_b_c_d, _MM_SHUFFLER(0, 0, 0, 0));
			__m128 ra = _mm_permute_ps(r.a_b_c_d, _MM_SHUFFLER(0, 0, 0, 0));

			__m128 qcdb = _mm_permute_ps(q.a_b_c_d, _MM_SHUFFLER(1, 2, 3, 1));
			__m128 qdbc = _mm_permute_ps(q.a_b_c_d, _MM_SHUFFLER(1, 3, 1, 2));
			__m128 rcdb = _mm_permute_ps(r.a_b_c_d, _MM_SHUFFLER(1, 2, 3, 1));
			__m128 rdbc = _mm_permute_ps(r.a_b_c_d, _MM_SHUFFLER(1, 3, 1, 2));
			__m128 cross_imag = _mm_sub_ps(_mm_mul_ps(qcdb, rdbc), _mm_mul_ps(qdbc, rcdb));
			
			__m128 dot_imag = _mm_mul_ps(qcdb, rcdb);
			dot_imag = _mm_add_ss(dot_imag, _mm_permute_ps(dot_imag, _MM_SHUFFLER(1,1,1,1)));
			dot_imag = _mm_add_ss(dot_imag, _mm_permute_ps(dot_imag, _MM_SHUFFLER(2,2,2,2)));
			
			__m128 result_real = _mm_sub_ps(_mm_mul_ss(qa, ra), dot_imag);
			__m128 result_imag = _mm_add_ps(_mm_add_ps(_mm_mul_ps(qa, r.a_b_c_d), _mm_mul_ps(ra, q.a_b_c_d)), cross_imag);
			return quaternion(_mm_shuffler_ps(
				_mm_shuffler_ps(result_real, result_imag, _MM_SHUFFLER(0, 0, 1, 1)),// { real, real, imagX, imagX }
				result_imag,// { x, imagX, imagY, imagZ }
				_MM_SHUFFLER(0, 2, 2, 3) ));
		}
		
		//friend quaternion operator-(quaternion q, quaternion r) {
		//	return quaternion{ q.a - r.a, q.b - r.b, q.c - r.c, q.d - r.d };
		//}
		//
		//friend quaternion operator*(quaternion q, quaternion r) {
		//	// { q.real * r.real - dot(q.imag, r.imag), cross3(q.imag, r.imag) + q.real * r.imag + q.imag * r.real }
		//	return quaternion{ q.a*r.a - (q.b*r.b + q.c*r.c + q.d*r.d),
		//					   q.a*r.b + q.b*r.a + (q.c*r.d - q.d*r.c),
		//					   q.a*r.c + q.c*r.a + (q.d*r.b - q.b*r.d),
		//					   q.a*r.d + q.d*r.a + (q.b*r.c - q.c*r.b) };
		//}
		//
		//friend quaternion operator/(quaternion q, quaternion r) {
		//	// { dot(q, r), cross3(r, q) - q.real * r.imag + q.imag * r.real } / dot(r, r)
		//	scalar_type denominator = r.a*r.a + r.b*r.b + r.c*r.c + r.d*r.d;
		//	return quaternion{ (+q.a*r.a + q.b*r.b + q.c*r.c + q.d*r.d) / denominator, 
		//					   (-q.a*r.b + (q.d*r.c - q.c*r.d) + q.b*r.a) / denominator,
		//					   (-q.a*r.c + (q.b*r.d - q.d*r.b) + q.c*r.a) / denominator,
		//					   (-q.a*r.d + (q.c*r.b - q.b*r.c) + q.d*r.a) / denominator };
		//}

		// a + b*i + c*j + d*k
		__m128 a_b_c_d;
	};
	
	using m128quaternion = quaternion<float, block_traits<__m128>>;


#endif

}// namespace calculation

//
//	quaternion rotate(quaternion p) const {// theta/2 in polar(...)
//		const auto& q = *this;
//		const auto  q_inv = quaternion(real(), -imag());// conj(*this);
//		return (q * p * q_inv);
//	}
//	quaternion operator()(quaternion p) const {// theta/2 in polar(...)
//		return this->rotate(p);
//	}
//	template<typename _VecTy>
//	_VecTy operator()(_VecTy p) const {
//		const quaternion qpq = this->rotate(quaternion(static_cast<_SclTy>(0), p[0], p[1], p[2]));
//		return reinterpret_cast<const _VecTy&>(qpq);
//	}
//
//	//template<typename _QuatTy> inline
//	//_QuatTy identity() {// (0, i*1, j*1, k*1)
//	//	return _QuatTy(0, 1, 1, 1);
//	//}
//
//	using m128quaternion = quaternion<float, block_traits<__m128>>;
//
//	
//
//	/*<theorem>
//		conj(conj(q)) = q
//		conj(q + r)   = conj(q) + conj(r)
//		conj(q * r)   = conj(r) * conj(q)
//	</theorem>*/
//
//	// q = cos(theta) + axis*sin(theta) = pow(e, axis*theta). 
//
//}// namespace WilliamRowanHamilton
//
//template<typename T>
//T real(calculation::quaternion<T> q) {
//	return q.real;
//}
//
//template<typename T>
//calculation::quaternion<T> unreal(calculation::quaternion<T> q) {
//	return calculation::quaternion<T>{ static_cast<T>(0), q.imag[0], q.imag[1], q.imag[2] };
//}
//
//template<typename T>
//calculation::quaternion<T> conj(calculation::quaternion<T> q) {
//	return calculation::quaternion<T>{ q.real, -q.imag[0], -q.imag[1], -q.imag[2] };
//}
//
//template<typename T>
//T norm(calculation::quaternion<T> q) {
//	return q * conj(q);
//}
//
//
//namespace QUATERNION {
//	template<typename _QuatTy> inline
//	_QuatTy IDENTITY() {// (0, i*1, j*1, k*1)
//		return _QuatTy(0, 1, 1, 1);
//	}
//
//	template<typename _QuatTy, typename _UvecTy, typename _RealTy> inline
//	_QuatTy POLAR(_UvecTy axis_unitvector, _RealTy radian) {// dived 2, because p*q*inverse(q) is rotate 2*radian
//		return WilliamRowanHamilton::polar(axis_unitvector, radian);
//	}
//
//	template<typename _QuatTy>
//		requires requires(_QuatTy __q) { __q.imag(); __q.real(); } inline
//	_QuatTy CONJ(_QuatTy q) {
//		return _QuatTy(q.real(), -q.imag());
//	}
//	
//	template<typename _QuatTy1, typename _QuatTy2> inline
//	auto DOT(_QuatTy1 quaternion1, _QuatTy2 quaternion2) {
//		return WilliamRowanHamilton::scalar_product(quaternion1, quaternion2);
//	}
//
//	template<typename _QuatTy> inline
//	auto NORM(_QuatTy quaternion) {// sqrt( q * conj(q) ) = sqrt( dot(q.imag(), conj(q).imag() + pow(q.real(),2) )
//		return WilliamRowanHamilton::norm(quaternion);
//	}
//
//	//     1
//	// -------- * conj(q)
//	//  norm(q)
//	template<typename _QuatTy> inline
//	_QuatTy INVERSE(_QuatTy quaternion) {
//		return WilliamRowanHamilton::inverse(quaternion);
//	}
//
//	template<typename _QuatTy> inline
//	_QuatTy NORMALIZE(_QuatTy quaternion) {
//		return WilliamRowanHamilton::normalize(quaternion);
//	}
//
//	template<typename _QuatTy, typename _RealTy>
//	_QuatTy SLERP(_QuatTy quaternion1, _QuatTy quaternion2, _RealTy t) {
//		return WilliamRowanHamilton::slerp(quaternion1, quaternion2, t);
//	}
//}

#endif