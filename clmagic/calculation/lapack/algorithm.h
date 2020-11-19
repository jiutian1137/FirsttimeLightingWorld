//+------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache License 2.0
// ------------------------------------------------------------------------------------+
#pragma once
#include "vector.h"
#include "matrix.h"

real32_t s_curve(real32_t t) {
	const real32_t tt = t * t;
	const real32_t ttt = tt * t;
	return tt * 3.0F - ttt * 2.0F;
}
real64_t s_curve(real64_t t) {
	const real64_t tt = t * t;
	const real64_t ttt = tt * t;
	return tt * 3.0 - ttt * 2.0;
}

inline real32_t fade(real32_t t) {
	const real32_t ttt = t * t * t;
	const real32_t tttt = ttt * t;
	const real32_t ttttt = tttt * t;
	return ttttt * 6.0F - tttt * 15.0F + ttt * 10.0F;
}
inline real64_t fade(real64_t t) {
	const real64_t ttt = t * t * t;
	const real64_t tttt = ttt * t;
	const real64_t ttttt = tttt * t;
	return ttttt * 6.0 - tttt * 15.0 + ttt * 10.0;
}

namespace calculation {
	template<typename T, typename W>
	T weighted_mean(const T* value_array, const W* weight_array, size_t array_size) {
		assert(array_size != 0);

		T total_value = value_array[0] * weight_array[0];
		W total_weight = weight_array[0];
		for (size_t i = 1; i != array_size; ++i) {
			total_value += value_array[i] * weight_array[i];
			total_weight += weight_array[i];
		}

		return total_value / total_weight;
		/*<idea>
											1
		value_mean = sum(value_array) * ----------
										array_size

											1 * c_weight
		value_mean = sum(value_array) * ---------------------
										array_size * c_weight

															 1
		value_mean = sum(value_array * c_weight) * ---------------------
												   array_size * c_weight
																		1
		weighted_value_mean = sum(value_array * weight_array) * -------------------
																 sum(weight_array)
		</idea>*/
	}


#pragma region vector operation
	//// dot(a,b) = magnitude(a)*magnitude(b)*cos(a,b)
	//template<typename _Vty>
	//	requires requires(_Vty __v) { __v.dot(__v); } inline
	//auto dot(const _Vty& _X, const _Vty& _Y) -> decltype(_X.dot(_Y)) {
	//	return _X.dot(_Y);
	//}
	//
	//template<typename _Ty1, typename _Ty2>
	//	requires std::is_scalar_v<_Ty1> && std::is_scalar_v<_Ty2> inline
	//auto dot(_Ty1 _X, _Ty2 _Y) -> decltype(_X * _Y) {
	//	return _X * _Y;
	//}

	//template<typename _Vty> inline
	//_Vty cross2(const _Vty& _X) {
	//	/*<idea>
	//		[ i  j  ]
	//		[ lx ly ]
	//	</idea>*/
	//	return _Vty{ _X[1], -_X[0] };
	//}
	//
	//// magnitude(cross3(a,b)) = magnitude(a)*magnitude(b)*sin(a,b)
	//template<typename _Vty1, typename _Vty2> inline
	//_Vty1 cross3(const _Vty1& _X, const _Vty2& _Y) {
	//	/*<idea>
	//		[ i  j  k  ]      
	//		[ lx ly lz ] = i*1*det(minor(0,0)) + j*-1*det(minor(0,1)) + k*1*det(minor(0,2)), 1.determinat expand
	//		[ rx ry rz ]
	//		             = vector{ det(minor(0,0)), -det(minor(0,1)), det(minor(0,2)) }      2.cast to vector
	//	</idea>*/
	//	return _Vty1{
	//		_X[1]* _Y[2] - _X[2]* _Y[1],
	//		_X[2]* _Y[0] - _X[0]* _Y[2],
	//		_X[0]* _Y[1] - _X[1]* _Y[0] };
	//}
	//
	//template<typename _Vty1, typename _Vty2, typename _Vty3> inline
	//_Vty1 cross4(const _Vty1& v0, const _Vty2& v1, const _Vty3& v2) {
	//	/*<idea>
	//		[   i       j      k     u  ]
	//		[ v0.x    v0.y   v0.z  v0.w ]
	//		[ v1.x    v1.y   v1.z  v1.w ] = i*1*det(minor(0,0)) + j*-1*det(minor(0,1)) + k*1*det(minor(0,2)) + u*-1*det(minor(0,3)), 1.determinat expand
	//		[ v2.x    v2.y   v2.z  v2.w ]
	//		    |      | |    |      |    = vector{ +(v0.y*detC - v0.z*detE + v0.w*detB),
	//		    +-detA-+-detB-+-detC-+              -(v0.x*detC - v0.z*detF + v0.w*detD),
	//		    |        |    |      |              +(v0.x*detE - v0.y*detF + v0.w*detA),
	//		    +---detD-+----+      |              -(v0.x*detB - v0.y*detD + v0.z*detA) }
	//		    |        |           |
	//			|   	 +----detE---+
	//			|                    |
	//			+-----detF-----------+
	//	</idea>*/
	//	const auto detA = v1[0] * v2[1] - v1[1] * v2[0];
	//	const auto detB = v1[1] * v2[2] - v1[2] * v2[1];
	//	const auto detC = v1[2] * v2[3] - v1[3] * v2[2];
	//	const auto detD = v1[0] * v2[2] - v1[2] * v2[0];
	//	const auto detE = v1[1] * v2[3] - v1[3] * v2[1];
	//	const auto detF = v1[0] * v2[3] - v1[3] * v2[0];
	//	return _Vty1{
	//		  v0[1]*detC - v0[2]*detE + v0[3]*detB,
	//		-(v0[0]*detC - v0[2]*detF + v0[3]*detD),
	//		  v0[0]*detE - v0[1]*detF + v0[3]*detA,
	//		-(v0[0]*detB - v0[1]*detD + v0[2]*detA) };
	//}

	//template<typename _Ty>
	//	requires requires(_Ty __v) { __v.dot(__v); } inline
	//auto magnitude(const _Ty& _X) {
	//	return sqrt(dot(_X, _X));
	//}

	//template<typename _Ty>
	//	requires requires(_Ty __v) { __v.dot(__v); } inline
	//_Ty normalize(const _Ty& _X) { 
	//	return _X / magnitude(_X);
	//}
	//
	//template<typename _Vty>
	//	requires requires(_Vty __v) { __v.dot(__v); } inline
	//_Vty proj(const _Vty& _X, const _Vty& _Proj) {// _X proj to _Proj
	//	return dot(_X, _Proj) / dot(_Proj, _Proj) * _Proj;
	//}

	//template<typename _Vty>
	//	requires requires(_Vty __v) { __v.dot(__v); } inline
	//_Vty reflect(const _Vty& _V, const _Vty& _N) {
	//	return (_V - proj(_V, _N) * 2);
	//	/*
	//		/|\
	//	   / 2*proj(d, n)
	//	  /  |  \
	//	 /   |
	//	/____|    \
	//	\    |    / Result
	//	 v   |   /
	//	  \  n  /
	//	   \ | /
	//	____\|/_________
	//	*/
	//}

	//template<typename _Iter>
	//void orthogonal(_Iter _First, _Iter _Last) {
	//	auto _Where = std::next(_First);
	//	for ( ; _Where != _Last; ++_Where) {
	//		auto _Prev = _First;
	//		for ( ; _Prev != _Where; ++_Prev) {
	//			*_Where -= proj(*_Where, *_Prev);
	//		}
	//	}
	//}
#pragma endregion

	template<typename _OutTy, typename _InTy> inline
	void _Shuffle_assign(_OutTy& _Dest, size_t _Dest_index, const _InTy& _Source, size_t _Source_index) {
		_Dest[_Dest_index] = _Source[_Source_index];
	}
	
	template<typename _OutTy, typename _InTy, typename ..._Tys>
	void _Shuffle_assign(_OutTy& _Dest, size_t _Dest_start_index, const _InTy& _Source, size_t _Source_index, _Tys... _Source_next_indices) {
		_Dest[_Dest_start_index] = _Source[_Source_index];
		_Shuffle_assign(_Dest, _Dest_start_index+1, _Source, _Source_next_indices...);
	}

	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0) {
		return _OutTy{ _Source[i0] };
	}
	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1) {
		return _OutTy{ _Source[i0], _Source[i1] };
	}
	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1, size_t i2) {
		return _OutTy{ _Source[i0], _Source[i1], _Source[i2] };
	}
	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1, size_t i2, size_t i3) {
		return _OutTy{ _Source[i0], _Source[i1], _Source[i2], _Source[i3] };
	}
	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
		return _OutTy{ _Source[i0], _Source[i1], _Source[i2], _Source[i3], _Source[i4] };
	}
	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) {
		return _OutTy{ _Source[i0], _Source[i1], _Source[i2], _Source[i3], _Source[i4], _Source[i5] };
	}
	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) {
		return _OutTy{ _Source[i0], _Source[i1], _Source[i2], _Source[i3], _Source[i4], _Source[i5], _Source[i6] };
	}
	template<typename _OutTy, typename _InTy> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7) {
		return _OutTy{ _Source[i0], _Source[i1], _Source[i2], _Source[i3], _Source[i4], _Source[i5], _Source[i6], _Source[i7] };
	}
	template<typename _OutTy, typename _InTy, typename ..._Tys> inline
	_OutTy shuffle(const _InTy& _Source, size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7, _Tys... _Selector) {
		_OutTy _Dest = shuffle<_OutTy>(_Source, i0, i1, i2, i3, i4, i5, i6, i7);
		_Shuffle_assign(_Dest, 7+1, _Source, _Selector...);
		return _Dest;
	}

	template<typename _InTy, typename _OutTy, typename ..._Tys> inline
	void shuffle(const _InTy& _Source, _OutTy& _Dest, _Tys... _Selector) {
		_Shuffle_assign(_Dest, 0, _Source, _Selector...);
	}

	/*<example> 
		vector3 v2 = shuffle<vector3>(v1, 1, 0, 1, ...);
		shuflle(v1, v2, 1, 0, 1, ...);
	</example>*/

	/*template<typename ..._Tys> constexpr 
	size_t types_size_v = std::tuple_size_v<std::tuple<_Tys...>>;*/

}// namespace clmagic

namespace StefanGustavson {
	template<typename T> inline
	T cnoise2(calculation::vector2<T> P) {
		static_assert( std::is_floating_point_v<T>, "real StefanGustavson::cnoise2<##ERROR##>(vec2 P)" );
		using calculation::vector2;
		using calculation::vector4;

		const auto permute = [](vector4<T> _X) { return mod( _X * _X * static_cast<T>(34) + _X, static_cast<T>(289) ); };

		// 1. Compute noise cell coordinates and offsets
		vector4<T> Pi  = floor(vector4<T>{ P[0], P[1], P[0], P[1] }) + vector4<T>{ static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(1) };
			       Pi  = mod(Pi, static_cast<T>(289));
		vector4<T> Pf  = frac(vector4<T>{ P[0], P[1], P[0], P[1] }) - vector4<T>{ static_cast<T>(0), static_cast<T>(0), static_cast<T>(1), static_cast<T>(1) };
		vector2<T> v00 = vector2<T>{ Pf[0], Pf[1] };
		vector2<T> v10 = vector2<T>{ Pf[2], Pf[1] };
		vector2<T> v01 = vector2<T>{ Pf[0], Pf[3] };
		vector2<T> v11 = vector2<T>{ Pf[2], Pf[3] };

		// 2. Compute gradient_vectors for <four> corners
		vector4<T> ix  = vector4<T>{ Pi[0], Pi[2], Pi[0], Pi[2] };
		vector4<T> iy  = vector4<T>{ Pi[1], Pi[1], Pi[3], Pi[3] };
		vector4<T> i   = permute( permute( ix ) + iy );
		vector4<T> gx  = frac(i / static_cast<T>(41)) * static_cast<T>(2) - static_cast<T>(1);
		vector4<T> gy  = abs(gx) - static_cast<T>(0.5);
			       gx  = gx - floor(gx + static_cast<T>(0.5));
		vector2<T> g00 = normalize(vector2<T>{ gx[0], gy[0] });
		vector2<T> g10 = normalize(vector2<T>{ gx[1], gy[1] });
		vector2<T> g01 = normalize(vector2<T>{ gx[2], gy[2] });
		vector2<T> g11 = normalize(vector2<T>{ gx[3], gy[3] });
		
		// 3. Compute gradient weights
		T w00 = dot(g00, v00);
		T w10 = dot(g10, v10);
		T w01 = dot(g01, v01);
		T w11 = dot(g11, v11);

		// 4. Compute bilinear interpolation of weights
		T fade_x = fade(Pf[0]);
		T fade_y = fade(Pf[1]);
		return lerp(lerp(w00, w10, fade_x), lerp(w01, w11, fade_x), fade_y) * static_cast<T>(2.2);
	}

	template<typename T> inline
	T cnoise3(calculation::vector3<T> P) {
		static_assert( std::is_floating_point_v<T>, "real StefanGustavson::cnoise3<##ERROR##>(vec3 P)" );
		using calculation::vector3;
		using calculation::vector4;
		using calculation::shuffle;

		const auto permute = [](vector4<T> _X) { return mod(_X * _X * static_cast<T>(34) + _X, static_cast<T>(289)); };

		// 1. Compute noise cell coordinates and offsets
		vector3<T> Pi0 = floor(P);
		vector3<T> Pi1 = Pi0 + static_cast<T>(1);
			       Pi0 = mod( Pi0, static_cast<T>(289) );
			       Pi1 = mod( Pi1, static_cast<T>(289) );
		vector4<T> ix  = vector4<T>{ Pi0[0], Pi1[0], Pi0[0], Pi1[0] };
		vector4<T> iy  = vector4<T>{ Pi0[1], Pi0[1], Pi1[1], Pi1[1] };
		vector4<T> iz0 = shuffle<vector4<T>>( Pi0, 2,2,2,2 );
		vector4<T> iz1 = shuffle<vector4<T>>( Pi1, 2,2,2,2 );

		vector3<T> v000 = frac( P );
		vector3<T> v111 = v000 - static_cast<T>(1);
		vector3<T> v100 = vector3<T>{ v111[0], v000[1], v000[2] };
		vector3<T> v010 = vector3<T>{ v000[0], v111[1], v000[2] };
		vector3<T> v110 = vector3<T>{ v111[0], v111[1], v000[2] };
		vector3<T> v001 = vector3<T>{ v000[0], v000[1], v111[2] };
		vector3<T> v101 = vector3<T>{ v111[0], v000[1], v111[2] };
		vector3<T> v011 = vector3<T>{ v000[0], v111[1], v111[2] };

		// 2. Compute gradient_vectors
		vector4<T> ixy  = permute( permute( ix ) + iy );
		vector4<T> ixy0 = permute( ixy + iz0 );
		vector4<T> ixy1 = permute( ixy + iz1 );

		vector4<T> gx0  = ixy0 * static_cast<T>(1.0/7.0);
		vector4<T> gy0  = frac( floor(gx0) * static_cast<T>(1.0/7.0) ) - static_cast<T>(0.5);
			       gx0  = frac( gx0 );
		vector4<T> gz0  = -(abs(gx0) + abs(gy0)) + static_cast<T>(0.5);
		vector4<T> sz0  = gz0 <= static_cast<T>(0);
			       gx0 -= sz0 * ( (gx0 >= static_cast<T>(0)) - static_cast<T>(0.5) );
			       gy0 -= sz0 * ( (gy0 >= static_cast<T>(0)) - static_cast<T>(0.5) );

		vector4<T> gx1  = ixy1 * static_cast<T>(1.0/7.0);
		vector4<T> gy1  = frac( floor(gx1) * static_cast<T>(1.0/7.0) ) - static_cast<T>(0.5);
			       gx1  = frac( gx1 );
		vector4<T> gz1  = -(abs(gx1) + abs(gy1)) + static_cast<T>(0.5);
		vector4<T> sz1  = gz1 <= static_cast<T>(0);
			       gx1 -= sz1 * ( (gx1 >= static_cast<T>(0)) - static_cast<T>(0.5) );
			       gy1 -= sz1 * ( (gy1 >= static_cast<T>(0)) - static_cast<T>(0.5) );

		vector3<T> g000 = normalize(vector3<T>{ gx0[0], gy0[0], gz0[0] });
		vector3<T> g100 = normalize(vector3<T>{ gx0[1], gy0[1], gz0[1] });
		vector3<T> g010 = normalize(vector3<T>{ gx0[2], gy0[2], gz0[2] });
		vector3<T> g110 = normalize(vector3<T>{ gx0[3], gy0[3], gz0[3] });
		vector3<T> g001 = normalize(vector3<T>{ gx1[0], gy1[0], gz1[0] });
		vector3<T> g101 = normalize(vector3<T>{ gx1[1], gy1[1], gz1[1] });
		vector3<T> g011 = normalize(vector3<T>{ gx1[2], gy1[2], gz1[2] });
		vector3<T> g111 = normalize(vector3<T>{ gx1[3], gy1[3], gz1[3] });

		// 3. Compute gradient weights
		T w000 = dot(g000, v000);
		T w100 = dot(g100, v100);
		T w010 = dot(g010, v010);
		T w110 = dot(g110, v110);
		T w001 = dot(g001, v001);
		T w101 = dot(g101, v101);
		T w011 = dot(g011, v011);
		T w111 = dot(g111, v111);

		// 4. Compute trilinear interpolation of weights
		T fade_x = fade(v000[0]);
		T fade_y = fade(v000[1]);
		T fade_z = fade(v000[2]);
		return lerp( lerp( lerp(w000, w100, fade_x), 
					       lerp(w010, w110, fade_x), fade_y ),
				     lerp( lerp(w001, w101, fade_x), 
						   lerp(w011, w111, fade_x), fade_y), fade_z ) * static_cast<T>(2.2);
	}
}

namespace Worley {
	template<typename T, typename _Fn>
	T Cells2(calculation::vector2<T> P, _Fn hash22) {
		using calculation::vector2;

		vector2<T> Pi  = floor(P);
		vector2<T> rhs = P - Pi;

		T d = static_cast<T>(8);
		for (int8_t i = -1; i <= 1; ++i) {
			for (int8_t j = -1; j <= 1; ++j) {
				vector2<T> g    = vector2<T>{ static_cast<T>(j), static_cast<T>(i) };
				vector2<T> lhs  = g + hash22(Pi + g);
				vector2<T> dist = rhs - lhs;
				d = min(d, dot(dist, dist));
			}
		}

		return sqrt(d);
	}

	template<typename T, typename _Fn>
	T Cells3(calculation::vector3<T> P, _Fn hash33) {
		using calculation::vector3;

		vector3<T> Pi  = floor(P);
		vector3<T> rhs = P - Pi;

		T d = static_cast<T>(8);
		for (int8_t i = -1; i <= 1; ++i) {
			for (int8_t j = -1; j <= 1; ++j) {
				for (int8_t k = -1; k <= 1; ++k) {
					vector3<T> g    = vector3<T>{ static_cast<T>(k), static_cast<T>(j), static_cast<T>(i) };
					vector3<T> lhs  = g + hash33(Pi + g);
					vector3<T> dist = rhs - lhs;
					d = min(d, dot(dist, dist));
				}
			}
		}

		return sqrt(d);
	}

	template<typename T> inline
	T Cells2(calculation::vector2<T> P) {
		auto hash22 = 
			[](calculation::vector2<T> x) {
				T a = dot(x, calculation::vector2<T>{ 127.1F, 311.7F });
				T b = dot(x, calculation::vector2<T>{ 269.5F, 183.3F });
				x[0] = a;
				x[1] = b;
				return frac( sin(x) * static_cast<T>(43758.5453F) );
			};

		return Cells2(P, hash22);
	}

	template<typename T> inline
	T Cells3(calculation::vector3<T> P) {
		auto hash33 = 
			[](calculation::vector3<T> x) {
				T a = dot(x, calculation::vector3<T>{ 127.1F, 311.7F, 74.7F });
				T b = dot(x, calculation::vector3<T>{ 269.5F, 183.3F, 246.1F });
				T c = dot(x, calculation::vector3<T>{ 113.5F, 271.9F, 124.6F });
				x[0] = a;
				x[1] = b;
				x[2] = c;
				return frac(sin(x) * static_cast<T>(43758.5453F));
			};

		return Cells3(P, hash33);
	}
}// namespace Worley




#define _DEFINE_VECTOR_UNARY_FUNC_(FNAME, TY, LEFT) \
using _SclTy = calculation::vector_scalar_t<TY>;  \
using _BlkTy = calculation::vector_block_t<TY>;   \
typedef _BlkTy(*_Fn)(_BlkTy);                 \
TY result_vector;                             \
calculation::transform_vector<_BlkTy>(##LEFT##, result_vector, _Fn(##FNAME##)); \
return std::move(result_vector);

#define _DEFINE_VECTOR_BINARY_FUNC_(FNAME, TY, LEFT, RIGHT) \
using scalar_type = calculation::vector_scalar_t<TY>; \
using block_type  = calculation::vector_block_t<TY>;  \
typedef block_type(*_Fn)(block_type, block_type); \
TY result_vector;                                 \
calculation::transform_vector<block_type>(##LEFT##, ##RIGHT##, result_vector, _Fn(##FNAME##)); \
return std::move(result_vector)

//#include "vectorsimd.h"

template<typename _Ty> inline
_Ty VECTOR_SIGN(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(SIGN, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_ABS(const _Ty& vector) {
	_DEFINE_VECTOR_UNARY_FUNC_(ABS, _Ty, vector);
}

template<typename _Ty> inline
_Ty VECTOR_FLOOR(const _Ty& vector) {
	_DEFINE_VECTOR_UNARY_FUNC_(FLOOR, _Ty, vector);
}

template<typename _Ty> inline
_Ty VECTOR_CEIL(const _Ty& vector) {
	_DEFINE_VECTOR_UNARY_FUNC_(CEIL, _Ty, vector);
}

template<typename _Ty> inline
_Ty VECTOR_TRUNC(const _Ty& vector) {
	_DEFINE_VECTOR_UNARY_FUNC_(TRUNC, _Ty, vector);
}

template<typename _Ty> inline
_Ty VECTOR_FRAC(const _Ty& vector) {
	_DEFINE_VECTOR_UNARY_FUNC_(FRAC, _Ty, vector);
}

template<typename _Ty> inline
_Ty VECTOR_ROUND(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(ROUND, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_MOD(const _Ty& numbers, const _Ty& divisors) {
	_DEFINE_VECTOR_BINARY_FUNC_(MOD, _Ty, numbers, divisors);
}
	
	
template<typename _Ty> inline
_Ty VECTOR_EXP(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(EXP, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_LN(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(LN, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_POWER(const _Ty& numbers, const _Ty& powers) {
	_DEFINE_VECTOR_BINARY_FUNC_(POWER, _Ty, numbers, powers);
}

template<typename _Ty> inline 
_Ty VECTOR_LOG(const _Ty& bases, const _Ty& numbers) {
	return VECTOR_LN(numbers) / VECTOR_LN(bases);
}

template<typename _Ty> inline
_Ty VECTOR_SQRT(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(SQRT, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_CBRT(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(CBRT, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_RSQRT(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(RSQRT, _Ty, numbers);
}


template<typename _Ty> inline
_Ty VECTOR_SIN(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(SIN, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_COS(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(COS, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_TAN(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(TAN, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_ASIN(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(ASIN, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_ACOS(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(ACOS, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_ATAN(const _Ty& numbers) {
	_DEFINE_VECTOR_UNARY_FUNC_(ATAN, _Ty, numbers);
}

template<typename _Ty> inline
_Ty VECTOR_ATAN2(const _Ty& y_numbers, const _Ty& x_numbers) {
	_DEFINE_VECTOR_BINARY_FUNC_(ATAN2, _Ty, y_numbers, x_numbers);
}


// SUM( vector[i] )
template<typename _Ty> inline
auto VECTOR_SUM(const _Ty& vector) -> calculation::vector_scalar_t<_Ty> {
	using scalar_type = calculation::vector_scalar_t<_Ty>;
	using block_type  = calculation::vector_block_t<_Ty>;
	return calculation::reduce_vector<block_type>(vector, static_cast<scalar_type>(0));
}

// SUM( _Transform(vector[i]) )
template<typename _Ty, typename _Fn> inline
auto VECTOR_SUM(const _Ty& vector, _Fn _Transform_op) -> calculation::vector_scalar_t<_Ty> {
	using scalar_type = calculation::vector_scalar_t<_Ty>;
	using block_type  = calculation::vector_block_t<_Ty>;
	return calculation::transform_reduce_vector<block_type>(vector, static_cast<scalar_type>(0), std::plus<>(), std::plus<>(), _Transform_op);
}


// { vector scalar product, SUM( vector1[i] * vector2[i] ) }
template<typename _Ty> inline
auto VECTOR_DOT(const _Ty& vector1, const _Ty& vector2) -> calculation::vector_scalar_t<_Ty> {
	using scalar_type = calculation::vector_scalar_t<_Ty>;
	using block_type  = calculation::vector_block_t<_Ty>;
	return calculation::transform_reduce_vector<block_type>(vector1, vector2, static_cast<scalar_type>(0),
		std::plus<>(), std::plus<>(), std::multiplies<>());
}

// { vector length, SQRT( DOT(vector, vector) ) or NORM(vector,2) }
template<typename _Ty> inline
auto VECTOR_LENGTH(const _Ty& vector) -> calculation::vector_scalar_t<_Ty> {
	return SQRT(VECTOR_DOT(vector, vector));
}

// { vector squared length, DOT(vector, vector) }
template<typename _Ty> inline
auto VECTOR_SQLENGTH(const _Ty& vector) -> calculation::vector_scalar_t<_Ty> {
	return VECTOR_DOT(vector, vector);
}

// { distance of from vector1 to vector2, VECTOR_LENGTH(vector2 - vector1) }
template<typename _Ty> inline
auto VECTOR_DISTANCE(const _Ty& vector1, const _Ty& vector2) -> calculation::vector_scalar_t<_Ty> {
	return VECTOR_LENGTH(vector2 - vector1);
}

// { squared_distance of from vector1 and vector2, VECTOR_SQLENGTH(vector2 - vector1) }
template<typename _Ty> inline
auto VECTOR_SQDISTANCE(const _Ty& vector1, const _Ty& vector2) -> calculation::vector_scalar_t<_Ty> {
	return VECTOR_SQLENGTH(vector2 - vector1);
}

// { vector norm , POWER( SUM(vector[i], L), 1/L ) }
template<typename _Ty> inline
auto VECTOR_NORM(const _Ty& vector, int32_t L) -> calculation::vector_scalar_t<_Ty> {
	using scalar_type = calculation::vector_scalar_t<_Ty>;
	using block_type  = calculation::vector_block_t<_Ty>;

	block_type Lb      = calculation::block_traits<block_type>::set( static_cast<scalar_type>(L) );
	auto       powerL = [Lb](block_type x) { return POWER(x, Lb); };
	return POWER( SUM(vector, powerL), 1.0F/L );
}

// { vector cross product }
template<typename _Ty> inline
_Ty VECTOR_CROSS3(const _Ty& vector1, const _Ty& vector2) {
	/*<idea>
		[ i  j  k  ]
		[ lx ly lz ] = i*1*det(minor(0,0)) + j*-1*det(minor(0,1)) + k*1*det(minor(0,2)), 1.determinat expand
		[ rx ry rz ]
						= vector{ det(minor(0,0)), -det(minor(0,1)), det(minor(0,2)) }      2.cast to vector
	</idea>*/
	return _Ty{
		vector1[1] * vector2[2] - vector1[2] * vector2[1],
		vector1[2] * vector2[0] - vector1[0] * vector2[2],
		vector1[0] * vector2[1] - vector1[1] * vector2[0]
	};
}

// { vector cross product }
template<typename _Ty> inline
_Ty VECTOR_CROSS4(const _Ty& vector1, const _Ty& vector2, const _Ty& vector3) {
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
	using real_t = calculation::vector_scalar_t<_Ty>;

	real_t detA = vector2[0] * vector3[1] - vector2[1] * vector3[0];
	real_t detB = vector2[1] * vector3[2] - vector2[2] * vector3[1];
	real_t detC = vector2[2] * vector3[3] - vector2[3] * vector3[2];
	real_t detD = vector2[0] * vector3[2] - vector2[2] * vector3[0];
	real_t detE = vector2[1] * vector3[3] - vector2[3] * vector3[1];
	real_t detF = vector2[0] * vector3[3] - vector2[3] * vector3[0];
	return _Ty{
		  vector1[1]*detC - vector1[2]*detE + vector1[3]*detB,
		-(vector1[0]*detC - vector1[2]*detF + vector1[3]*detD),
		  vector1[0]*detE - vector1[1]*detF + vector1[3]*detA,
		-(vector1[0]*detB - vector1[1]*detD + vector1[2]*detA) 
	};
}

// { vector normalize, LENGTH(result_vector) == 1 }
template<typename _Ty> inline
_Ty VECTOR_NORMALIZE(const _Ty& vector) {
	using real_t = calculation::vector_scalar_t<_Ty>;

	real_t _Sqlength = VECTOR_DOT(vector, vector);
	real_t _Error    = _Sqlength - 1;
	if ( ABS(_Error) > std::numeric_limits<real_t>::epsilon() ) {
		return vector / SQRT(_Sqlength);
	} else {
		return vector;
	}
}