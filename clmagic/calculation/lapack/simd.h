//--------------------------------------------------------------------------------------
// Copyright (c) 2019 LongJiangnan
// All Rights Reserved
// Apache License 2.0
//--------------------------------------------------------------------------------------
#pragma once
#include "algorithm.h"
#include <intrin.h>

#ifdef _INCLUDED_MM2

namespace Worley {
    inline float _Cells2(__m128 P, __m128(*hash22)(__m128)) {
        __m128 Pi  = _mm_floor_ps(P);
        __m128 rhs = _mm_sub_ps(P, Pi);

        float d = 8.0F;
        for (int8_t i = -1; i <= 1; ++i) {
            for (int8_t j = -1; j <= 1; ++j) {
                __m128 g    = _mm_set_ps(0.0F, 0.0F, static_cast<float>(i), static_cast<float>(j));
                __m128 lhs  = _mm_add_ps(g, hash22(_mm_add_ps(Pi, g)));
                __m128 dist = _mm_sub_ps(rhs, lhs);
                dist = _mm_mul_ps(dist, dist);
                d    = min(d, dist.m128_f32[0] + dist.m128_f32[1]);
            }
        }

        return sqrt(d);
    }

    inline float _Cells3(__m128 P, __m128(*hash33)(__m128)) {
        __m128 Pi  = _mm_floor_ps(P);
        __m128 rhs = _mm_sub_ps(P, Pi);

        float d = 8.0F;
        for (int8_t i = -1; i <= 1; ++i) {
            for (int8_t j = -1; j <= 1; ++j) {
                for (int8_t k = -1; k <= 1; ++k) {
                    __m128 g    = _mm_set_ps(0.0F, static_cast<float>(i), static_cast<float>(j), static_cast<float>(k));
                    __m128 lhs  = _mm_add_ps(g, hash33( _mm_fmod_ps(_mm_add_ps(Pi, g), _mm_set1_ps(4.0F)) ));
                    __m128 dist = _mm_sub_ps(rhs, lhs);
                    dist = _mm_mul_ps(dist, dist);
                    d    = min(d, dist.m128_f32[0] + dist.m128_f32[1] + dist.m128_f32[2]);
                }
            }
        }

        return sqrt(d);
    }

    inline float Cells2(calculation::m128vector<4> P) {
        auto hash22 = 
			[](__m128 x) {
                __m128 a = _mm_mul_ps(x, _mm_set_ps(0.0F, 0.0F, 311.7F, 127.1F) );
                __m128 b = _mm_mul_ps(x, _mm_set_ps(0.0F, 0.0F, 183.3F, 269.5F) );
                x = _mm_set_ps(0.0F, 0.0F, b.m128_f32[0] + b.m128_f32[1], a.m128_f32[0] + a.m128_f32[1]);
                x = _mm_mul_ps(_mm_sin_ps(x), _mm_set1_ps(43758.5453F));
				return _mm_sub_ps(x, _mm_floor_ps(x));
			};

		return _Cells2(reinterpret_cast<__m128&>(P), hash22);
    }

    inline float Cells3(calculation::m128vector<4> P) {
        auto hash33 = 
            [](__m128 x) {
                __m128 a = _mm_mul_ps(x, _mm_set_ps(0.0F, 74.7F, 311.7F, 127.1F));
                __m128 b = _mm_mul_ps(x, _mm_set_ps(0.0F, 246.1F, 183.3F, 269.5F));
                __m128 c = _mm_mul_ps(x, _mm_set_ps(0.0F, 124.6F, 271.9F, 113.5F));
                x = _mm_set_ps(0.0F, c.m128_f32[0]+c.m128_f32[1]+ c.m128_f32[2], b.m128_f32[0]+b.m128_f32[1]+b.m128_f32[2], a.m128_f32[0]+a.m128_f32[1]+a.m128_f32[2]);
                x = _mm_mul_ps(_mm_sin_ps(x), _mm_set1_ps(43758.5453F));
                return _mm_sub_ps(x, _mm_floor_ps(x));
			};

		return _Cells3(reinterpret_cast<__m128&>(P), hash33);
    }
}// namespace Worley

namespace StefanGustavson {
    inline __m128 _mm_mod289_ps(__m128 x) {
        return _mm_fmod_ps(x, _mm_set1_ps(289.0F));
    }

    inline __m128 _mm_random_ps(__m128 x) {
        return _mm_mod289_ps(
            _mm_add_ps(_mm_mul_ps(_mm_mul_ps(x, x), _mm_set1_ps(34.0F)),
                       x)
        );
    }

    inline float _mm_cnoise2_ps(__m128 x) {
		// 1. Compute noise cell coordinates and offsets
        __m128 Gi = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1,0,1,0));// { x[0],x[1],x[0],x[1] }
        __m128 p  = _mm_add_ps( _mm_floor_ps(Gi), _mm_setr_ps(0.0F, 0.0F, 1.0F, 1.0F) );
			   p  = _mm_mod289_ps(p);
		__m128 f  = _mm_sub_ps( _mm_frac_ps(Gi), _mm_setr_ps(0.0F, 0.0F, 1.0F, 1.0F) );
        __m128 v00x10 = _mm_shuffle_ps(f, f, _MM_SHUFFLE(1,1,2,0));// { {f[0], f[1]} X {f[2], f[1]} }
        __m128 v01x11 = _mm_shuffle_ps(f, f, _MM_SHUFFLE(3,3,2,0));// { {f[0], f[3]} X {f[2], f[3]} }

		// 2. Compute gradient_vectors for <four> corners
        __m128 ix = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,0,2,0));// { p[0],p[2],p[0],p[2] }
        __m128 iy = _mm_shuffle_ps(p, p, _MM_SHUFFLE(3,3,1,1));// { p[1],p[1],p[3],p[3] }
        __m128 i  = _mm_random_ps( _mm_add_ps(_mm_random_ps( ix ), iy) );
        __m128 gx = _mm_div_ps(i, _mm_set1_ps(41.0F));
               gx = _mm_sub_ps(gx,_mm_floor_ps(gx));
               gx = _mm_sub_ps(_mm_mul_ps(gx, _mm_set1_ps(2)), _mm_set1_ps(1));
		__m128 gy = _mm_sub_ps(_mm_and_ps(gx, __f32vec4_abs_mask_cheat.m), _mm_set1_ps(0.5F));
			   gx = _mm_sub_ps( gx, _mm_floor_ps(_mm_add_ps(gx, _mm_set1_ps(0.5F))) );

        __m128 norm   = _mm_rsqrt_ps(_mm_add_ps(_mm_mul_ps(gx,gx), _mm_mul_ps(gy,gy)));// { g00_norm, g10_norm, g01_norm, g11_norm }
               gx     = _mm_mul_ps(gx, norm);
               gy     = _mm_mul_ps(gy, norm);
        __m128 g00x10 = _mm_shuffle_ps(gx, gy, _MM_SHUFFLE(1,0, 1,0));// { {gx[0], gy[0]} X {gx[1], gy[1]} }
        __m128 g01x11 = _mm_shuffle_ps(gx, gy, _MM_SHUFFLE(3,2, 3,2));// { {gx[2], gy[2]} X {gx[3], gy[3]} }

		// 3. Compute gradient weights
        __m128 w00x10 = _mm_mul_ps(v00x10, g00x10);
        __m128 w01x11 = _mm_mul_ps(v01x11, g01x11);
        // {x0, x1, y0, y1}
        // {x2, x3, y2, y3}
        // op: shuffle(x0,x1, x2,x3) + shuffle(y0,y1, y2,y3)
        // {x0+y0,x1+y1,x2+y2,x3+y3 }
        __m128 w00_10_01_11 = _mm_add_ps(_mm_shuffle_ps(w00x10, w01x11, _MM_SHUFFLE(1,0, 1,0)), 
                                         _mm_shuffle_ps(w00x10, w01x11, _MM_SHUFFLE(3,2, 3,2)));

		// 4. Compute bilinear interpolation of weights
        __m128 fade_xy = _mm_fade_ps(f);
        __m128 lerp_y = _mm_lerp_ps(w00_10_01_11, _mm_shuffle_ps(w00_10_01_11, w00_10_01_11, _MM_SHUFFLE(0,0, 3,2)), _mm_shuffle_ps(fade_xy, fade_xy, _MM_SHUFFLE(0,0, 1,1)));
        __m128 lerp_x = _mm_lerp_ss(lerp_y, _mm_shuffle_ps(lerp_y, lerp_y,  _MM_SHUFFLE(0,0,0, 1)), _mm_shuffle_ps(fade_xy, fade_xy, _MM_SHUFFLE(0,0,0, 0)));
        return _mm_cvtss_f32(lerp_x) * 2.2F;
	}

    inline float _mm_cnoise3_ps(__m128 P) {
        // 1. Compute noise cell coordinates and offsets
        __m128 Pi0 = _mm_floor_ps(P);
        __m128 Pi1 = _mm_add_ps(Pi0, _mm_set1_ps(1.0F));
			   Pi0 = _mm_mod289_ps( Pi0 );
			   Pi1 = _mm_mod289_ps( Pi1 );
		__m128 ix  = _mm_set_ps(Pi1.m128_f32[0], Pi0.m128_f32[0], Pi1.m128_f32[0], Pi0.m128_f32[0]);// { Pi0[0], Pi1[0], Pi0[0], Pi1[0] }
        __m128 iy  = _mm_shuffle_ps( Pi0, Pi1, _MM_SHUFFLE(1,1,1,1) );// { Pi0[1], Pi0[1], Pi1[1], Pi1[1] }
        __m128 iz0 = _mm_shuffle_ps( Pi0, Pi0, _MM_SHUFFLE(2,2,2,2) );
        __m128 iz1 = _mm_shuffle_ps( Pi1, Pi1, _MM_SHUFFLE(2,2,2,2) );

		__m128 v000 = _mm_frac_ps( P );
        __m128 v111 = _mm_sub_ps(v000, _mm_set1_ps(1.0F));
        __m128 v100 = _mm_set_ps(0.0F, v000.m128_f32[2], v000.m128_f32[1], v111.m128_f32[0]);
        __m128 v010 = _mm_set_ps(0.0F, v000.m128_f32[2], v111.m128_f32[1], v000.m128_f32[0]);
        __m128 v110 = _mm_set_ps(0.0F, v000.m128_f32[2], v111.m128_f32[1], v111.m128_f32[0]);
        __m128 v001 = _mm_set_ps(0.0F, v111.m128_f32[2], v000.m128_f32[1], v000.m128_f32[0]);
        __m128 v101 = _mm_set_ps(0.0F, v111.m128_f32[2], v000.m128_f32[1], v111.m128_f32[0]);
        __m128 v011 = _mm_set_ps(0.0F, v111.m128_f32[2], v111.m128_f32[1], v000.m128_f32[0]);

		// 2. Compute gradient_vectors
        __m128 ixy  = _mm_random_ps( _mm_add_ps(_mm_random_ps( ix ), iy) );
        __m128 ixy0 = _mm_random_ps( _mm_add_ps(ixy, iz0) );
        __m128 ixy1 = _mm_random_ps( _mm_add_ps(ixy, iz1) );
        __m128 tmp;

        __m128 gx0 = _mm_div_ps(ixy0, _mm_set1_ps(7.0F));
        __m128 gy0 = _mm_sub_ps(_mm_frac_ps( _mm_div_ps(_mm_floor_ps(gx0), _mm_set1_ps(7.0F)) ), _mm_set1_ps(0.5F));
			   gx0 = _mm_frac_ps(gx0);
		__m128 gz0 = _mm_sub_ps( _mm_sub_ps(_mm_set1_ps(0.5F), _mm_abs_ps(gx0)), _mm_abs_ps(gy0) );
        __m128 sz0 = _mm_and_ps(_mm_cmple_ps(gz0, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
               tmp = _mm_and_ps(_mm_cmpge_ps(gx0, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
			   gx0 = _mm_sub_ps( gx0, _mm_mul_ps(sz0, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );
               tmp = _mm_and_ps(_mm_cmpge_ps(gy0, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
			   gy0 = _mm_sub_ps( gy0, _mm_mul_ps(sz0, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );

		__m128 gx1 = _mm_div_ps(ixy1, _mm_set1_ps(7.0F));
		__m128 gy1 = _mm_sub_ps(_mm_frac_ps( _mm_div_ps(_mm_floor_ps(gx1), _mm_set1_ps(7.0F)) ), _mm_set1_ps(0.5F));
               gx1 = _mm_frac_ps(gx1);
		__m128 gz1 = _mm_sub_ps( _mm_sub_ps(_mm_set1_ps(0.5F), _mm_abs_ps(gx1)), _mm_abs_ps(gy1) );
        __m128 sz1 = _mm_and_ps(_mm_cmple_ps(gz1, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
               tmp = _mm_and_ps(_mm_cmpge_ps(gx1, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
               gx1 = _mm_sub_ps( gx1, _mm_mul_ps(sz1, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );
               tmp = _mm_and_ps(_mm_cmpge_ps(gy1, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
			   gy1 = _mm_sub_ps( gy1, _mm_mul_ps(sz1, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );

        tmp = _mm_add_ps(_mm_add_ps(_mm_mul_ps(gx0, gx0), _mm_mul_ps(gy0, gy0)), _mm_mul_ps(gz0, gz0));
        tmp = _mm_rsqrt_ps(tmp);
        gx0 = _mm_mul_ps(gx0, tmp);
        gy0 = _mm_mul_ps(gy0, tmp);
        gz0 = _mm_mul_ps(gz0, tmp);
        tmp = _mm_add_ps(_mm_add_ps(_mm_mul_ps(gx1, gx1), _mm_mul_ps(gy1, gy1)), _mm_mul_ps(gz1, gz1));
        tmp = _mm_rsqrt_ps(tmp);
        gx1 = _mm_mul_ps(gx1, tmp);
        gy1 = _mm_mul_ps(gy1, tmp);
        gz1 = _mm_mul_ps(gz1, tmp);

        __m128 g000 = _mm_set_ps(0.0F, gz0.m128_f32[0], gy0.m128_f32[0], gx0.m128_f32[0]);
        __m128 g100 = _mm_set_ps(0.0F, gz0.m128_f32[1], gy0.m128_f32[1], gx0.m128_f32[1]);
        __m128 g010 = _mm_set_ps(0.0F, gz0.m128_f32[2], gy0.m128_f32[2], gx0.m128_f32[2]);
        __m128 g110 = _mm_set_ps(0.0F, gz0.m128_f32[3], gy0.m128_f32[3], gx0.m128_f32[3]);
        __m128 g001 = _mm_set_ps(0.0F, gz1.m128_f32[0], gy1.m128_f32[0], gx1.m128_f32[0]);
        __m128 g101 = _mm_set_ps(0.0F, gz1.m128_f32[1], gy1.m128_f32[1], gx1.m128_f32[1]);
        __m128 g011 = _mm_set_ps(0.0F, gz1.m128_f32[2], gy1.m128_f32[2], gx1.m128_f32[2]);
        __m128 g111 = _mm_set_ps(0.0F, gz1.m128_f32[3], gy1.m128_f32[3], gx1.m128_f32[3]);

		// 3. Compute gradient weights
        __m128 wz0 = _mm_setr_ps(_mm_cvtss_f32(_mm_dp_ps(v000, g000, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v100, g100, 0xff)), 
                                 _mm_cvtss_f32(_mm_dp_ps(v010, g010, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v110, g110, 0xff)));
        __m128 wz1 = _mm_setr_ps(_mm_cvtss_f32(_mm_dp_ps(v001, g001, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v101, g101, 0xff)), 
                                 _mm_cvtss_f32(_mm_dp_ps(v011, g011, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v111, g111, 0xff)));

		// 4. Compute trilinear interpolation of weights
		__m128 fade_xyz = _mm_fade_ps(v000);
		__m128 wz  = _mm_lerp_ps( wz0, wz1, _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(2,2,2,2)) );
		__m128 wy  = _mm_lerp_ps( wz, _mm_shuffle_ps(wz, wz, _MM_SHUFFLE(0,0, 3,2)), _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(0,0, 1,1)) );
		__m128 wx  = _mm_lerp_ss( wy, _mm_shuffle_ps(wy, wy, _MM_SHUFFLE(0,0,0, 1)), _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(0,0,0, 0)) );
		return _mm_cvtss_f32(wx) * 2.2F;
    }

    inline float _mm_cnoise3_ps(__m128 P, __m128 rep) {
        const auto permute = [](__m128 _X) { 
            return _mm_fmod_ps( _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_X,_X), _mm_set1_ps(34.0F)), _X), _mm_set1_ps(289.0F) ); 
        };

        // 1. Compute noise cell coordinates and offsets
        __m128 Pi0 = _mm_fmod_ps(_mm_floor_ps(P), rep);
        __m128 Pi1 = _mm_fmod_ps(_mm_add_ps(Pi0, _mm_set1_ps(1.0F)), rep);
			   Pi0 = _mm_fmod_ps( Pi0, _mm_set1_ps(289.0F) );
			   Pi1 = _mm_fmod_ps( Pi1, _mm_set1_ps(289.0F) );
		__m128 ix  = _mm_set_ps(Pi1.m128_f32[0], Pi0.m128_f32[0], Pi1.m128_f32[0], Pi0.m128_f32[0]);// { Pi0[0], Pi1[0], Pi0[0], Pi1[0] }
        __m128 iy  = _mm_shuffle_ps( Pi0, Pi1, _MM_SHUFFLE(1,1,1,1) );// { Pi0[1], Pi0[1], Pi1[1], Pi1[1] }
        __m128 iz0 = _mm_shuffle_ps( Pi0, Pi0, _MM_SHUFFLE(2,2,2,2) );
        __m128 iz1 = _mm_shuffle_ps( Pi1, Pi1, _MM_SHUFFLE(2,2,2,2) );

		__m128 v000 = _mm_frac_ps( P );
        __m128 v111 = _mm_sub_ps(v000, _mm_set1_ps(1.0F));
        __m128 v100 = _mm_set_ps(0.0F, v000.m128_f32[2], v000.m128_f32[1], v111.m128_f32[0]);
        __m128 v010 = _mm_set_ps(0.0F, v000.m128_f32[2], v111.m128_f32[1], v000.m128_f32[0]);
        __m128 v110 = _mm_set_ps(0.0F, v000.m128_f32[2], v111.m128_f32[1], v111.m128_f32[0]);
        __m128 v001 = _mm_set_ps(0.0F, v111.m128_f32[2], v000.m128_f32[1], v000.m128_f32[0]);
        __m128 v101 = _mm_set_ps(0.0F, v111.m128_f32[2], v000.m128_f32[1], v111.m128_f32[0]);
        __m128 v011 = _mm_set_ps(0.0F, v111.m128_f32[2], v111.m128_f32[1], v000.m128_f32[0]);

		// 2. Compute gradient_vectors
        __m128 ixy  = permute( _mm_add_ps(permute( ix ), iy) );
        __m128 ixy0 = permute( _mm_add_ps(ixy, iz0) );
        __m128 ixy1 = permute( _mm_add_ps(ixy, iz1) );
        __m128 tmp;

        __m128 gx0 = _mm_mul_ps(ixy0, _mm_set1_ps(1.0F/7.0F));
        __m128 gy0 = _mm_mul_ps(_mm_floor_ps(gx0), _mm_set1_ps(1.0F/7.0F));
               gy0 = _mm_sub_ps(_mm_frac_ps(gy0), _mm_set1_ps(0.5F));
			   gx0 = _mm_frac_ps(gx0);
		__m128 gz0 = _mm_sub_ps( _mm_sub_ps(_mm_set1_ps(0.5F), _mm_abs_ps(gx0)), _mm_abs_ps(gy0) );
        __m128 sz0 = _mm_and_ps(_mm_cmple_ps(gz0, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
               tmp = _mm_and_ps(_mm_cmpge_ps(gx0, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
			   gx0 = _mm_sub_ps( gx0, _mm_mul_ps(sz0, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );
               tmp = _mm_and_ps(_mm_cmpge_ps(gy0, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
			   gy0 = _mm_sub_ps( gy0, _mm_mul_ps(sz0, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );

		__m128 gx1 = _mm_mul_ps(ixy1, _mm_set1_ps(1.0F/7.0F));
		__m128 gy1 = _mm_mul_ps(_mm_floor_ps(gx1), _mm_set1_ps(1.0F/7.0F));
               gy1 = _mm_sub_ps(_mm_frac_ps(gy1), _mm_set1_ps(0.5F));
               gx1 = _mm_frac_ps(gx1);
		__m128 gz1 = _mm_sub_ps( _mm_sub_ps(_mm_set1_ps(0.5F), _mm_abs_ps(gx1)), _mm_abs_ps(gy1) );
        __m128 sz1 = _mm_and_ps(_mm_cmple_ps(gz1, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
               tmp = _mm_and_ps(_mm_cmpge_ps(gx1, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
               gx1 = _mm_sub_ps( gx1, _mm_mul_ps(sz1, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );
               tmp = _mm_and_ps(_mm_cmpge_ps(gy1, _mm_set1_ps(0.0F)), __f32vec4_eq_mask_cheat.m);
			   gy1 = _mm_sub_ps( gy1, _mm_mul_ps(sz1, _mm_sub_ps(tmp, _mm_set1_ps(0.5F))) );

        tmp = _mm_add_ps(_mm_add_ps(_mm_mul_ps(gx0, gx0), _mm_mul_ps(gy0, gy0)), _mm_mul_ps(gz0, gz0));
        tmp = _mm_rsqrt_ps(tmp);
        gx0 = _mm_mul_ps(gx0, tmp);
        gy0 = _mm_mul_ps(gy0, tmp);
        gz0 = _mm_mul_ps(gz0, tmp);
        tmp = _mm_add_ps(_mm_add_ps(_mm_mul_ps(gx1, gx1), _mm_mul_ps(gy1, gy1)), _mm_mul_ps(gz1, gz1));
        tmp = _mm_rsqrt_ps(tmp);
        gx1 = _mm_mul_ps(gx1, tmp);
        gy1 = _mm_mul_ps(gy1, tmp);
        gz1 = _mm_mul_ps(gz1, tmp);

        __m128 g000 = _mm_set_ps(0.0F, gz0.m128_f32[0], gy0.m128_f32[0], gx0.m128_f32[0]);
        __m128 g100 = _mm_set_ps(0.0F, gz0.m128_f32[1], gy0.m128_f32[1], gx0.m128_f32[1]);
        __m128 g010 = _mm_set_ps(0.0F, gz0.m128_f32[2], gy0.m128_f32[2], gx0.m128_f32[2]);
        __m128 g110 = _mm_set_ps(0.0F, gz0.m128_f32[3], gy0.m128_f32[3], gx0.m128_f32[3]);
        __m128 g001 = _mm_set_ps(0.0F, gz1.m128_f32[0], gy1.m128_f32[0], gx1.m128_f32[0]);
        __m128 g101 = _mm_set_ps(0.0F, gz1.m128_f32[1], gy1.m128_f32[1], gx1.m128_f32[1]);
        __m128 g011 = _mm_set_ps(0.0F, gz1.m128_f32[2], gy1.m128_f32[2], gx1.m128_f32[2]);
        __m128 g111 = _mm_set_ps(0.0F, gz1.m128_f32[3], gy1.m128_f32[3], gx1.m128_f32[3]);

		// 3. Compute gradient weights
        __m128 wz0 = _mm_setr_ps(_mm_cvtss_f32(_mm_dp_ps(v000, g000, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v100, g100, 0xff)), 
                                 _mm_cvtss_f32(_mm_dp_ps(v010, g010, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v110, g110, 0xff)));
        __m128 wz1 = _mm_setr_ps(_mm_cvtss_f32(_mm_dp_ps(v001, g001, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v101, g101, 0xff)), 
                                 _mm_cvtss_f32(_mm_dp_ps(v011, g011, 0xff)),
                                 _mm_cvtss_f32(_mm_dp_ps(v111, g111, 0xff)));

		// 4. Compute trilinear interpolation of weights
		__m128 fade_xyz = _mm_fade_ps(v000);
		__m128 wz  = _mm_lerp_ps( wz0, wz1, _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(2,2,2,2)) );
		__m128 wy  = _mm_lerp_ps( wz, _mm_shuffle_ps(wz, wz, _MM_SHUFFLE(0,0, 3,2)), _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(0,0, 1,1)) );
		__m128 wx  = _mm_lerp_ss( wy, _mm_shuffle_ps(wy, wy, _MM_SHUFFLE(0,0,0, 1)), _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(0,0,0, 0)) );
		return _mm_cvtss_f32(wx) * 2.2F;
    }

    inline float cnoise2(calculation::m128vector<2> x) {
        return _mm_cnoise2_ps(_mm_load_ps(x.data()));// safe
    }

    inline float cnoise3(calculation::m128vector<3> x) {
        return _mm_cnoise2_ps(_mm_load_ps(x.data()));// safe
    }

    inline float cnoise3(calculation::m128vector<3> x, calculation::m128vector<3> rep) {
        return _mm_cnoise3_ps(_mm_load_ps(x.data()), _mm_load_ps(rep.data()));
    }
}// namespace StefanGustavson

namespace frostbite {
    inline float _mm_vnoise3_ps(__m128 x) {
		const auto hash = [](__m128 n) {// frac(sin(n+1.951)*43758.5453)
			n = _mm_add_ps(n, _mm_set1_ps(1.951F));
			n = _mm_sin_ps(n);
			n = _mm_mul_ps(n, _mm_set1_ps(43758.5453F));
			return _mm_frac_ps(n);
		};

		// little
		__m128 p = _mm_floor_ps(x);
		__m128 f = _mm_frac_ps(x);

		// weight values
		__m128 n = _mm_dp_ps( p, _mm_setr_ps(1.0F, 57.0F, 113.0F, 0.0F), 0xff );
		__m128 wz0 = hash(_mm_add_ps(n, _mm_setr_ps(0.0F, 1.0F, 57.0F, 58.0F)));
		__m128 wz1 = hash(_mm_add_ps(n, _mm_setr_ps(113.0F, 114.0F, 170.0F, 171.0F)));
		
		// trilerp 
		__m128 fade_xyz = _mm_scurve_ps(f);
		__m128 wz  = _mm_lerp_ps( wz0, wz1, _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(2,2,2,2)) );
		__m128 wy  = _mm_lerp_ps( wz, _mm_shuffle_ps(wz, wz, _MM_SHUFFLE(0,0, 3,2)), _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(0,0, 1,1)) );
		__m128 wx  = _mm_lerp_ss( wy, _mm_shuffle_ps(wy, wy, _MM_SHUFFLE(0,0,0, 1)), _mm_shuffle_ps(fade_xyz, fade_xyz, _MM_SHUFFLE(0,0,0, 0)) );
		return _mm_cvtss_f32(wx);
	}

    inline float _mm_cells3_ps(__m128 p, float cellCount) {
		__m128 pCell = _mm_mul_ps(p, _mm_setr_ps(cellCount, cellCount, cellCount, 0.0F));
		__m128 d     = _mm_setr_ps(1.0e10, 0.0F, 0.0F, 0.0F);

		for (int xo = -1; xo <= 1; xo++) {
			for (int yo = -1; yo <= 1; yo++) {
				for (int zo = -1; zo <= 1; zo++) {
					__m128 tp = _mm_add_ps(_mm_floor_ps(pCell), _mm_setr_ps(float(xo), float(yo), float(zo), 0.0F));
					float rng = _mm_vnoise3_ps(_mm_fmod_ps(tp, _mm_set1_ps(cellCount / 1.0F)));

					tp = _mm_sub_ps(_mm_sub_ps(pCell, tp), _mm_setr_ps(rng, rng, rng, 0.0F));

					d = _mm_min_ss(d, _mm_dp_ps(tp, tp, 0xff));
				}
			}
		}

		d = _mm_min_ss(d, _mm_set1_ps(1.0F));
		d = _mm_max_ss(d, _mm_set1_ps(0.0F));
		return _mm_cvtss_f32(d);
	}

    inline float cells3(calculation::m128vector<3> x, float cellCount) {
        return _mm_cells3_ps(_mm_load_ps(x.data()), cellCount);// safe
    }
}

// { same of <cmath> }
inline __m128 ABS(__m128 _X) {
    return _mm_and_ps(_X, __f32vec4_abs_mask_cheat.m);
}
inline __m128 TRUNC(__m128 _X) {
    return _mm_trunc_ps(_X);
}
inline __m128 FLOOR(__m128 _X) {
    return _mm_floor_ps(_X);
}
inline __m128 CEIL(__m128 _X) {
    return _mm_ceil_ps(_X);
}
inline __m128 ROUND(__m128 _X) {
    return _mm_round_ps(_X, _MM_ROUND_MODE_NEAREST);
}
inline __m128 MOD(__m128 _X, __m128 _Y) {
    return _mm_fmod_ps(_X, _Y);
}
// numeric1
inline __m128 POWER(__m128 _X, __m128 _Y) {
    return _mm_pow_ps(_X, _Y);
}
inline __m128 SQRT(__m128 _X) {
    return _mm_sqrt_ps(_X);
}
inline __m128 CBRT(__m128 _X) {
    return _mm_cbrt_ps(_X);
}
inline __m128 RSQRT(__m128 _X) {
    return _mm_invsqrt_ps(_X);
}
inline __m128 RCBRT(__m128 _X) {
    return _mm_invcbrt_ps(_X);
}
inline __m128 EXP(__m128 _X) {
    return _mm_exp_ps(_X);
}
inline __m128 LN(__m128 _X) {
    return _mm_log_ps(_X);
}
inline __m128 EXP2(__m128 _X) {
    return _mm_exp2_ps(_X);
}
inline __m128 LOG2(__m128 _X) {
    return _mm_log2_ps(_X);
}
inline __m128 EXP10(__m128 _X) {
    return _mm_exp10_ps(_X);
}
inline __m128 LOG10(__m128 _X) {
    return _mm_log10_ps(_X);
}
// numeric2
inline __m128 LOG1P(__m128 _X) {
    return _mm_log1p_ps(_X);
}
inline __m128 EXPM1(__m128 _X) {
    return _mm_expm1_ps(_X);
}
inline __m128 ERF(__m128 _X) {
    return _mm_erf_ps(_X);
}
inline __m128 ERFC(__m128 _X) {
    return _mm_erfc_ps(_X);
}
// trigonometric
inline __m128 HYPOT(__m128 _X, __m128 _Y) {
    return _mm_hypot_ps(_X, _Y);
}
inline __m128 SIN(__m128 _X) {
    return _mm_sin_ps(_X);
}
inline __m128 COS(__m128 _X) {
    return _mm_cos_ps(_X);
}
inline __m128 TAN(__m128 _X) {
    return _mm_tan_ps(_X);
}
inline __m128 ASIN(__m128 _X) {
    return _mm_asin_ps(_X);
}
inline __m128 ACOS(__m128 _X) {
    return _mm_acos_ps(_X);
}
inline __m128 ATAN(__m128 _X) {
    return _mm_atan_ps(_X);
}
inline __m128 ATAN2(__m128 _Y, __m128 _X) {
    return _mm_atan2_ps(_Y, _X);
}
// hyperbolic
inline __m128 SINH(__m128 _X) {
    return _mm_sinh_ps(_X);
}
inline __m128 COSH(__m128 _X) {
    return _mm_cosh_ps(_X);
}
inline __m128 TANH(__m128 _X) {
    return _mm_tanh_ps(_X);
}
inline __m128 ASINH(__m128 _X) {
    return _mm_asinh_ps(_X);
}
inline __m128 ACOSH(__m128 _X) {
    return _mm_acosh_ps(_X);
}
inline __m128 ATANH(__m128 _X) {
    return _mm_atanh_ps(_X);
}

#endif

#ifdef _INCLUDED_EMM
const union {
    int i[4];
    __m128i m;
} __i32vec4_eq_mask_cheat = { 1, 1, 1, 1 };
const union {
    int i[4];
    __m128i m;
} __i32vec4_neq_mask_cheat = { 0, 0, 0, 0 };
// operator
inline __m128i operator-(__m128i _X) {
    return _mm_mul_epi32(_X, _mm_set1_epi32(-1));
}
inline __m128i operator+(__m128i _X, __m128i _Y) {
    return _mm_add_epi32(_X, _Y);
}
inline __m128i operator-(__m128i _X, __m128i _Y) {
    return _mm_sub_epi32(_X, _Y);
}
inline __m128i operator*(__m128i _X, __m128i _Y) {
    return _mm_mul_epi32(_X, _Y);
}
inline __m128i operator/(__m128i _X, __m128i _Y) {
    return _mm_div_epi32(_X, _Y);
}
inline __m128i operator%(__m128i _X, __m128i _Y) {
    return _mm_sub_epi32(_X, _mm_mul_epi32(_Y, _mm_div_epi32(_X, _Y)));
}
//comparator
inline __m128i operator==(__m128i _X, __m128i _Y) {
    return _mm_and_si128(_mm_cmpeq_epi32(_X, _Y), __i32vec4_eq_mask_cheat.m);
}
inline __m128i operator!=(__m128i _X, __m128i _Y) {
    return _mm_cmpeq_epi32(_X,_Y) == __i32vec4_neq_mask_cheat.m;
}
inline __m128i operator<(__m128i _X, __m128i _Y) {
    return _mm_and_si128(_mm_cmplt_epi32(_X, _Y), __i32vec4_eq_mask_cheat.m);
}
inline __m128i operator<=(__m128i _X, __m128i _Y) {
    return _mm_cmplt_epi32(_Y,_X) == __i32vec4_neq_mask_cheat.m;
}
inline __m128i operator>(__m128i _X, __m128i _Y) {
    return _mm_and_si128(_mm_cmpgt_epi32(_X, _Y), __i32vec4_eq_mask_cheat.m);
}
inline __m128i operator>=(__m128i _X, __m128i _Y) {
    return _mm_cmpgt_epi32(_Y,_X) == __i32vec4_neq_mask_cheat.m;
}
// numeric0
inline __m128i ABS(__m128i _X) {
    return _mm_abs_epi32(_X);
}
inline __m128i FLOOR(__m128i _X) { return _X; }
inline __m128i CEIL(__m128i _X)  { return _X; }
inline __m128i TRUNC(__m128i _X) { return _mm_set1_epi32(0); }
inline __m128i ROUND(__m128i _X) { return _X; }
inline __m128i MOD(__m128i _X, __m128i _Y) {
    return _mm_sub_epi32(_X, _mm_mul_epi32(_Y, _mm_div_epi32(_X, _Y)));
}
namespace calculation {
    // { specialization, calculation/lapack/vector.h, Section[1] }
    //_SPECIALIZATION_CLMAGIC_BLOCK_TRAITS_FOR_SIMD(int32_t, __m128i, 4, m128i_i32, _mm_set1_epi32)

    template<size_t _Size>
    using __m128i_vector = vector<int32_t, _Size, __m128i>;
}


const union {
    int i[4];
    __m128d m;
} __f64vec2_abs_mask_cheat = { -1, 0x7fffffff, -1, 0x7fffffff };
const union {
    double d[2];
    __m128d m;
} __f64vec2_eq_mask_cheat = { 1.0, 1.0 };
// operator
inline __m128d operator-(__m128d _X) {
    return _mm_mul_pd(_X, _mm_set1_pd(-1.0));
}
inline __m128d operator+(__m128d _X, __m128d _Y) {
    return _mm_add_pd(_X, _Y);
}
inline __m128d operator-(__m128d _X, __m128d _Y) {
    return _mm_sub_pd(_X, _Y);
}
inline __m128d operator*(__m128d _X, __m128d _Y) {
    return _mm_mul_pd(_X, _Y);
}
inline __m128d operator/(__m128d _X, __m128d _Y) {
    return _mm_div_pd(_X, _Y);
}
inline __m128d operator%(__m128d _X, __m128d _Y) {
    return _mm_fmod_pd(_X, _Y);
}
// comparator
inline __m128d operator==(__m128d _X, __m128d _Y) {
    return _mm_and_pd(_mm_cmpeq_pd(_X, _Y), __f64vec2_eq_mask_cheat.m);
}
inline __m128d operator!=(__m128d _X, __m128d _Y) {
    return _mm_and_pd(_mm_cmpneq_pd(_X, _Y), __f64vec2_eq_mask_cheat.m);
}
inline __m128d operator<(__m128d _X, __m128d _Y) {
    return _mm_and_pd(_mm_cmplt_pd(_X, _Y), __f64vec2_eq_mask_cheat.m);
}
inline __m128d operator<=(__m128d _X, __m128d _Y) {
    return _mm_and_pd(_mm_cmple_pd(_X, _Y), __f64vec2_eq_mask_cheat.m);
}
inline __m128d operator>(__m128d _X, __m128d _Y) {
    return _mm_and_pd(_mm_cmpgt_pd(_X, _Y), __f64vec2_eq_mask_cheat.m);
}
inline __m128d operator>=(__m128d _X, __m128d _Y) {
    return _mm_and_pd(_mm_cmpge_pd(_X, _Y), __f64vec2_eq_mask_cheat.m);
}
// numeric0
inline __m128d ABS(__m128d _X) {
    return _mm_and_pd(_X, __f64vec2_abs_mask_cheat.m);
}
inline __m128d TRUNC(__m128d _X) {
    return _mm_trunc_pd(_X);
}
inline __m128d CEIL(__m128d _X) {
    return _mm_ceil_pd(_X);
}
inline __m128d ROUND(__m128d _X) {
    return _mm_round_pd(_X, _MM_ROUND_MODE_NEAREST);
}
inline __m128d MOD(__m128d _X, __m128d _Y) {
    return _mm_fmod_pd(_X, _Y);
}
// numeric1
inline __m128d POWER(__m128d _X, __m128d _Y) {
    return _mm_pow_pd(_X, _Y);
}
inline __m128d SQRT(__m128d _X) {
    return _mm_sqrt_pd(_X);
}
inline __m128d CBRT(__m128d _X) {
    return _mm_cbrt_pd(_X);
}
inline __m128d RSQRT(__m128d _X) {
    return _mm_invsqrt_pd(_X);
}
inline __m128d RCBRT(__m128d _X) {
    return _mm_invcbrt_pd(_X);
}
inline __m128d EXP(__m128d _X) {
    return _mm_exp_pd(_X);
}
inline __m128d LN(__m128d _X) {
    return _mm_log_pd(_X);
}
inline __m128d EXP2(__m128d _X) {
    return _mm_exp2_pd(_X);
}
inline __m128d LOG2(__m128d _X) {
    return _mm_log2_pd(_X);
}
inline __m128d EXP10(__m128d _X) {
    return _mm_exp10_pd(_X);
}
inline __m128d LOG10(__m128d _X) {
    return _mm_log10_pd(_X);
}
// numeric2
inline __m128d LOG1P(__m128d _X) {
    return _mm_log1p_pd(_X);
}
inline __m128d EXPM1(__m128d _X) {
    return _mm_expm1_pd(_X);
}
inline __m128d ERF(__m128d _X) {
    return _mm_erf_pd(_X);
}
inline __m128d ERFC(__m128d _X) {
    return _mm_erfc_pd(_X);
}
// trigonometric
inline __m128d HYPOT(__m128d _X, __m128d _Y) {
    return _mm_hypot_pd(_X, _Y);
}
inline __m128d SIN(__m128d _X) {
    return _mm_sin_pd(_X);
}
inline __m128d COS(__m128d _X) {
    return _mm_cos_pd(_X);
}
inline __m128d TAN(__m128d _X) {
    return _mm_tan_pd(_X);
}
inline __m128d ASIN(__m128d _X) {
    return _mm_asin_pd(_X);
}
inline __m128d ACOS(__m128d _X) {
    return _mm_acos_pd(_X);
}
inline __m128d ATAN(__m128d _X) {
    return _mm_atan_pd(_X);
}
inline __m128d ATAN2(__m128d _Y, __m128d _X) {
    return _mm_atan2_pd(_Y, _X);
}
// hyperbolic
inline __m128d SINH(__m128d _X) {
    return _mm_sinh_pd(_X);
}
inline __m128d COSH(__m128d _X) {
    return _mm_cosh_pd(_X);
}
inline __m128d TANH(__m128d _X) {
    return _mm_tanh_pd(_X);
}
inline __m128d ASINH(__m128d _X) {
    return _mm_asinh_pd(_X);
}
inline __m128d ACOSH(__m128d _X) {
    return _mm_acosh_pd(_X);
}
inline __m128d ATANH(__m128d _X) {
    return _mm_atanh_pd(_X);
}
namespace calculation {
    // { specialization, calculation/lapack/vector.h, Section[1] }
    //_SPECIALIZATION_CLMAGIC_BLOCK_TRAITS_FOR_SIMD(double, __m128d, 2, m128d_f64, _mm_set1_pd)

    template<size_t _Size>
    using __m128d_vector = vector<double, _Size, __m128d>;
}
#endif

#ifdef _INCLUDED_IMM
const union {
    int i[8];
    __m256 m;
} __f32vec8_abs_mask_cheat = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
// operator
inline __m256 operator-(__m256 A) {
    return _mm256_mul_ps(A, _mm256_set1_ps(-1.0f));
}
inline __m256 operator+(__m256 A, __m256 B) {
    return _mm256_add_ps(A, B);
}
inline __m256 operator-(__m256 A, __m256 B) {
    return _mm256_sub_ps(A, B);
}
inline __m256 operator*(__m256 A, __m256 B) {
    return _mm256_mul_ps(A, B);
}
inline __m256 operator/(__m256 A, __m256 B) {
    return _mm256_div_ps(A, B);
}
inline __m256 operator%(__m256 A, __m256 B) {
    return _mm256_fmod_ps(A, B);
}
// numeric0
inline __m256 ABS(__m256 _X) { 
    return _mm256_and_ps(_X, __f32vec8_abs_mask_cheat.m); 
}
inline __m256 TRUNC(__m256 _X) { 
    return _mm256_trunc_ps(_X); 
}
inline __m256 FLOOR(__m256 _X) { 
    return _mm256_floor_ps(_X); 
}
inline __m256 CEIL(__m256 _X) { 
    return _mm256_ceil_ps(_X); 
}
inline __m256 ROUND(__m256 _X) { 
    return _mm256_round_ps(_X, _MM_ROUND_MODE_NEAREST); 
}
inline __m256 MOD(__m256 _X, __m256 _Y) { 
    return _mm256_fmod_ps(_X, _Y); 
}
// numeric1
inline __m256 POWER(__m256 _X, __m256 _Y) { 
    return _mm256_pow_ps(_X, _Y); 
}
inline __m256 SQRT(__m256 _X) { return _mm256_sqrt_ps(_X); }
inline __m256 CBRT(__m256 _X) { return _mm256_cbrt_ps(_X); }
inline __m256 RSQRT(__m256 _X) { return _mm256_invsqrt_ps(_X); }
inline __m256 RCBRT(__m256 _X) { return _mm256_invcbrt_ps(_X); }
inline __m256 EXP(__m256 _X) { return _mm256_exp_ps(_X); }
inline __m256 LN(__m256 _X) { return _mm256_log_ps(_X); }
inline __m256 EXP2(__m256 _X) { return _mm256_exp2_ps(_X); }
inline __m256 LOG2(__m256 _X) { return _mm256_log2_ps(_X); }
inline __m256 EXP10(__m256 _X) { return _mm256_exp10_ps(_X); }
inline __m256 LOG10(__m256 _X) { return _mm256_log10_ps(_X); }
// numeric2
inline __m256 LOG1P(__m256 _X) { return _mm256_log1p_ps(_X); }
inline __m256 EXPM1(__m256 _X) { return _mm256_expm1_ps(_X); }
inline __m256 ERF(__m256 _X) { return _mm256_erf_ps(_X); }
inline __m256 ERFC(__m256 _X) { return _mm256_erfc_ps(_X); }
// trigonometric
inline __m256 HYPOT(__m256 _X, __m256 _Y) { return _mm256_hypot_ps(_X, _Y); }
inline __m256 SIN(__m256 _X) { return _mm256_sin_ps(_X); }
inline __m256 COS(__m256 _X) { return _mm256_cos_ps(_X); }
inline __m256 TAN(__m256 _X) { return _mm256_tan_ps(_X); }
inline __m256 ASIN(__m256 _X) { return _mm256_asin_ps(_X); }
inline __m256 ACOS(__m256 _X) { return _mm256_acos_ps(_X); }
inline __m256 ATAN(__m256 _X) { return _mm256_atan_ps(_X); }
inline __m256 ATAN2(__m256 _Y, __m256 _X) { return _mm256_atan2_ps(_Y, _X); }
// hyperbolic
inline __m256 SINH(__m256 _X) { return _mm256_sinh_ps(_X); }
inline __m256 COSH(__m256 _X) { return _mm256_cosh_ps(_X); }
inline __m256 TANH(__m256 _X) { return _mm256_tanh_ps(_X); }
inline __m256 ASINH(__m256 _X) { return _mm256_asinh_ps(_X); }
inline __m256 ACOSH(__m256 _X) { return _mm256_acosh_ps(_X); }
inline __m256 ATANH(__m256 _X) { return _mm256_atanh_ps(_X); }


const union {
    int i[8];
    __m256d m;
} __f64vec4_abs_mask_cheat = { -1, 0x7fffffff, -1, 0x7fffffff, -1, 0x7fffffff, -1, 0x7fffffff };
// operator
inline __m256d operator-(__m256d A) {
    return _mm256_mul_pd(A, _mm256_set1_pd(-1.0));
}
inline __m256d operator+(__m256d A, __m256d B) {
    return _mm256_add_pd(A, B);
}
inline __m256d operator-(__m256d A, __m256d B) {
    return _mm256_sub_pd(A, B);
}
inline __m256d operator*(__m256d A, __m256d B) {
    return _mm256_mul_pd(A, B);
}
inline __m256d operator/(__m256d A, __m256d B) {
    return _mm256_div_pd(A, B);
}
inline __m256d operator%(__m256d A, __m256d B) {
    return _mm256_fmod_pd(A, B);
}
// numeric0
inline __m256d ABS(__m256d _X) { return _mm256_and_pd(_X, __f64vec4_abs_mask_cheat.m); }
inline __m256d FLOOR(__m256d _X) { return _mm256_floor_pd(_X); }
inline __m256d CEIL(__m256d _X) { return _mm256_ceil_pd(_X); }
inline __m256d TRUNC(__m256d _X) { return _mm256_trunc_pd(_X); }
inline __m256d ROUND(__m256d _X) { return _mm256_round_pd(_X, _MM_ROUND_MODE_NEAREST); }
inline __m256d MOD(__m256d _X, __m256d _Y) { return _mm256_fmod_pd(_X, _Y); }
// numeric1
inline __m256d POWER(__m256d _X, __m256d _Y) { return _mm256_pow_pd(_X, _Y); }
inline __m256d SQRT(__m256d _X) { return _mm256_sqrt_pd(_X); }
inline __m256d CBRT(__m256d _X) { return _mm256_cbrt_pd(_X); }
inline __m256d RSQRT(__m256d _X) { return _mm256_invsqrt_pd(_X); }
inline __m256d RCBRT(__m256d _X) { return _mm256_invcbrt_pd(_X); }
inline __m256d EXP(__m256d _X) { return _mm256_exp_pd(_X); }
inline __m256d LN(__m256d _X) { return _mm256_log_pd(_X); }
inline __m256d EXP2(__m256d _X) { return _mm256_exp2_pd(_X); }
inline __m256d LOG2(__m256d _X) { return _mm256_log2_pd(_X); }
inline __m256d EXP10(__m256d _X) { return _mm256_exp10_pd(_X); }
inline __m256d LOG10(__m256d _X) { return _mm256_log10_pd(_X); }
// numeric2
inline __m256d LOG1P(__m256d _X) { return _mm256_log1p_pd(_X); }
inline __m256d EXPM1(__m256d _X) { return _mm256_expm1_pd(_X); }
inline __m256d ERF(__m256d _X) { return _mm256_erf_pd(_X); }
inline __m256d ERFC(__m256d _X) { return _mm256_erfc_pd(_X); }
// trigonometric
inline __m256d HYPOT(__m256d _X, __m256d _Y) { return _mm256_hypot_pd(_X, _Y); }
inline __m256d SIN(__m256d _X) { return _mm256_sin_pd(_X); }
inline __m256d COS(__m256d _X) { return _mm256_cos_pd(_X); }
inline __m256d TAN(__m256d _X) { return _mm256_tan_pd(_X); }
inline __m256d ASIN(__m256d _X) { return _mm256_asin_pd(_X); }
inline __m256d ACOS(__m256d _X) { return _mm256_acos_pd(_X); }
inline __m256d ATAN(__m256d _X) { return _mm256_atan_pd(_X); }
inline __m256d ATAN2(__m256d _Y, __m256d _X) { return _mm256_atan2_pd(_Y, _X); }
// hyperbolic
inline __m256d SINH(__m256d _X) { return _mm256_sinh_pd(_X); }
inline __m256d COSH(__m256d _X) { return _mm256_cosh_pd(_X); }
inline __m256d TANH(__m256d _X) { return _mm256_tanh_pd(_X); }
inline __m256d ASINH(__m256d _X) { return _mm256_asinh_pd(_X); }
inline __m256d ACOSH(__m256d _X) { return _mm256_acosh_pd(_X); }
inline __m256d ATANH(__m256d _X) { return _mm256_atanh_pd(_X); }

inline __m256i operator-(__m256i A) {
    return _mm256_mul_epi32(A, _mm256_set1_epi32(-1));
}
inline __m256i operator+(__m256i A, __m256i B) {
    return _mm256_add_epi32(A, B);
}
inline __m256i operator-(__m256i A, __m256i B) {
    return _mm256_sub_epi32(A, B);
}
inline __m256i operator*(__m256i A, __m256i B) {
    return _mm256_mul_epi32(A, B);
}
inline __m256i operator/(__m256i A, __m256i B) {
    return _mm256_div_epi32(A, B);
}
inline __m256i operator%(__m256i _X, __m256i _Y) {
    return _mm256_sub_epi32(_X, _mm256_mul_epi32(_Y, _mm256_div_epi32(_X, _Y)));
}
inline __m256i ABS(__m256i _X) { return _mm256_abs_epi32(_X); }
inline __m256i FLOOR(__m256i _X) { return _X; }
inline __m256i CEIL(__m256i _X) { return _X; }
inline __m256i TRUNC(__m256i _X) { return _mm256_set1_epi32(0); }
inline __m256i ROUND(__m256i _X) { return _X; }
inline __m256i MOD(__m256i _X, __m256i _Y) { return _X % _Y; }
namespace calculation {
    
}
#endif

#ifdef _ZMMINTRIN_H_INCLUDED
inline __m512 operator-(__m512 A) {
    return _mm512_mul_ps(A, _mm512_set1_ps(-1.0f));
}
inline __m512 operator+(__m512 A, __m512 B) {
    return _mm512_add_ps(A, B);
}
inline __m512 operator-(__m512 A, __m512 B) {
    return _mm512_sub_ps(A, B);
}
inline __m512 operator*(__m512 A, __m512 B) {
    return _mm512_mul_ps(A, B);
}
inline __m512 operator/(__m512 A, __m512 B) {
    return _mm512_div_ps(A, B);
}
inline __m512 operator%(__m512 A, __m512 B) {
    return _mm512_fmod_ps(A, B);
}
// numeric0
inline __m512 ABS(__m512 _X) { return _mm512_abs_ps(_X); }
inline __m512 FLOOR(__m512 _X) { return _mm512_floor_ps(_X); }
inline __m512 CEIL(__m512 _X) { return _mm512_ceil_ps(_X); }
inline __m512 TRUNC(__m512 _X) { return _mm512_trunc_ps(_X); }
inline __m512 ROUND(__m512 _X) { return _mm512_roundscale_ps(_X, _MM_ROUND_MODE_NEAREST); }
inline __m512 MOD(__m512 _X, __m512 _Y) { return _mm512_fmod_ps(_X, _Y); }
// numeric1
inline __m512 POWER(__m512 _X, __m512 _Y) { return _mm512_pow_ps(_X, _Y); }
inline __m512 SQRT(__m512 _X) { return _mm512_sqrt_ps(_X); }
inline __m512 CBRT(__m512 _X) { return _mm512_cbrt_ps(_X); }
inline __m512 RSQRT(__m512 _X) { return _mm512_invsqrt_ps(_X); }
inline __m512 RCBRT(__m512 _X) { return _mm512_invcbrt_ps(_X); }
inline __m512 EXP(__m512 _X) { return _mm512_exp_ps(_X); }
inline __m512 LN(__m512 _X) { return _mm512_log_ps(_X); }
inline __m512 EXP2(__m512 _X) { return _mm512_exp2_ps(_X); }
inline __m512 LOG2(__m512 _X) { return _mm512_log2_ps(_X); }
inline __m512 EXP10(__m512 _X) { return _mm512_exp10_ps(_X); }
inline __m512 LOG10(__m512 _X) { return _mm512_log10_ps(_X); }
// numeric2
inline __m512 LOG1P(__m512 _X) { return _mm512_log1p_ps(_X); }
inline __m512 EXPM1(__m512 _X) { return _mm512_expm1_ps(_X); }
inline __m512 ERF(__m512 _X) { return _mm512_erf_ps(_X); }
inline __m512 ERFC(__m512 _X) { return _mm512_erfc_ps(_X); }
// trigonometric
inline __m512 HYPOT(__m512 _X, __m512 _Y) { return _mm512_hypot_ps(_X, _Y); }
inline __m512 SIN(__m512 _X) { return _mm512_sin_ps(_X); }
inline __m512 COS(__m512 _X) { return _mm512_cos_ps(_X); }
inline __m512 TAN(__m512 _X) { return _mm512_tan_ps(_X); }
inline __m512 ASIN(__m512 _X) { return _mm512_asin_ps(_X); }
inline __m512 ACOS(__m512 _X) { return _mm512_acos_ps(_X); }
inline __m512 ATAN(__m512 _X) { return _mm512_atan_ps(_X); }
inline __m512 ATAN2(__m512 _Y, __m512 _X) { return _mm512_atan2_ps(_Y, _X); }
// hyperbolic
inline __m512 SINH(__m512 _X) { return _mm512_sinh_ps(_X); }
inline __m512 COSH(__m512 _X) { return _mm512_cosh_ps(_X); }
inline __m512 TANH(__m512 _X) { return _mm512_tanh_ps(_X); }
inline __m512 ASINH(__m512 _X) { return _mm512_asinh_ps(_X); }
inline __m512 ACOSH(__m512 _X) { return _mm512_acosh_ps(_X); }
inline __m512 ATANH(__m512 _X) { return _mm512_atanh_ps(_X); }


inline __m512d operator-(__m512d A) {
    return _mm512_mul_pd(A, _mm512_set1_pd(-1.0));
}
inline __m512d operator+(__m512d A, __m512d B) {
    return _mm512_add_pd(A, B);
}
inline __m512d operator-(__m512d A, __m512d B) {
    return _mm512_sub_pd(A, B);
}
inline __m512d operator*(__m512d A, __m512d B) {
    return _mm512_mul_pd(A, B);
}
inline __m512d operator/(__m512d A, __m512d B) {
    return _mm512_div_pd(A, B);
}
inline __m512d operator%(__m512d A, __m512d B) {
    return _mm512_fmod_pd(A, B);
}
// numeric0
inline __m512d ABS(__m512d _X) { return _mm512_abs_pd(_X); }
inline __m512d TRUNC(__m512d _X) { return _mm512_trunc_pd(_X); }
inline __m512d FLOOR(__m512d _X) { return _mm512_floor_pd(_X); }
inline __m512d CEIL(__m512d _X) { return _mm512_ceil_pd(_X); }
inline __m512d ROUND(__m512d _X) { return _mm512_roundscale_pd(_X, _MM_ROUND_MODE_NEAREST); }
inline __m512d MOD(__m512d _X, __m512d _Y) { return _mm512_fmod_pd(_X, _Y); }
// numeric1
inline __m512d POWER(__m512d _X, __m512d _Y) { return _mm512_pow_pd(_X, _Y); }
inline __m512d SQRT(__m512d _X) { return _mm512_sqrt_pd(_X); }
inline __m512d CBRT(__m512d _X) { return _mm512_cbrt_pd(_X); }
inline __m512d RSQRT(__m512d _X) { return _mm512_invsqrt_pd(_X); }
inline __m512d RCBRT(__m512d _X) { return _mm512_invcbrt_pd(_X); }
// log
inline __m512d LN(__m512d _X) { return _mm512_log_pd(_X); }
inline __m512d LOG2(__m512d _X) { return _mm512_log2_pd(_X); }
inline __m512d LOG10(__m512d _X) { return _mm512_log10_pd(_X); }
// exp
inline __m512d EXP(__m512d _X) { return _mm512_exp_pd(_X); }
inline __m512d EXP2(__m512d _X) { return _mm512_exp2_pd(_X); }
inline __m512d EXP10(__m512d _X) { return _mm512_exp10_pd(_X); }
// numeric2
inline __m512d LOG1P(__m512d _X) { return _mm512_log1p_pd(_X); }
inline __m512d EXPM1(__m512d _X) { return _mm512_expm1_pd(_X); }
inline __m512d ERF(__m512d _X) { return _mm512_erf_pd(_X); }
inline __m512d ERFC(__m512d _X) { return _mm512_erfc_pd(_X); }
// trigonometric
inline __m512d HYPOT(__m512d _X, __m512d _Y) { return _mm512_hypot_pd(_X, _Y); }
inline __m512d SIN(__m512d _X) { return _mm512_sin_pd(_X); }
inline __m512d COS(__m512d _X) { return _mm512_cos_pd(_X); }
inline __m512d TAN(__m512d _X) { return _mm512_tan_pd(_X); }
inline __m512d ASIN(__m512d _X) { return _mm512_asin_pd(_X); }
inline __m512d ACOS(__m512d _X) { return _mm512_acos_pd(_X); }
inline __m512d ATAN(__m512d _X) { return _mm512_atan_pd(_X); }
inline __m512d ATAN2(__m512d _Y, __m512d _X) { return _mm512_atan2_pd(_Y, _X); }
// hyperbolic
inline __m512d SINH(__m512d _X) { return _mm512_sinh_pd(_X); }
inline __m512d COSH(__m512d _X) { return _mm512_cosh_pd(_X); }
inline __m512d TANH(__m512d _X) { return _mm512_tanh_pd(_X); }
inline __m512d ASINH(__m512d _X) { return _mm512_asinh_pd(_X); }
inline __m512d ACOSH(__m512d _X) { return _mm512_acosh_pd(_X); }
inline __m512d ATANH(__m512d _X) { return _mm512_atanh_pd(_X); }

inline __m512i operator-(__m512i A) {
    return _mm512_mul_epi32(A, _mm512_set1_epi32(-1));
}
inline __m512i operator+(__m512i A, __m512i B) {
    return _mm512_add_epi32(A, B);
}
inline __m512i operator-(__m512i A, __m512i B) {
    return _mm512_sub_epi32(A, B);
}
inline __m512i operator*(__m512i A, __m512i B) {
    return _mm512_mul_epi32(A, B);
}
inline __m512i operator/(__m512i A, __m512i B) {
    return _mm512_div_epi32(A, B);
}
inline __m512i operator%(__m512i A, __m512i B) {
    return _mm512_sub_epi32(A, _mm512_mul_epi32(B, _mm512_div_epi32(A, B)));
}
// numeric0
inline __m512i ABS(__m512i _X) { return _mm512_abs_epi32(_X); }
inline __m512i TRUNC(__m512i _X) { return _mm512_set1_epi32(0); }
inline __m512i FLOOR(__m512i _X) { return _X; }
inline __m512i CEIL(__m512i _X) { return _X; }
inline __m512i ROUND(__m512i _X) { return _X; }
inline __m512i MOD(__m512i _X, __m512i _Y) { return _X % _Y; }

#endif
