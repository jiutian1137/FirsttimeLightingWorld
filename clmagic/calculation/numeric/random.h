//+-------------------------------------------------------------------------------------
// Copyright (c) { Ken-Perlin, Stefan-Gustavson, Steven-Worley }
// All Rights Reserved
// -------------------------------------------------------------------------------------+
#pragma once
#include "real.h"
//#include "../lapack/vector.h"

namespace Perlin {
	struct gradient {
		static constexpr size_t  _NOISE_PERM_SIZE = 256;
		static constexpr uint8_t _NOISE_PERM[_NOISE_PERM_SIZE * 2] = {
			151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96,
			53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142,
			// Rest of noise permutation table
			8, 99, 37, 240, 21, 10, 23,
			190,  6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
			88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168,  68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
			77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
			102, 143, 54,  65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208,  89, 18, 169, 200, 196,
			135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64, 52, 217, 226, 250, 124, 123,
			5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
			223, 183, 170, 213, 119, 248, 152,  2, 44, 154, 163,  70, 221, 153, 101, 155, 167,  43, 172, 9,
			129, 22, 39, 253,  19, 98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246, 97, 228,
			251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,  81, 51, 145, 235, 249, 14, 239, 107,
			49, 192, 214,  31, 181, 199, 106, 157, 184,  84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254,
			138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
			151, 160, 137, 91, 90, 15,
			131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
			190,  6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
			88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168,  68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
			77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
			102, 143, 54,  65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208,  89, 18, 169, 200, 196,
			135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64, 52, 217, 226, 250, 124, 123,
			5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
			223, 183, 170, 213, 119, 248, 152,  2, 44, 154, 163,  70, 221, 153, 101, 155, 167,  43, 172, 9,
			129, 22, 39, 253,  19, 98, 108, 110, 79, 113, 224, 232, 178, 185,  112, 104, 218, 246, 97, 228,
			251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,  81, 51, 145, 235, 249, 14, 239, 107,
			49, 192, 214,  31, 181, 199, 106, 157, 184,  84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254,
			138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
		};

		size_t index(ptrdiff_t x) const {
			return x & (_NOISE_PERM_SIZE - 1);
		}

		template<typename _Ty>
		_Ty operator()(size_t ix, _Ty dx) const {
			uint8_t h = _NOISE_PERM[ix];
			h &= 3;
			return (h & 1 ? -dx : dx);
		}

		template<typename _Ty>
		_Ty operator()(size_t ix, size_t iy, _Ty dx, _Ty dy) const {
			uint8_t h = _NOISE_PERM[_NOISE_PERM[ix] + iy];
			h &= 3;
			return ((h & 1) ? -dx : dx) + ((h & 2) ? -dy : dy);
		}

		template<typename _Ty>
		_Ty operator()(size_t ix, size_t iy, size_t iz, _Ty dx, _Ty dy, _Ty dz) const {
			uint8_t h = _NOISE_PERM[ _NOISE_PERM[ _NOISE_PERM[ix] + iy ] + iz ];
			h &= 15;
			_Ty u = h < 8 || h == 12 || h == 13 ? dx : dy;
			_Ty v = h < 4 || h == 12 || h == 13 ? dy : dz;
			return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
		}
	};
}// namespace Ken Perlin

namespace StefanGustavson{
	struct gradient {
		size_t permute(size_t x) const {
			return (x * x * 34 + x) % 289;
		}

		size_t index(ptrdiff_t x) const {
			return x % 289;
		}

		template<typename _Ty>
		_Ty operator()(size_t ix, size_t iy, _Ty dx, _Ty dy) const {
			// get gradient vector
			size_t i  = permute(permute(ix) + iy);
			_Ty    gx = FRAC(i / 41.0F) * 2 - 1;
			_Ty    gy = ABS(gx) - 0.5F;
				   gx = gx - FLOOR(gx + 0.5F);
			// normalize gradient vector
			_Ty    gnorm = 1.0F / SQRT(gx * gx + gy * gy);
			gx *= gnorm;
			gy *= gnorm;
			// dot(gradient_vector, {dx,dy})
			return gx * dx + gy * dy;
		}
	};
}// namespace Stefan Gustavson

namespace calculation {

	/*<idea>
		<reference type="paper" author="Stefan Gustavson">
			http://www.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
			https://github.com/ashima/webgl-noise
		</reference>
	</idea>*/

	/*<idea>
		<reference type="book" author="...">
			{ Texturing and Modeling, chapter[Making Noises] }
		</reference>
	</idea>*/

	/*<method>
		<reference type="book" author="...">
			{ Physically Based Rendering From Theory to Implementation, chapter[10.6] }
		</reference>
	</method>*/

	template<typename _Ty, typename _Fn>
		requires requires(_Fn __f) { __f.index(0); __f(0, _Ty()); } 
	inline _Ty noise1(_Ty x, _Fn perm) {
		// 1. Compute noise cell coordinates and offsets
		const _Ty  xi = floor(x);
		const _Ty  dx = x - xi;
		// 2. Compute gradient weights
		const auto ix  = perm.index(static_cast<ptrdiff_t>(xi));
		const auto ix1 = perm.index(static_cast<ptrdiff_t>(xi + 1));
		const _Ty  w0  = perm(ix,  dx);
		const _Ty  w1  = perm(ix1, dx-1);
		// 3. Compute linear interpolation of weights
		return LERP(w0, w1, FADE(dx));
	}

	template<typename _Ty, typename _Fn> 
		requires requires(_Fn __f) { __f.index(0); __f(0,0,_Ty(),_Ty()); } 
	inline _Ty noise2(_Ty x, _Ty y, _Fn perm) {
		// 1. Compute noise cell coordinates and offsets
		const _Ty xi = floor(x);
		const _Ty yi = floor(y);
		const _Ty dx = x - xi;
		const _Ty dy = y - yi;
		// 2. Compute gradient weights
		const auto ix  = perm.index(static_cast<ptrdiff_t>(xi));
		const auto ix1 = perm.index(static_cast<ptrdiff_t>(xi + 1));
		const auto iy  = perm.index(static_cast<ptrdiff_t>(yi));
		const auto iy1 = perm.index(static_cast<ptrdiff_t>(yi + 1));
		const _Ty  w00 = perm(ix,  iy,  dx,   dy);
		const _Ty  w10 = perm(ix1, iy,  dx-1, dy);
		const _Ty  w01 = perm(ix,  iy1, dx,   dy - 1);
		const _Ty  w11 = perm(ix1, iy1, dx-1, dy - 1);
		// 3. Compute bilinear interpolation of weights
		return BILERP(w00, w10, w01, w11, FADE(dx), FADE(dy));
	}

	template<typename _Ty, typename _Fn>
		requires requires(_Fn __f) { __f.index(0); __f(0, 0, 0, _Ty(), _Ty(), _Ty()); } 
	inline _Ty noise3(_Ty x, _Ty y, _Ty z, _Fn perm) {
		// 1. Compute noise cell coordinates and offsets
		const _Ty xi = floor(x);
		const _Ty yi = floor(y);
		const _Ty zi = floor(z);
		const _Ty dx = x - xi;
		const _Ty dy = y - yi;
		const _Ty dz = z - zi;
		// 2. Compute gradient weights
		const auto ix   = perm.index(static_cast<ptrdiff_t>(xi));
		const auto ix1  = perm.index(static_cast<ptrdiff_t>(xi + 1));
		const auto iy   = perm.index(static_cast<ptrdiff_t>(yi));
		const auto iy1  = perm.index(static_cast<ptrdiff_t>(yi + 1));
		const auto iz   = perm.index(static_cast<ptrdiff_t>(zi));
		const auto iz1  = perm.index(static_cast<ptrdiff_t>(zi + 1));
		const _Ty  w000 = perm(ix,  iy,  iz,  dx,   dy,   dz);
		const _Ty  w100 = perm(ix1, iy,  iz,  dx-1, dy,   dz);
		const _Ty  w010 = perm(ix,  iy1, iz,  dx,   dy-1, dz);
		const _Ty  w110 = perm(ix1, iy1, iz,  dx-1, dy-1, dz);
		const _Ty  w001 = perm(ix,  iy,  iz1, dx,   dy,   dz-1);
		const _Ty  w101 = perm(ix1, iy,  iz1, dx-1, dy,   dz-1);
		const _Ty  w011 = perm(ix,  iy1, iz1, dx,   dy-1, dz-1);
		const _Ty  w111 = perm(ix1, iy1, iz1, dx-1, dy-1, dz-1);
		// 3. Compute trilinear interpolation of weights
		return TRILERP(w000, w100, w010, w110, w001, w101, w011, w111, FADE(dx), FADE(dy), FADE(dz));
	}

}// namespace clmagic
