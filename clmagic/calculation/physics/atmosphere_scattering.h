//+------------------------------------------------------------------------------------
// Copyright (c) xxxx { John-William-Strutt(Lord-Rayleigh)[1842-1919],
//						Lorentz-Mie,
//					    Louis-George-Henyey[1910-1970],
//						Jesse-Leonard-Greenstein[1909-2003],
//						Preetham,
//                      William M. Cornette,
//						Joseph G. Shanks }
// All Rights Reserved
// Free License
// https://mathshistory.st-andrews.ac.uk/Biographies/
// ------------------------------------------------------------------------------------+
#pragma once
#include "../numeric/real.h"
#include <array>

namespace Rayleigh {
	template<typename _RealTy> inline
	_RealTy phase_function(_RealTy mu) {
		/*
			 4
			--- * (1 + mu^2)
			 3
		*/
		return (4.0F/3.0F) * (1.0 + POWER(mu,2));
	}
	template<typename _RealTy> inline
	_RealTy normalized_phase_function(_RealTy mu) {
		_RealTy pi = static_cast<_RealTy>(3.141592653589793);
		return phase_function(mu) / (4 * pi);
	}
	template<typename _RealTy>
	_RealTy scattering_cross_section(_RealTy lambda, _RealTy N = 2.545e+25F, _RealTy n = 1.0003F) {
		// { 680e-9, 550e-9, 440e-9 }
		// Light-Scattering-Cross-Section for lambda Wave-Length with Number-Particles N and Refract-Index n
		/*
		     8 * pi^3 * (n^2 - 1)^2
			------------------------ * (6 + 3*delta)/(6 - 7*delta)
			  3 * lambda^4 * N^2
		*/
		_RealTy pi = static_cast<_RealTy>(3.141592653589793);
		_RealTy a  = 8 * POWER(pi, 3) * POWER(POWER(n,2) - 1, 2);
		_RealTy b  = 3 * POWER(lambda, 4) * POWER(N, /*2*/1);

		_RealTy delta = 0.035F;
		_RealTy correction_factor = (6 + 3*delta) / (6 - 7*delta);

		return a/b * correction_factor;
	}
	
	template<typename _RealTy> constexpr
	_RealTy particle_height_scale = static_cast<_RealTy>(8000.0);
}// namespace Rayleigh

namespace Mie {
	template<typename _RealTy> inline
	std::array<_RealTy, 3> scattering_cross_section(_RealTy _Aerosol_density_scale) {
		// _Aerosol_density_scale in [0,1]
		_RealTy betaMie = 2e-5F * _Aerosol_density_scale;
		return { betaMie, betaMie, betaMie };
	}

	template<typename _RealTy> constexpr
	_RealTy particle_height_scale = static_cast<_RealTy>(1200.0);
}// namespace Mie

namespace HenyeyGreenstein {
	template<typename _RealTy> inline
	_RealTy phase_function(_RealTy mu, _RealTy g) {
		/*
		          1 - g^2
			-----------------------
			 (1+g^2 - 2*g*mu)^(3/2)
		*/
		_RealTy gg = g * g;
		_RealTy a  = 1 - gg;
		_RealTy b  = POWER(1+gg - 2*g*mu, 3.0F/2.0F);
		return a / b;
	}
	template<typename _RealTy> inline
	_RealTy normalized_phase_function(_RealTy mu, _RealTy g) {
		_RealTy pi = static_cast<_RealTy>(3.141592653589793);
		return phase_function(mu, g) / (4 * pi);
	}
}// namespace Henyey and Greenstein

namespace CornetteShanks {
	template<typename _RealTy> inline
	_RealTy phase_function(_RealTy mu, _RealTy g) {
		/*
			 3    (1 - g^2) * (1 + mu^2)
			--- * ------------------------------------
			 2    (2 + g^2) * (1 + g^2 - 2*g*mu)^(3/2)
		*/
		_RealTy gg = g * g;
		_RealTy a = (1 - gg) * (1 + mu * mu);
		_RealTy b = (2 + gg) * POWER(1+gg - 2*g*mu, 3.0F/2.0F);
		return (3.0F / 2.0F) * a / b;
	}
	template<typename _RealTy> inline
	_RealTy normalized_phase_function(_RealTy mu, _RealTy g) {
		_RealTy pi = static_cast<_RealTy>(3.141592653589793);
		return phase_function(mu, g) / (4 * pi);
	}
}// namespace Cornette and Shanks

namespace Preetham {
	template<typename _RealTy> inline
	std::array<_RealTy, 3> scattering_cross_section(std::array<_RealTy, 3> lambda, _RealTy _Turbidity = 1.02F) {
		assert(_Turbidity >= 1.F);
		
		_RealTy K[] = {
			0.68455F,					   /* K[650nm] */
			0.678781F,					   /* K[570nm] */
			(0.668532F + 0.669765F) / 2.0F /* (K[470nm]+K[480nm])/2 */
		};

		// Beta is an Angstrom's turbidity coefficient and is approximated by:
		// float beta = 0.04608365822050f * m_fTurbidity - 0.04586025928522f; ???????
		_RealTy Pi      = static_cast<_RealTy>(3.141592653589793);
		_RealTy c       = (0.6544F * _Turbidity - 0.6510F) * 1e-16F; // concentration factor
		_RealTy v       = 4; // Junge's exponent
		_RealTy betaMie = 0.434F * c * Pi * pow(2*Pi, v-2);

		std::array<realmax_t, 3> results;
		for (size_t i = 0; i < 3; ++i) {
			results[i] = betaMie * K[i] / pow(lambda[i], v-2);
		}
		return results;
	}
}// namespace Preetham

template<typename _RealTy> inline
_RealTy RPHASE(_RealTy mu) {
	return Rayleigh::normalized_phase_function(mu);
}

template<typename _RealTy> inline
_RealTy HGPHASE(_RealTy mu, _RealTy g) {
	return HenyeyGreenstein::normalized_phase_function(mu, g);
}

template<typename _RealTy> inline
_RealTy CSPHASE(_RealTy mu, _RealTy g) {
	return CornetteShanks::normalized_phase_function(mu, g);
}

template<typename _RealTy> inline
_RealTy DLOBPHASE(_RealTy mu, _RealTy g0, _RealTy g1, _RealTy w) {
	return LERP(HGPHASE(mu, g0), HGPHASE(mu, g1), w);
}

namespace clmagic {

	template<typename T, typename B, typename _Fn>
	T integrate_opticaldepth(vector3<T,B> origin, unit_vector3<T,B> direction, T distance, _Fn Fdensity, size_t step = 128) {
		T result = T(0.0F);

		T ds = distance / static_cast<T>(step);
		vector3<T,B> dx = direction * ds;
		vector3<T,B> x  = origin;
		for (size_t i = 0U; i != step; ++i, x += dx) {
			result += Fdensity(x) * ds;
		}

		return result;
	}

}// namespace clmagic
