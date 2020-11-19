#pragma once
#include <cmath>
#include "../calculation/lapack/vector.h"
#include "../calculation/numeric/real.h"

namespace Bruneton {


}// namespace Bruneton

namespace enviroment {
	// { Beer }
	template<typename T>
	T exponential_transmittance(T opticalDepth) {
		// behavior: { absorb, out_scatter }
		return exp(-opticalDepth);
	}

	// { Andrew Schneider }
	template<typename T>
	T powdersugar_transmittance(T opticalDepth) {
		// behavior: { out_scatter, in_scatter }
		return static_cast<T>(1) - exp(-(static_cast<T>(2) * opticalDepth));
	}


////////////////////////////////////////////////////////////////////////////////////////////////
//////////////// Bruneton //////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename T, typename B = T>
	struct AtmosphereParameters {
		using Length        = T;
		using Wavelength    = T;
		using InverseLength = T;
	
		using Angle         = T;
		using SolidAngle    = T;
		using InverseSolidAngle = T;
	
		using Number        = T;
		using Area          = T;
		using Volume        = T;
		
		using Power         = T;
		using LuminousPower = T;
		
		using Irradiance    = T;
		using Radiance      = T;
		using Luminance     = T;
		using Illuminance   = T;

		using SpectralPower      = T;
		using SpectralIrradiance = T;
		using SpectralRadiance   = T;
		using SpectralRadianceDensity = T;
		using ScatteringCoefficient   = T;
		using LuminousIntensity       = T;

		using AbstractSpectrum      = clmagic::vector3<T, B>;
		using DimensionlessSpectrum = clmagic::vector3<T, B>;
		using PowerSpectrum         = clmagic::vector3<T, B>;
		using IrradianceSpectrum    = clmagic::vector3<T, B>;
		using RadianceSpectrum      = clmagic::vector3<T, B>;
		using ScatteringSpectrum    = clmagic::vector3<T, B>;
		using Position  = clmagic::vector3<T, B>;
		using Direction = clmagic::vector3<T, B>;

	private:
		// units
		static const Number     PI  = static_cast<Number>(3.14159265358979323846);
		static const Length     m   = static_cast<Length>(1.0);
		static const Length     km  = Number(1000.0) * m;
		static const Wavelength nm  = static_cast<Wavelength>(1.0);

		static const Area       m2  = m * m;
		static const Volume     m3  = m * m * m;
		
		static const Angle      rad = static_cast<Angle>(1.0);
		static const Angle      pi  = PI * rad;
		static const Angle      deg = pi / Angle(180);
		
		static const SolidAngle sr  = static_cast<SolidAngle>(1.0);
		
		static const Power      watt = static_cast<Power>(1.0);
		static const Irradiance watt_per_square_meter = watt / m2;
		static const Radiance   watt_per_square_meter_per_sr = watt / (m2 * sr);
		static const SpectralIrradiance watt_per_square_meter_per_nm = watt / (m2 * nm);
		static const SpectralRadiance   watt_per_square_meter_per_sr_per_nm = watt / (m2 * sr * nm);
		static const SpectralRadianceDensity watt_per_cubic_meter_per_sr_per_nm = watt / (m3 * sr * nm);
		
		static const LuminousPower     lm  = static_cast<LuminousPower>(1.0);
		static const LuminousIntensity cd  = lm / sr;
		static const LuminousIntensity kcd = Number(1000.0) * cd;
		static const Luminance         cd_per_square_meter = cd / m2;
		static const Luminance kcd_per_square_meter = kcd / m2;

		static const DimensionlessSpectrum SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = DimensionlessSpectrum{ static_cast<T>(114974.916437), static_cast<T>(71305.954816), static_cast<T>(65310.548555) };
		static const DimensionlessSpectrum SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = DimensionlessSpectrum{ static_cast<T>(98242.786222), static_cast<T>(69954.398112), static_cast<T>(66475.012354) };
	
	public:
		struct DensityProfileLayer {
			Length width;
			Number exp_term;
			InverseLength exp_scale;
			InverseLength linear_term;
			Number constant_term;
		};

		struct DensityProfile {
			DensityProfileLayer layers[2];
		};

		IrradianceSpectrum solar_irradiance      = { 1.474000f * watt_per_square_meter, 1.850400f * watt_per_square_meter, 1.911980f * watt_per_square_meter };
		Angle              sun_angular_radius    = 0.004675f * rad;
		Length             bottom_radius         = 6360000.0f * m;
		Length             top_radius            = 6420000.0f * m;
		
		DensityProfile     rayleigh_density      = { DensityProfileLayer{ 0.000000f*m, 0.000000f,  0.000000f, 0.000000f, 0.000000f }, 
												     DensityProfileLayer{ 0.000000f*m, 1.000000f, -0.000125f, 0.000000f, 0.000000f } };
		ScatteringSpectrum rayleigh_scattering   = { 0.000006f, 0.000014f, 0.000033f };
		
		DensityProfile     mie_density           = { DensityProfileLayer{ 0.000000f*m, 0.000000f,  0.000000f, 0.000000f, 0.000000f }, 
												     DensityProfileLayer{ 0.000000f*m, 1.000000f, -0.000833f, 0.000000f, 0.000000f } };
		ScatteringSpectrum mie_scattering        = { 0.000004, 0.000004, 0.000004 };
		ScatteringSpectrum mie_extinction        = mie_scattering;
		Number             mie_phase_function_g  = 0.800000f;

		DensityProfile     absorption_density    = { DensityProfileLayer{ 25000.000000f*m, 0.000000f, 0.000000f,  0.000067f, -0.666667f }, 
													 DensityProfileLayer{ 0.000000f*m,     0.000000f, 0.000000f, -0.000067f, 2.666667f } };
		ScatteringSpectrum absorption_extinction = { 0.000001f, 0.000002f, 0.000000f };

		DimensionlessSpectrum ground_albedo      = { 0.100000f, 0.100000f, 0.100000f };
		Number             mu_s_min              = -0.500000f;
	};

	template<typename DensityProfileLayer, typename Length, typename Number = Length>
	Number _Get_layer_density(const DensityProfileLayer& layer, Length altitude) {
		Number density = layer.exp_term * exp(layer.exp_scale * altitude) + layer.linear_term * altitude + layer.constant_term;
		return saturate(density);
	}

	template<typename DensityProfile, typename Length, typename Number = Length>
	Number _Get_profile_density(const DensityProfile& profile, Length altitude) {
		return altitude < profile.layers[0].width ?
			_Get_layer_density(profile.layers[0], altitude) :
			_Get_layer_density(profile.layers[1], altitude);
	}
	 

	template<typename Number> inline
	Number clamp_cosine(Number mu) {
		return clamp(mu, static_cast<Number>(-1.0), static_cast<Number>(1.0));
	}

	template<typename Length> inline
	Length clamp_distance(Length d) {
		return max(d, static_cast<Length>(0.0));
	}

	template<typename Atmosphere, typename Length> inline
	Length clamp_radius(const Atmosphere& atmosphere, Length r) {
		return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
	}

	template<typename Area, typename Length = Area> inline
	Length safe_sqrt(Area a) {
		return sqrt( max(a, static_cast<Area>(0.0)) );
	}

	template<typename Atmosphere, typename Length, typename Number> inline //requires requires(Atmosphere __a) { __a.top_radius; }
	Length distance_to_atmosphere_top_boundary(const Atmosphere& atmosphere, Length r, Number mu) {
		assert( r <= atmosphere.top_radius );
		using Area = decltype(r * r);

		Area discriminant = r * r * (mu * mu - 1.0f) + atmosphere.top_radius * atmosphere.top_radius;
		return clamp_distance(-r * mu + safe_sqrt(discriminant));
	}

	template<typename Atmosphere, typename Length, typename Number> inline //requires requires(Atmosphere __a) { __a.bottom_radius; }
	Length distance_to_atmosphere_bottom_boundary(const Atmosphere& atmosphere, Length r, Number mu) {
		assert( r >= atmosphere.bottom_radius );
		using Area = decltype(r * r);

		Area discriminant = r * r * (mu * mu - 1.0f) + atmosphere.bottom_radius * atmosphere.bottom_radius;
		return clamp_distance(-r * mu - safe_sqrt(discriminant));
	}

	template<typename Atmosphere, typename Length, typename Number> inline
	bool ray_intersect_ground(const Atmosphere& atmosphere, Length r, Number mu) {
		assert( r >= atmosphere.bottom_radius );
		return (mu < 0.0f) && (r * r * (mu * mu - 1.0f) + atmosphere.bottom_radius*atmosphere.bottom_radius >= 0.0f);
	}

	template<typename Atmosphere, typename DensityProfile, typename Length, typename Number>
	Length integrate_optical_length_to_atmosphere_top_boundary(const Atmosphere& atmosphere, const DensityProfile& profile, Length r, Number mu, size_t SAMPLE_COUNT = 500) {
		assert( atmosphere.bottom_radius <= r  && r <= atmosphere.top_radius );
		
		// The integration step, i.e. the length of each integration interval.
		Length dx = distance_to_atmosphere_top_boundary(atmosphere, r, mu) / Number(SAMPLE_COUNT);
		// Integration loop.
		Length result = 0.0f * m;
		
		for (size_t i = 0; i <= SAMPLE_COUNT; ++i) {
			Length d_i = Number(i) * dx;
			// Distance between the current sample point and the planet center.
			Length r_i = sqrt(d_i * d_i + 2.0f * r * mu * d_i + r * r);
			// Number density at the current sample point (divided by the number density
			// at the bottom of the atmosphere, yielding a dimensionless number).
			Number y_i = _Get_profile_density(profile, r_i - atmosphere.bottom_radius);
			// Sample weight (from the trapezoidal rule).
			Number weight_i = (i == 0 || i == SAMPLE_COUNT ? 0.5f : 1.0f);
			
			result += y_i * weight_i * dx;
		}

		return result;
	}

	template<typename Atmosphere, typename Length, typename Number> inline
	Number integrate_transmittance_to_atmosphere_top_boundary(const Atmosphere& atmosphere, Length r, Number mu) {
		assert( atmosphere.bottom_radius <= r && r <= atmosphere.top_radius );
		return exponential_transmittance( atmosphere.rayleigh_scattering * 
			integrate_optical_length_to_atmosphere_top_boundary(atmosphere, atmosphere.rayleigh_density, r, mu) );
	}

}// namespace enviroment