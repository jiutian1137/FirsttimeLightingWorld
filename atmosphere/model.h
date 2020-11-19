/**
 * Copyright (c) 2017 Eric Bruneton
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/*<h2>atmosphere/model.h</h2>

<p>This file defines the API to use our atmosphere model in OpenGL applications.
To use it:
<ul>
<li>create a <code>Model</code> instance with the desired atmosphere
parameters.</li>
<li>call <code>Init</code> to precompute the atmosphere textures,</li>
<li>link <code>GetShader</code> with your shaders that need access to the
atmosphere shading functions.</li>
<li>for each GLSL program linked with <code>GetShader</code>, call
<code>SetProgramUniforms</code> to bind the precomputed textures to this
program (usually at each frame).</li>
<li>delete your <code>Model</code> when you no longer need its shader and
precomputed textures (the destructor deletes these resources).</li>
</ul>

<p>The shader returned by <code>GetShader</code> provides the following
functions (that you need to forward declare in your own shaders to be able to
compile them separately):

<pre class="prettyprint">
// Returns the radiance of the Sun, outside the atmosphere.
vec3 GetSolarRadiance();

// Returns the sky radiance along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'view_ray', as well as the transmittance
// along this segment.
vec3 GetSkyRadiance(vec3 camera, vec3 view_ray, double shadow_length,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sky radiance along the segment from 'camera' to 'p', as well as
// the transmittance along this segment.
vec3 GetSkyRadianceToPoint(vec3 camera, vec3 p, double shadow_length,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sun and sky irradiance received on a surface patch located at 'p'
// and whose normal vector is 'normal'.
vec3 GetSunAndSkyIrradiance(vec3 p, vec3 normal, vec3 sun_direction,
    out vec3 sky_irradiance);

// Returns the luminance of the Sun, outside the atmosphere.
vec3 GetSolarLuminance();

// Returns the sky luminance along the segment from 'camera' to the nearest
// atmosphere boundary in direction 'view_ray', as well as the transmittance
// along this segment.
vec3 GetSkyLuminance(vec3 camera, vec3 view_ray, double shadow_length,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sky luminance along the segment from 'camera' to 'p', as well as
// the transmittance along this segment.
vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 p, double shadow_length,
    vec3 sun_direction, out vec3 transmittance);

// Returns the sun and sky illuminance received on a surface patch located at
// 'p' and whose normal vector is 'normal'.
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 normal, vec3 sun_direction,
    out vec3 sky_illuminance);
</pre>

<p>where
<ul>
<li><code>camera</code> and <code>p</code> must be expressed in a reference
frame where the planet center is at the origin, and measured in the unit passed
to the constructor's <code>length_unit_in_meters</code> argument.
<code>camera</code> can be in space, but <code>p</code> must be inside the
atmosphere,</li>
<li><code>view_ray</code>, <code>sun_direction</code> and <code>normal</code>
are unit direction vectors expressed in the same reference frame (with
<code>sun_direction</code> pointing <i>towards</i> the Sun),</li>
<li><code>shadow_length</code> is the length along the segment which is in
shadow, measured in the unit passed to the constructor's
<code>length_unit_in_meters</code> argument.</li>
</ul>

<p>and where
<ul>
<li>the first 4 functions return spectral radiance and irradiance values
(in $W.m^{-2}.sr^{-1}.nm^{-1}$ and $W.m^{-2}.nm^{-1}$), at the 3 wavelengths
<code>kLambdaR</code>, <code>kLambdaG</code>, <code>kLambdaB</code> (in this
order),</li>
<li>the other functions return luminance and illuminance values (in
$cd.m^{-2}$ and $lx$) in linear <a href="https://en.wikipedia.org/wiki/SRGB">
sRGB</a> space (i.e. before adjustements for gamma correction),</li>
<li>all the functions return the (unitless) transmittance of the atmosphere
along the specified segment at the 3 wavelengths <code>kLambdaR</code>,
<code>kLambdaG</code>, <code>kLambdaB</code> (in this order).</li>
</ul>

<p><b>Note</b> The precomputed atmosphere textures can store either irradiance
or illuminance values (see the <code>num_precomputed_wavelengths</code>
parameter):
<ul>
  <li>when using irradiance values, the RGB channels of these textures contain
  spectral irradiance values, in $W.m^{-2}.nm^{-1}$, at the 3 wavelengths
  <code>kLambdaR</code>, <code>kLambdaG</code>, <code>kLambdaB</code> (in this
  order). The API functions returning radiance values return these precomputed
  values (times the phase functions), while the API functions returning
  luminance values use the approximation described in
  <a href="https://arxiv.org/pdf/1612.04336.pdf">A Qualitative and Quantitative
  Evaluation of 8 Clear Sky Models</a>, section 14.3, to convert 3 radiance
  values to linear sRGB luminance values.</li>
  <li>when using illuminance values, the RGB channels of these textures contain
  illuminance values, in $lx$, in linear sRGB space. These illuminance values
  are precomputed as described in
  <a href="http://www.oskee.wz.cz/stranka/uploads/SCCG10ElekKmoch.pdf">Real-time
  Spectral Scattering in Large-scale Natural Participating Media</a>, section
  4.4 (i.e. <code>num_precomputed_wavelengths</code> irradiance values are
  precomputed, and then converted to sRGB via a numerical integration of this
  spectrum with the CIE color matching functions). The API functions returning
  luminance values return these precomputed values (times the phase functions),
  while <i>the API functions returning radiance values are not provided</i>.
  </li>
</ul>

<p>The concrete API definition is the following:
*/

#ifndef ATMOSPHERE_MODEL_H_
#define ATMOSPHERE_MODEL_H_

#include <cassert>
#include <array>
#include <vector>
#include <string>
#include <functional>
#include "../clmagic/openglutil/GL/glew.h"
#include "../clmagic/openglutil/glshader.h"

namespace physics {
    template<typename T>
    inline T InterpolateSpectrum(const std::vector<T>& wavelengths, const std::vector<T>& wavelength_function, T wavelength) {
        assert(wavelength_function.size() == wavelengths.size());
        if (wavelength < wavelengths[0]) {
            return wavelength_function[0];
        }

        for (size_t i = 0; i < wavelengths.size() - 1; ++i) {
            if (wavelength < wavelengths[i + 1]) {
                T u = (wavelength - wavelengths[i]) / (wavelengths[i + 1] - wavelengths[i]);
                return wavelength_function[i] * (static_cast<T>(1.0) - u) + wavelength_function[i + 1] * u;
            }
        }

        return wavelength_function[wavelength_function.size() - 1];
    }

    template<typename T>
    constexpr T visiblelight_min = static_cast<T>(360);//[nm]

    template<typename T>
    constexpr T visiblelight_max = static_cast<T>(830);//[nm]

    template<typename T>
    constexpr T visiblelight_red = static_cast<T>(680.0);//[nm]

    template<typename T>
    constexpr T visiblelight_green = static_cast<T>(550.0);//[nm]

    template<typename T>
    constexpr T visiblelight_blue = static_cast<T>(440.0);//[nm]

    constexpr double MAX_LUMINOUS_EFFICACY = 683.0;// The conversion factor between watts and lumens.
   
    constexpr double CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[380] = {
        // Values from "CIE (1931) 2-deg color matching functions", see
        // "http://web.archive.org/web/20081228084047/
        //  http://www.cvrl.org/database/data/cmfs/ciexyz31.txt".
        360, 0.000129900000, 0.000003917000, 0.000606100000,
        365, 0.000232100000, 0.000006965000, 0.001086000000,
        370, 0.000414900000, 0.000012390000, 0.001946000000,
        375, 0.000741600000, 0.000022020000, 0.003486000000,
        380, 0.001368000000, 0.000039000000, 0.006450001000,
        385, 0.002236000000, 0.000064000000, 0.010549990000,
        390, 0.004243000000, 0.000120000000, 0.020050010000,
        395, 0.007650000000, 0.000217000000, 0.036210000000,
        400, 0.014310000000, 0.000396000000, 0.067850010000,
        405, 0.023190000000, 0.000640000000, 0.110200000000,
        410, 0.043510000000, 0.001210000000, 0.207400000000,
        415, 0.077630000000, 0.002180000000, 0.371300000000,
        420, 0.134380000000, 0.004000000000, 0.645600000000,
        425, 0.214770000000, 0.007300000000, 1.039050100000,
        430, 0.283900000000, 0.011600000000, 1.385600000000,
        435, 0.328500000000, 0.016840000000, 1.622960000000,
        440, 0.348280000000, 0.023000000000, 1.747060000000,
        445, 0.348060000000, 0.029800000000, 1.782600000000,
        450, 0.336200000000, 0.038000000000, 1.772110000000,
        455, 0.318700000000, 0.048000000000, 1.744100000000,
        460, 0.290800000000, 0.060000000000, 1.669200000000,
        465, 0.251100000000, 0.073900000000, 1.528100000000,
        470, 0.195360000000, 0.090980000000, 1.287640000000,
        475, 0.142100000000, 0.112600000000, 1.041900000000,
        480, 0.095640000000, 0.139020000000, 0.812950100000,
        485, 0.057950010000, 0.169300000000, 0.616200000000,
        490, 0.032010000000, 0.208020000000, 0.465180000000,
        495, 0.014700000000, 0.258600000000, 0.353300000000,
        500, 0.004900000000, 0.323000000000, 0.272000000000,
        505, 0.002400000000, 0.407300000000, 0.212300000000,
        510, 0.009300000000, 0.503000000000, 0.158200000000,
        515, 0.029100000000, 0.608200000000, 0.111700000000,
        520, 0.063270000000, 0.710000000000, 0.078249990000,
        525, 0.109600000000, 0.793200000000, 0.057250010000,
        530, 0.165500000000, 0.862000000000, 0.042160000000,
        535, 0.225749900000, 0.914850100000, 0.029840000000,
        540, 0.290400000000, 0.954000000000, 0.020300000000,
        545, 0.359700000000, 0.980300000000, 0.013400000000,
        550, 0.433449900000, 0.994950100000, 0.008749999000,
        555, 0.512050100000, 1.000000000000, 0.005749999000,
        560, 0.594500000000, 0.995000000000, 0.003900000000,
        565, 0.678400000000, 0.978600000000, 0.002749999000,
        570, 0.762100000000, 0.952000000000, 0.002100000000,
        575, 0.842500000000, 0.915400000000, 0.001800000000,
        580, 0.916300000000, 0.870000000000, 0.001650001000,
        585, 0.978600000000, 0.816300000000, 0.001400000000,
        590, 1.026300000000, 0.757000000000, 0.001100000000,
        595, 1.056700000000, 0.694900000000, 0.001000000000,
        600, 1.062200000000, 0.631000000000, 0.000800000000,
        605, 1.045600000000, 0.566800000000, 0.000600000000,
        610, 1.002600000000, 0.503000000000, 0.000340000000,
        615, 0.938400000000, 0.441200000000, 0.000240000000,
        620, 0.854449900000, 0.381000000000, 0.000190000000,
        625, 0.751400000000, 0.321000000000, 0.000100000000,
        630, 0.642400000000, 0.265000000000, 0.000049999990,
        635, 0.541900000000, 0.217000000000, 0.000030000000,
        640, 0.447900000000, 0.175000000000, 0.000020000000,
        645, 0.360800000000, 0.138200000000, 0.000010000000,
        650, 0.283500000000, 0.107000000000, 0.000000000000,
        655, 0.218700000000, 0.081600000000, 0.000000000000,
        660, 0.164900000000, 0.061000000000, 0.000000000000,
        665, 0.121200000000, 0.044580000000, 0.000000000000,
        670, 0.087400000000, 0.032000000000, 0.000000000000,
        675, 0.063600000000, 0.023200000000, 0.000000000000,
        680, 0.046770000000, 0.017000000000, 0.000000000000,
        685, 0.032900000000, 0.011920000000, 0.000000000000,
        690, 0.022700000000, 0.008210000000, 0.000000000000,
        695, 0.015840000000, 0.005723000000, 0.000000000000,
        700, 0.011359160000, 0.004102000000, 0.000000000000,
        705, 0.008110916000, 0.002929000000, 0.000000000000,
        710, 0.005790346000, 0.002091000000, 0.000000000000,
        715, 0.004109457000, 0.001484000000, 0.000000000000,
        720, 0.002899327000, 0.001047000000, 0.000000000000,
        725, 0.002049190000, 0.000740000000, 0.000000000000,
        730, 0.001439971000, 0.000520000000, 0.000000000000,
        735, 0.000999949300, 0.000361100000, 0.000000000000,
        740, 0.000690078600, 0.000249200000, 0.000000000000,
        745, 0.000476021300, 0.000171900000, 0.000000000000,
        750, 0.000332301100, 0.000120000000, 0.000000000000,
        755, 0.000234826100, 0.000084800000, 0.000000000000,
        760, 0.000166150500, 0.000060000000, 0.000000000000,
        765, 0.000117413000, 0.000042400000, 0.000000000000,
        770, 0.000083075270, 0.000030000000, 0.000000000000,
        775, 0.000058706520, 0.000021200000, 0.000000000000,
        780, 0.000041509940, 0.000014990000, 0.000000000000,
        785, 0.000029353260, 0.000010600000, 0.000000000000,
        790, 0.000020673830, 0.000007465700, 0.000000000000,
        795, 0.000014559770, 0.000005257800, 0.000000000000,
        800, 0.000010253980, 0.000003702900, 0.000000000000,
        805, 0.000007221456, 0.000002607800, 0.000000000000,
        810, 0.000005085868, 0.000001836600, 0.000000000000,
        815, 0.000003581652, 0.000001293400, 0.000000000000,
        820, 0.000002522525, 0.000000910930, 0.000000000000,
        825, 0.000001776509, 0.000000641530, 0.000000000000,
        830, 0.000001251141, 0.000000451810, 0.000000000000,
    };
    
    constexpr double XYZ_TO_SRGB[9] = {
        // The conversion matrix from XYZ to linear sRGB color spaces.
        // Values from https://en.wikipedia.org/wiki/SRGB.
        +3.2406, -1.5372, -0.4986,
        -0.9689, +1.8758, +0.0415,
        +0.0557, -0.2040, +1.0570
    };


    template<typename T>
    T CieColorMatchingFunctionTableValue(T wavelength, int column) {
        if (wavelength <= visiblelight_min<T> || visiblelight_max<T> <= wavelength) {
            return static_cast<T>(0.0);
        }

        T u = (wavelength - visiblelight_min<T>) / static_cast<T>(5.0);

        int row = static_cast<int>(floor(u));
        assert(row >= 0 && row + 1 < 95);
        assert(CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row] <= wavelength &&
               CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1)] >= wavelength);

        u -= row;
        return static_cast<T>(CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * row + column] * (1.0 - u) + 
            CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[4 * (row + 1) + column] * u);
    }

    template<typename T>
    std::array<T, 3> ComputeSpectralRadianceToLuminanceFactors(const std::vector<T>& wavelengths, const std::vector<T>& solar_irradiance, T lambda_power) {
        // The returned constants are in lumen.nm / watt.
        T k_r = static_cast<T>(0.0);
        T k_g = static_cast<T>(0.0);
        T k_b = static_cast<T>(0.0);
        T solar_r = InterpolateSpectrum(wavelengths, solar_irradiance, visiblelight_red<T>);
        T solar_g = InterpolateSpectrum(wavelengths, solar_irradiance, visiblelight_green<T>);
        T solar_b = InterpolateSpectrum(wavelengths, solar_irradiance, visiblelight_blue<T>);
       
        const T dlambda = 1;

        for (T lambda = visiblelight_min<T>; lambda < visiblelight_max<T>; lambda += dlambda) {
            T x_bar = CieColorMatchingFunctionTableValue(lambda, 1);
            T y_bar = CieColorMatchingFunctionTableValue(lambda, 2);
            T z_bar = CieColorMatchingFunctionTableValue(lambda, 3);
            const double* xyz2srgb = XYZ_TO_SRGB;
            double r_bar =
                xyz2srgb[0] * x_bar + xyz2srgb[1] * y_bar + xyz2srgb[2] * z_bar;
            double g_bar =
                xyz2srgb[3] * x_bar + xyz2srgb[4] * y_bar + xyz2srgb[5] * z_bar;
            double b_bar =
                xyz2srgb[6] * x_bar + xyz2srgb[7] * y_bar + xyz2srgb[8] * z_bar;
            double irradiance = InterpolateSpectrum(wavelengths, solar_irradiance, lambda);
            k_r += r_bar * irradiance / solar_r *
                pow(lambda / visiblelight_red<T>, lambda_power);
            k_g += g_bar * irradiance / solar_g *
                pow(lambda / visiblelight_green<T>, lambda_power);
            k_b += b_bar * irradiance / solar_b *
                pow(lambda / visiblelight_blue<T>, lambda_power);
        }

        k_r *= MAX_LUMINOUS_EFFICACY * dlambda;
        k_g *= MAX_LUMINOUS_EFFICACY * dlambda;
        k_b *= MAX_LUMINOUS_EFFICACY * dlambda;

        return { k_r, k_g, k_b };
    }

    template<typename T>
    std::array<T, 3> ConvertSpectrumToLinearSrgb(const std::vector<T>& wavelengths, const std::vector<T>& spectrum) {
        T x = static_cast<T>(0.0);
        T y = static_cast<T>(0.0);
        T z = static_cast<T>(0.0);
        const T dlambda = 1;

        for (T lambda = visiblelight_min<T>; lambda < visiblelight_max<T>; lambda += dlambda) {
            T value = InterpolateSpectrum(wavelengths, spectrum, lambda);
            x += CieColorMatchingFunctionTableValue(lambda, 1) * value;
            y += CieColorMatchingFunctionTableValue(lambda, 2) * value;
            z += CieColorMatchingFunctionTableValue(lambda, 3) * value;
        }

        T r = MAX_LUMINOUS_EFFICACY *
            (XYZ_TO_SRGB[0] * x + XYZ_TO_SRGB[1] * y + XYZ_TO_SRGB[2] * z) * dlambda;
        T g = MAX_LUMINOUS_EFFICACY *
            (XYZ_TO_SRGB[3] * x + XYZ_TO_SRGB[4] * y + XYZ_TO_SRGB[5] * z) * dlambda;
        T b = MAX_LUMINOUS_EFFICACY *
            (XYZ_TO_SRGB[6] * x + XYZ_TO_SRGB[7] * y + XYZ_TO_SRGB[8] * z) * dlambda;
        return { r, g, b };
    }
}// namespace physics


namespace atmosphere {
    struct DensityProfileLayer {
        double width;
        double exp_term;
        double exp_scale;
        double linear_term;
        double constant_term;
    };

    struct AtmosphereParameters {
        // The wavelength values, in nanometers, and sorted in increasing order, for
        // which the solar_irradiance, rayleigh_scattering, mie_scattering,
        // mie_extinction and ground_albedo samples are provided. If your shaders
        // use luminance values (as opposed to radiance values, see above), use a
        // large number of wavelengths (e.g. between 15 and 50) to get accurate
        // results (this number of wavelengths has absolutely no impact on the
        // shader performance).
        std::vector<double> wavelengths;
        // The solar irradiance at the top of the atmosphere, in W/m^2/nm. This
        // vector must have the same size as the wavelengths parameter.
        std::vector<double> solar_irradiance;
        // The sun's angular radius, in radians. Warning: the implementation uses
        // approximations that are valid only if this value is smaller than 0.1.
        double sun_angular_radius;
        // The distance between the planet center and the bottom of the atmosphere,
        // in m.
        double bottom_radius;
        // The distance between the planet center and the top of the atmosphere,
        // in m.
        double top_radius;
        // The density profile of air molecules, i.e. a function from altitude to
        // dimensionless values between 0 (null density) and 1 (maximum density).
        // Layers must be sorted from bottom to top. The width of the last layer is
        // ignored, i.e. it always extend to the top atmosphere boundary. At most 2
        // layers can be specified.
        std::vector<DensityProfileLayer> rayleigh_density;
        // The scattering coefficient of air molecules at the altitude where their
        // density is maximum (usually the bottom of the atmosphere), as a function
        // of wavelength, in m^-1. The scattering coefficient at altitude h is equal
        // to 'rayleigh_scattering' times 'rayleigh_density' at this altitude. This
        // vector must have the same size as the wavelengths parameter.
        std::vector<double> rayleigh_scattering;
        // The density profile of aerosols, i.e. a function from altitude to
        // dimensionless values between 0 (null density) and 1 (maximum density).
        // Layers must be sorted from bottom to top. The width of the last layer is
        // ignored, i.e. it always extend to the top atmosphere boundary. At most 2
        // layers can be specified.
        std::vector<DensityProfileLayer> mie_density;
        // The scattering coefficient of aerosols at the altitude where their
        // density is maximum (usually the bottom of the atmosphere), as a function
        // of wavelength, in m^-1. The scattering coefficient at altitude h is equal
        // to 'mie_scattering' times 'mie_density' at this altitude. This vector
        // must have the same size as the wavelengths parameter.
        std::vector<double> mie_scattering;
        // The extinction coefficient of aerosols at the altitude where their
        // density is maximum (usually the bottom of the atmosphere), as a function
        // of wavelength, in m^-1. The extinction coefficient at altitude h is equal
        // to 'mie_extinction' times 'mie_density' at this altitude. This vector
        // must have the same size as the wavelengths parameter.
        std::vector<double> mie_extinction;
        // The asymetry parameter for the Cornette-Shanks phase function for the
        // aerosols.
        double mie_phase_function_g;
        // The density profile of air molecules that absorb light (e.g. ozone), i.e.
        // a function from altitude to dimensionless values between 0 (null density)
        // and 1 (maximum density). Layers must be sorted from bottom to top. The
        // width of the last layer is ignored, i.e. it always extend to the top
        // atmosphere boundary. At most 2 layers can be specified.
        std::vector<DensityProfileLayer> absorption_density;
        // The extinction coefficient of molecules that absorb light (e.g. ozone) at
        // the altitude where their density is maximum, as a function of wavelength,
        // in m^-1. The extinction coefficient at altitude h is equal to
        // 'absorption_extinction' times 'absorption_density' at this altitude. This
        // vector must have the same size as the wavelengths parameter.
        std::vector<double> absorption_extinction;
        // The average albedo of the ground, as a function of wavelength. This
        // vector must have the same size as the wavelengths parameter.
        std::vector<double> ground_albedo;
        // The maximum Sun zenith angle for which atmospheric scattering must be
        // precomputed, in radians (for maximum precision, use the smallest Sun
        // zenith angle yielding negligible sky light radiance values. For instance,
        // for the Earth case, 102 degrees is a good choice for most cases (120
        // degrees is necessary for very high exposure values).
        double max_sun_zenith_angle;
        // The length unit used in your shaders and meshes. This is the length unit
        // which must be used when calling the atmosphere model shader functions.
        double length_unit_in_meters;
        // The number of wavelengths for which atmospheric scattering must be
        // precomputed (the temporary GPU memory used during precomputations, and
        // the GPU memory used by the precomputed results, is independent of this
        // number, but the <i>precomputation time is directly proportional to this
        // number</i>):
        // - if this number is less than or equal to 3, scattering is precomputed
        // for 3 wavelengths, and stored as irradiance values. Then both the
        // radiance-based and the luminance-based API functions are provided (see
        // the above note).
        // - otherwise, scattering is precomputed for this number of wavelengths
        // (rounded up to a multiple of 3), integrated with the CIE color matching
        // functions, and stored as illuminance values. Then only the
        // luminance-based API functions are provided (see the above note).
        unsigned int num_precomputed_wavelengths;
    };

    class Model {
    public:
        Model() = default;
        Model(const Model&) = delete;
        Model& operator=(const Model&) = delete;
        Model(Model&& _Right) noexcept { 
            _Right.Swap(*this);
            _Right.Release();
        }
        Model& operator=(Model&& _Right) noexcept {
            _Right.Swap(*this);
            _Right.Release();
            return *this;
        }

        explicit Model(const AtmosphereParameters& _unknown, bool combinedtexture = true);
        ~Model() { this->Release(); }

        void Init(unsigned int num_scattering_orders = 4);
        
        void Swap(Model& _Right) {
            std::swap(_Myprofile, _Right._Myprofile);
            std::swap(shader_macros_, _Right.shader_macros_);
            std::swap(transmittance_texture_, _Right.transmittance_texture_);
            std::swap(scattering_texture_, _Right.scattering_texture_);
            std::swap(optional_single_mie_scattering_texture_, _Right.optional_single_mie_scattering_texture_);
            std::swap(irradiance_texture_, _Right.irradiance_texture_);
        }

        void Release() {
            if (glIsTexture(transmittance_texture_)) { glDeleteTextures(1, &transmittance_texture_); }
            if (glIsTexture(scattering_texture_)) { glDeleteTextures(1, &scattering_texture_); }
            if (glIsTexture(optional_single_mie_scattering_texture_)) { glDeleteShader(optional_single_mie_scattering_texture_); }
            if (glIsTexture(irradiance_texture_)) { glDeleteTextures(1, &irradiance_texture_); }

            _Myprofile = AtmosphereParameters();
            shader_macros_.clear();
            transmittance_texture_ = -1;
            scattering_texture_ = -1;
            optional_single_mie_scattering_texture_ = -1;
            irradiance_texture_ = -1;
        }

        GLuint GetTransmittanceTexture() const {
            assert( glIsTexture(transmittance_texture_) );
            return transmittance_texture_;
        }

        GLuint GetScatteringTexture() const {
            assert( glIsTexture(scattering_texture_) );
            return scattering_texture_;
        }

        GLuint GetIrradianceTexture() const {
            assert( glIsTexture(irradiance_texture_) );
            return irradiance_texture_;
        }

    private:
        typedef std::array<double, 3> vec3;
        typedef std::array<float, 9> mat3;

        void Precompute(
            GLuint fbo,
            GLuint delta_irradiance_texture,
            GLuint delta_rayleigh_scattering_texture,
            GLuint delta_mie_scattering_texture,
            GLuint delta_scattering_density_texture,
            GLuint delta_multiple_scattering_texture,
            const vec3& lambdas,
            const mat3& luminance_from_radiance,
            bool blend,
            unsigned int num_scattering_orders);

        AtmosphereParameters _Myprofile;
        std::map<std::string, std::string> shader_macros_;
        GLuint transmittance_texture_;
        GLuint scattering_texture_;// { RGB:Rayleigh, A:Mie }
        GLuint optional_single_mie_scattering_texture_;
        GLuint irradiance_texture_;
    };
}  // namespace atmosphere

struct GLuniformLocationAtmosphere {
    GLint solar_irradiance;
    GLint sun_angular_radius;
    GLint bottom_radius;
    GLint top_radius;

    GLint rayleigh_density[2][5];
    GLint rayleigh_scattering;

    GLint mie_density[2][5];
    GLint mie_scattering;
    GLint mie_extinction;
    GLint mie_phase_function_g;

    GLint absorption_density[2][5];
    GLint absorption_extinction;

    GLint ground_albedo;
    GLint mu_s_min;
};

inline GLuniformLocationAtmosphere glGetUniformLocationAtmosphere(GLuint program, const std::string& name) {
    GLuniformLocationAtmosphere location;
    
    location.solar_irradiance   = glGetUniformLocation(program, (name + ".solar_irradiance").c_str());
    location.sun_angular_radius = glGetUniformLocation(program, (name + ".sun_angular_radius").c_str());
    location.bottom_radius = glGetUniformLocation(program, (name + ".bottom_radius").c_str());
    location.top_radius = glGetUniformLocation(program, (name + ".top_radius").c_str());

    location.rayleigh_density[0][0] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[0].width").c_str());
    location.rayleigh_density[0][1] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[0].exp_term").c_str());
    location.rayleigh_density[0][2] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[0].exp_scale").c_str());
    location.rayleigh_density[0][3] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[0].linear_term").c_str());
    location.rayleigh_density[0][4] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[0].constant_term").c_str());
    location.rayleigh_density[1][0] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[1].width").c_str());
    location.rayleigh_density[1][1] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[1].exp_term").c_str());
    location.rayleigh_density[1][2] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[1].exp_scale").c_str());
    location.rayleigh_density[1][3] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[1].linear_term").c_str());
    location.rayleigh_density[1][4] = glGetUniformLocation(program, (name + ".rayleigh_density.layers[1].constant_term").c_str());
    location.rayleigh_scattering = glGetUniformLocation(program, (name + ".rayleigh_scattering").c_str());

    location.mie_density[0][0] = glGetUniformLocation(program, (name + ".mie_density.layers[0].width").c_str());
    location.mie_density[0][1] = glGetUniformLocation(program, (name + ".mie_density.layers[0].exp_term").c_str());
    location.mie_density[0][2] = glGetUniformLocation(program, (name + ".mie_density.layers[0].exp_scale").c_str());
    location.mie_density[0][3] = glGetUniformLocation(program, (name + ".mie_density.layers[0].linear_term").c_str());
    location.mie_density[0][4] = glGetUniformLocation(program, (name + ".mie_density.layers[0].constant_term").c_str());
    location.mie_density[1][0] = glGetUniformLocation(program, (name + ".mie_density.layers[1].width").c_str());
    location.mie_density[1][1] = glGetUniformLocation(program, (name + ".mie_density.layers[1].exp_term").c_str());
    location.mie_density[1][2] = glGetUniformLocation(program, (name + ".mie_density.layers[1].exp_scale").c_str());
    location.mie_density[1][3] = glGetUniformLocation(program, (name + ".mie_density.layers[1].linear_term").c_str());
    location.mie_density[1][4] = glGetUniformLocation(program, (name + ".mie_density.layers[1].constant_term").c_str());
    location.mie_scattering = glGetUniformLocation(program, (name + ".mie_scattering").c_str());
    location.mie_extinction = glGetUniformLocation(program, (name + ".mie_extinction").c_str());
    location.mie_phase_function_g = glGetUniformLocation(program, (name + ".mie_phase_function_g").c_str());

    location.absorption_density[0][0] = glGetUniformLocation(program, (name + ".absorption_density.layers[0].width").c_str());
    location.absorption_density[0][1] = glGetUniformLocation(program, (name + ".absorption_density.layers[0].exp_term").c_str());
    location.absorption_density[0][2] = glGetUniformLocation(program, (name + ".absorption_density.layers[0].exp_scale").c_str());
    location.absorption_density[0][3] = glGetUniformLocation(program, (name + ".absorption_density.layers[0].linear_term").c_str());
    location.absorption_density[0][4] = glGetUniformLocation(program, (name + ".absorption_density.layers[0].constant_term").c_str());
    location.absorption_density[1][0] = glGetUniformLocation(program, (name + ".absorption_density.layers[1].width").c_str());
    location.absorption_density[1][1] = glGetUniformLocation(program, (name + ".absorption_density.layers[1].exp_term").c_str());
    location.absorption_density[1][2] = glGetUniformLocation(program, (name + ".absorption_density.layers[1].exp_scale").c_str());
    location.absorption_density[1][3] = glGetUniformLocation(program, (name + ".absorption_density.layers[1].linear_term").c_str());
    location.absorption_density[1][4] = glGetUniformLocation(program, (name + ".absorption_density.layers[1].constant_term").c_str());
    location.absorption_extinction = glGetUniformLocation(program, (name + ".absorption_extinction").c_str());
    
    location.ground_albedo = glGetUniformLocation(program, (name + ".ground_albedo").c_str());

    location.mu_s_min = glGetUniformLocation(program, (name + ".mu_s_min").c_str());

    return std::move(location);
}

inline void glUniformAtmosphere(const GLuniformLocationAtmosphere& location, const std::array<double,3> lambdas, const atmosphere::AtmosphereParameters& atmosphere) {
    // ...
    glUniform3f(location.solar_irradiance, physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.solar_irradiance, lambdas[0]),
                                            physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.solar_irradiance, lambdas[1]),
                                            physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.solar_irradiance, lambdas[2]));
    glUniform1f(location.sun_angular_radius, atmosphere.sun_angular_radius);
    glUniform1f(location.bottom_radius, atmosphere.bottom_radius / atmosphere.length_unit_in_meters);
    glUniform1f(location.top_radius, atmosphere.top_radius / atmosphere.length_unit_in_meters);
   
    // Rayleigh
    for (size_t i = 0; i != 2; ++i) {
        glUniform1f(location.rayleigh_density[i][0], atmosphere.rayleigh_density[i].width / atmosphere.length_unit_in_meters);
        glUniform1f(location.rayleigh_density[i][1], atmosphere.rayleigh_density[i].exp_term);
        glUniform1f(location.rayleigh_density[i][2], atmosphere.rayleigh_density[i].exp_scale * atmosphere.length_unit_in_meters);
        glUniform1f(location.rayleigh_density[i][3], atmosphere.rayleigh_density[i].linear_term * atmosphere.length_unit_in_meters);
        glUniform1f(location.rayleigh_density[i][4], atmosphere.rayleigh_density[i].constant_term);
    }
    glUniform3f(location.rayleigh_scattering, physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.rayleigh_scattering, lambdas[0]) * atmosphere.length_unit_in_meters,
                                              physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.rayleigh_scattering, lambdas[1]) * atmosphere.length_unit_in_meters,
                                              physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.rayleigh_scattering, lambdas[2]) * atmosphere.length_unit_in_meters);

    // Mie..
    for (size_t i = 0; i != 2; ++i) {
        glUniform1f(location.mie_density[i][0], atmosphere.mie_density[i].width / atmosphere.length_unit_in_meters);
        glUniform1f(location.mie_density[i][1], atmosphere.mie_density[i].exp_term);
        glUniform1f(location.mie_density[i][2], atmosphere.mie_density[i].exp_scale * atmosphere.length_unit_in_meters);
        glUniform1f(location.mie_density[i][3], atmosphere.mie_density[i].linear_term * atmosphere.length_unit_in_meters);
        glUniform1f(location.mie_density[i][4], atmosphere.mie_density[i].constant_term);
    }
    glUniform3f(location.mie_scattering, physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.mie_scattering, lambdas[0]) * atmosphere.length_unit_in_meters,
                                         physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.mie_scattering, lambdas[1]) * atmosphere.length_unit_in_meters,
                                         physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.mie_scattering, lambdas[2]) * atmosphere.length_unit_in_meters);
    glUniform3f(location.mie_extinction, physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.mie_extinction, lambdas[0]) * atmosphere.length_unit_in_meters,
                                         physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.mie_extinction, lambdas[1]) * atmosphere.length_unit_in_meters,
                                         physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.mie_extinction, lambdas[2]) * atmosphere.length_unit_in_meters);
    glUniform1f(location.mie_phase_function_g, atmosphere.mie_phase_function_g);

    // Ozone..
    for (size_t i = 0; i != 2; ++i) {
        glUniform1f(location.absorption_density[i][0], atmosphere.absorption_density[i].width / atmosphere.length_unit_in_meters);
        glUniform1f(location.absorption_density[i][1], atmosphere.absorption_density[i].exp_term);
        glUniform1f(location.absorption_density[i][2], atmosphere.absorption_density[i].exp_scale * atmosphere.length_unit_in_meters);
        glUniform1f(location.absorption_density[i][3], atmosphere.absorption_density[i].linear_term * atmosphere.length_unit_in_meters);
        glUniform1f(location.absorption_density[i][4], atmosphere.absorption_density[i].constant_term);
    }
    glUniform3f(location.absorption_extinction, physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.absorption_extinction, lambdas[0]) * atmosphere.length_unit_in_meters,
                                                physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.absorption_extinction, lambdas[1]) * atmosphere.length_unit_in_meters,
                                                physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.absorption_extinction, lambdas[2]) * atmosphere.length_unit_in_meters);

    // ground albedo
    glUniform3f(location.ground_albedo, physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.ground_albedo, lambdas[0]),
                                        physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.ground_albedo, lambdas[1]),
                                        physics::InterpolateSpectrum(atmosphere.wavelengths, atmosphere.ground_albedo, lambdas[2]));

    // 
    glUniform1f(location.mu_s_min, cos(atmosphere.max_sun_zenith_angle));
}

inline void glUniformAtmospherescatteringTexture(GLint transmittance_texture_location, GLint scattering_texture_location, GLint irradiance_texture_location,
    GLenum transmittance_texture_unit, GLenum scattering_texture_unit, GLenum irradiance_texture_unit, const atmosphere::Model& atmosphere_model) {
    glActiveTexture(transmittance_texture_unit);
    glBindTexture(GL_TEXTURE_2D, atmosphere_model.GetTransmittanceTexture());
    glUniform1i(transmittance_texture_location, transmittance_texture_unit - GL_TEXTURE0);

    glActiveTexture(scattering_texture_unit);
    glBindTexture(GL_TEXTURE_3D, atmosphere_model.GetScatteringTexture());
    glUniform1i(scattering_texture_location, scattering_texture_unit - GL_TEXTURE0);

    glActiveTexture(irradiance_texture_unit);
    glBindTexture(GL_TEXTURE_2D, atmosphere_model.GetIrradianceTexture());
    glUniform1i(irradiance_texture_location, irradiance_texture_unit - GL_TEXTURE0);
}

#endif  // ATMOSPHERE_MODEL_H_
