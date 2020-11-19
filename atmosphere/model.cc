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

/*<h2>atmosphere/model.cc</h2>

<p>This file implements the <a href="model.h.html">API of our atmosphere
model</a>. Its main role is to precompute the transmittance, scattering and
irradiance textures. The GLSL functions to precompute them are provided in
<a href="functions.glsl.html">functions.glsl</a>, but they are not sufficient.
They must be used in fully functional shaders and programs, and these programs
must be called in the correct order, with the correct input and output textures
(via framebuffer objects), to precompute each scattering order in sequence, as
described in Algorithm 4.1 of
<a href="https://hal.inria.fr/inria-00288758/en">our paper</a>. This is the role
of the following C++ code.
*/

#include "model.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <iostream>

#include "constants.h"

/*
<p>The rest of this file is organized in 3 parts:
<ul>
<li>the <a href="#shaders">first part</a> defines the shaders used to precompute
the atmospheric textures,</li>
<li>the <a href="#utilities">second part</a> provides utility classes and
functions used to compile shaders, create textures, draw quads, etc,</li>
<li>the <a href="#implementation">third part</a> provides the actual
implementation of the <code>Model</code> class, using the above tools.</li>
</ul>

<h3 id="shaders">Shader definitions</h3>

<p>In order to precompute a texture we attach it to a framebuffer object (FBO)
and we render a full quad in this FBO. For this we need a basic vertex shader:
*/

namespace atmosphere {

namespace {

    const char kVertexShader[] = R"(
        #version 330
        layout(location = 0) in vec2 vertex;
        void main() {
          gl_Position = vec4(vertex, 0.0, 1.0);
        })";

    /*
    <p>a basic geometry shader (only for 3D textures, to specify in which layer we
    want to write):
    */

    const char kGeometryShader[] = R"(
        #version 330
        layout(triangles) in;
        layout(triangle_strip, max_vertices = 3) out;
        uniform int layer;
        void main() {
          gl_Position = gl_in[0].gl_Position;
          gl_Layer = layer;
          EmitVertex();
          gl_Position = gl_in[1].gl_Position;
          gl_Layer = layer;
          EmitVertex();
          gl_Position = gl_in[2].gl_Position;
          gl_Layer = layer;
          EmitVertex();
          EndPrimitive();
        })";

    /*
    <p>and a fragment shader, which depends on the texture we want to compute. This
    is the role of the following shaders, which simply wrap the precomputation
    functions from <a href="functions.glsl.html">functions.glsl</a> in complete
    shaders (with a <code>main</code> function and a proper declaration of the
    shader inputs and outputs). Note that these strings must be concatenated with
    <code>definitions.glsl</code> and <code>functions.glsl</code> (provided as C++
    string literals by the generated <code>.glsl.inc</code> files), as well as with
    a definition of the <code>ATMOSPHERE</code> constant - containing the atmosphere
    parameters, to really get a complete shader. Note also the
    <code>luminance_from_radiance</code> uniforms: these are used in precomputed
    illuminance mode to convert the radiance values computed by the
    <code>functions.glsl</code> functions to luminance values (see the
    <code>Init</code> method for more details).
    */


    const char kComputeTransmittanceShader[] = R"(
        layout(location = 0) out vec3 transmittance;
        void main() {
          transmittance = ComputeTransmittanceToTopAtmosphereBoundaryTexture(
              ATMOSPHERE, gl_FragCoord.xy);
        })";

    const char kComputeDirectIrradianceShader[] = R"(
        layout(location = 0) out vec3 delta_irradiance;
        layout(location = 1) out vec3 irradiance;
        uniform sampler2D transmittance_texture;
        void main() {
          delta_irradiance = ComputeDirectIrradianceTexture(
              ATMOSPHERE, transmittance_texture, gl_FragCoord.xy);
          irradiance = vec3(0.0);
        })";

    const char kComputeSingleScatteringShader[] = R"(
        layout(location = 0) out vec3 delta_rayleigh;
        layout(location = 1) out vec3 delta_mie;
        layout(location = 2) out vec4 scattering;
        layout(location = 3) out vec3 single_mie_scattering;
        uniform mat3 luminance_from_radiance;
        uniform sampler2D transmittance_texture;
        uniform int layer;
        void main() {
          ComputeSingleScatteringTexture(
              ATMOSPHERE, transmittance_texture, vec3(gl_FragCoord.xy, layer + 0.5),
              delta_rayleigh, delta_mie);
          scattering = vec4(luminance_from_radiance * delta_rayleigh.rgb,
              (luminance_from_radiance * delta_mie).r);
          single_mie_scattering = luminance_from_radiance * delta_mie;
        })";

    const char kComputeScatteringDensityShader[] = R"(
        layout(location = 0) out vec3 scattering_density;
        uniform sampler2D transmittance_texture;
        uniform sampler3D single_rayleigh_scattering_texture;
        uniform sampler3D single_mie_scattering_texture;
        uniform sampler3D multiple_scattering_texture;
        uniform sampler2D irradiance_texture;
        uniform int scattering_order;
        uniform int layer;
        void main() {
          scattering_density = ComputeScatteringDensityTexture(
              ATMOSPHERE, transmittance_texture, single_rayleigh_scattering_texture,
              single_mie_scattering_texture, multiple_scattering_texture,
              irradiance_texture, vec3(gl_FragCoord.xy, layer + 0.5),
              scattering_order);
        })";

    const char kComputeIndirectIrradianceShader[] = R"(
        layout(location = 0) out vec3 delta_irradiance;
        layout(location = 1) out vec3 irradiance;
        uniform mat3 luminance_from_radiance;
        uniform sampler3D single_rayleigh_scattering_texture;
        uniform sampler3D single_mie_scattering_texture;
        uniform sampler3D multiple_scattering_texture;
        uniform int scattering_order;
        void main() {
          delta_irradiance = ComputeIndirectIrradianceTexture(
              ATMOSPHERE, single_rayleigh_scattering_texture,
              single_mie_scattering_texture, multiple_scattering_texture,
              gl_FragCoord.xy, scattering_order);
          irradiance = luminance_from_radiance * delta_irradiance;
        })";

    const char kComputeMultipleScatteringShader[] = R"(
        layout(location = 0) out vec3 delta_multiple_scattering;
        layout(location = 1) out vec4 scattering;
        uniform mat3 luminance_from_radiance;
        uniform sampler2D transmittance_texture;
        uniform sampler3D scattering_density_texture;
        uniform int layer;
        void main() {
          float nu;
          delta_multiple_scattering = ComputeMultipleScatteringTexture(
              ATMOSPHERE, transmittance_texture, scattering_density_texture,
              vec3(gl_FragCoord.xy, layer + 0.5), nu);
          scattering = vec4(
              luminance_from_radiance *
                  delta_multiple_scattering.rgb / RayleighPhaseFunction(nu),
              0.0);
        })";

    /*
    <p>We finally need a shader implementing the GLSL functions exposed in our API,
    which can be done by calling the corresponding functions in
    <a href="functions.glsl.html#rendering">functions.glsl</a>, with the precomputed
    texture arguments taken from uniform variables (note also the
    *<code>_RADIANCE_TO_LUMINANCE</code> conversion constants in the last functions:
    they are computed in the <a href="#utilities">second part</a> below, and their
    definitions are concatenated to this GLSL code to get a fully functional
    shader).
    */

    const char kAtmosphereShader[] = R"(
        uniform sampler2D transmittance_texture;
        uniform sampler3D scattering_texture;
        uniform sampler3D single_mie_scattering_texture;
        uniform sampler2D irradiance_texture;
        #ifdef RADIANCE_API_ENABLED
        RadianceSpectrum GetSolarRadiance() {
          return ATMOSPHERE.solar_irradiance /
              (PI * ATMOSPHERE.sun_angular_radius * ATMOSPHERE.sun_angular_radius);
        }
        RadianceSpectrum GetSkyRadiance(
            Position camera, Direction view_ray, Length shadow_length,
            Direction sun_direction, out DimensionlessSpectrum transmittance) {
          return GetSkyRadiance(ATMOSPHERE, transmittance_texture,
              scattering_texture, single_mie_scattering_texture,
              camera, view_ray, shadow_length, sun_direction, transmittance);
        }
        RadianceSpectrum GetSkyRadianceToPoint(
            Position camera, Position point, Length shadow_length,
            Direction sun_direction, out DimensionlessSpectrum transmittance) {
          return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture,
              scattering_texture, single_mie_scattering_texture,
              camera, point, shadow_length, sun_direction, transmittance);
        }
        IrradianceSpectrum GetSunAndSkyIrradiance(
           Position p, Direction normal, Direction sun_direction,
           out IrradianceSpectrum sky_irradiance) {
          return GetSunAndSkyIrradiance(ATMOSPHERE, transmittance_texture,
              irradiance_texture, p, normal, sun_direction, sky_irradiance);
        }
        #endif
        Luminance3 GetSolarLuminance() {
          return ATMOSPHERE.solar_irradiance /
              (PI * ATMOSPHERE.sun_angular_radius * ATMOSPHERE.sun_angular_radius) *
              SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
        }
        Luminance3 GetSkyLuminance(
            Position camera, Direction view_ray, Length shadow_length,
            Direction sun_direction, out DimensionlessSpectrum transmittance) {
          return GetSkyRadiance(ATMOSPHERE, transmittance_texture,
              scattering_texture, single_mie_scattering_texture,
              camera, view_ray, shadow_length, sun_direction, transmittance) *
              SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
        }
        Luminance3 GetSkyLuminanceToPoint(
            Position camera, Position point, Length shadow_length,
            Direction sun_direction, out DimensionlessSpectrum transmittance) {
          return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture,
              scattering_texture, single_mie_scattering_texture,
              camera, point, shadow_length, sun_direction, transmittance) *
              SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
        }
        Illuminance3 GetSunAndSkyIlluminance(
           Position p, Direction normal, Direction sun_direction,
           out IrradianceSpectrum sky_irradiance) {
          IrradianceSpectrum sun_irradiance = GetSunAndSkyIrradiance(
              ATMOSPHERE, transmittance_texture, irradiance_texture, p, normal,
              sun_direction, sky_irradiance);
          sky_irradiance *= SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
          return sun_irradiance * SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
        })";

    /*<h3 id="utilities">Utility classes and functions</h3>

    <p>To compile and link these shaders into programs, and to set their uniforms,
    we use the following utility class:
    */

    GLuint NewTexture2d(int width, int height) {
      GLuint texture;
      glGenTextures(1, &texture);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      // 16F precision for the transmittance gives artifacts.
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
      return texture;
    }

    GLuint NewTexture3d(int width, int height, int depth, GLenum format) {
      GLuint texture;
      glGenTextures(1, &texture);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_3D, texture);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      GLenum internal_format = format == GL_RGBA ? GL_RGBA32F : GL_RGB32F;
      glTexImage3D(GL_TEXTURE_3D, 0, internal_format, width, height, depth, 0, format, GL_FLOAT, NULL);
      return texture;
    }

    /*
    <p>a function to test whether the RGB format is a supported renderbuffer color
    format (the OpenGL 3.3 Core Profile specification requires support for the RGBA
    formats, but not for the RGB ones):
    */

    bool IsFramebufferRgbFormatSupported(bool half_precision) {
        GLuint test_fbo = 0;
        glGenFramebuffers(1, &test_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, test_fbo);
        GLuint test_texture = 0;
        glGenTextures(1, &test_texture);
        glBindTexture(GL_TEXTURE_2D, test_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, half_precision ? GL_RGB16F : GL_RGB32F,
                    1, 1, 0, GL_RGB, GL_FLOAT, NULL);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                GL_TEXTURE_2D, test_texture, 0);
        bool rgb_format_supported =
            glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        glDeleteTextures(1, &test_texture);
        glDeleteFramebuffers(1, &test_fbo);
        return rgb_format_supported;
    }

}  // anonymous namespace

/*<h3 id="implementation">Model implementation</h3>

<p>Using the above utility functions and classes, we can now implement the
constructor of the <code>Model</code> class. This constructor generates a piece
of GLSL code that defines an <code>ATMOSPHERE</code> constant containing the
atmosphere parameters (we use constants instead of uniforms to enable constant
folding and propagation optimizations in the GLSL compiler), concatenated with
<a href="functions.glsl.html">functions.glsl</a>, and with
<code>kAtmosphereShader</code>, to get the shader exposed by our API in
<code>GetShader</code>. It also allocates the precomputed textures (but does not
initialize them), as well as a vertex buffer object to render a full screen quad
(used to render into the precomputed textures).
*/
Model::Model(const AtmosphereParameters& _unknown, bool combinedtexture)
    : _Myprofile(_unknown), transmittance_texture_(-1), scattering_texture_(-1), irradiance_texture_(-1) {
    
    transmittance_texture_ = NewTexture2d(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
    scattering_texture_    = NewTexture3d(SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA);
    irradiance_texture_    = NewTexture2d(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
    if (!combinedtexture) {
        optional_single_mie_scattering_texture_ = NewTexture3d(SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA);
    }

    /* Compute the values for the SKY_RADIANCE_TO_LUMINANCE constant. In theory
        this should be 1 in precomputed illuminance mode (because the precomputed
        textures already contain illuminance values). In practice, however, storing
        true illuminance values in half precision textures yields artefacts
        (because the values are too large), so we store illuminance values divided
        by MAX_LUMINOUS_EFFICACY instead. This is why, in precomputed illuminance
        mode, we set SKY_RADIANCE_TO_LUMINANCE to MAX_LUMINOUS_EFFICACY. */
    std::array<double, 3> sky_k;
    bool precompute_illuminance = _Myprofile.num_precomputed_wavelengths > 3;
    if (precompute_illuminance) {
        sky_k = { physics::MAX_LUMINOUS_EFFICACY, physics::MAX_LUMINOUS_EFFICACY, physics::MAX_LUMINOUS_EFFICACY };
    } else {
        sky_k = physics::ComputeSpectralRadianceToLuminanceFactors(_Myprofile.wavelengths, _Myprofile.solar_irradiance,
            -3.0/* lambda_power */);
    }

    /* Compute the values for the SUN_RADIANCE_TO_LUMINANCE constant. 
    */
    std::array<double, 3> sun_k = physics::ComputeSpectralRadianceToLuminanceFactors(_Myprofile.wavelengths, _Myprofile.solar_irradiance,
                                        0.0 /* lambda_power */);

    shader_macros_.insert_or_assign("COMBINED_SCATTERING_TEXTURES", combinedtexture ? "1" : "0");
    shader_macros_.insert_or_assign("TRANSMITTANCE_TEXTURE_WIDTH",  std::to_string(TRANSMITTANCE_TEXTURE_WIDTH));
    shader_macros_.insert_or_assign("TRANSMITTANCE_TEXTURE_HEIGHT", std::to_string(TRANSMITTANCE_TEXTURE_HEIGHT));
    shader_macros_.insert_or_assign("SCATTERING_TEXTURE_R_SIZE",    std::to_string(SCATTERING_TEXTURE_R_SIZE));
    shader_macros_.insert_or_assign("SCATTERING_TEXTURE_MU_SIZE",   std::to_string(SCATTERING_TEXTURE_MU_SIZE));
    shader_macros_.insert_or_assign("SCATTERING_TEXTURE_MU_S_SIZE", std::to_string(SCATTERING_TEXTURE_MU_S_SIZE));
    shader_macros_.insert_or_assign("SCATTERING_TEXTURE_NU_SIZE",   std::to_string(SCATTERING_TEXTURE_NU_SIZE));
    shader_macros_.insert_or_assign("IRRADIANCE_TEXTURE_WIDTH",     std::to_string(IRRADIANCE_TEXTURE_WIDTH));
    shader_macros_.insert_or_assign("IRRADIANCE_TEXTURE_HEIGHT",    std::to_string(IRRADIANCE_TEXTURE_HEIGHT));
    shader_macros_.insert_or_assign("SKY_SPECTRAL_RADIANCE_TO_LUMINANCE", "vec3(" + std::to_string(sky_k[0]) + std::to_string(sky_k[1]) + std::to_string(sky_k[2]) + ")");
    shader_macros_.insert_or_assign("SUN_SPECTRAL_RADIANCE_TO_LUMINANCE", "vec3(" + std::to_string(sun_k[0]) + std::to_string(sun_k[1]) + std::to_string(sun_k[2]) + ")");
}

/*
<p>The Init method precomputes the atmosphere textures. It first allocates the
temporary resources it needs, then calls <code>Precompute</code> to do the
actual precomputations, and finally destroys the temporary resources.

<p>Note that there are two precomputation modes here, depending on whether we
want to store precomputed irradiance or illuminance values:
<ul>
  <li>In precomputed irradiance mode, we simply need to call
  <code>Precompute</code> with the 3 wavelengths for which we want to precompute
  irradiance, namely <code>kLambdaR</code>, <code>kLambdaG</code>,
  <code>kLambdaB</code> (with the identity matrix for
  <code>luminance_from_radiance</code>, since we don't want any conversion from
  radiance to luminance)</li>
  <li>In precomputed illuminance mode, we need to precompute irradiance for
  <code>num_precomputed_wavelengths_</code>, and then integrate the results,
  multiplied with the 3 CIE xyz color matching functions and the XYZ to sRGB
  matrix to get sRGB illuminance values.
  <p>A naive solution would be to allocate temporary textures for the
  intermediate irradiance results, then perform the integration from irradiance
  to illuminance and store the result in the final precomputed texture. In
  pseudo-code (and assuming one wavelength per texture instead of 3):
  <pre>
    create n temporary irradiance textures
    for each wavelength lambda in the n wavelengths:
       precompute irradiance at lambda into one of the temporary textures
    initializes the final illuminance texture with zeros
    for each wavelength lambda in the n wavelengths:
      accumulate in the final illuminance texture the product of the
      precomputed irradiance at lambda (read from the temporary textures)
      with the value of the 3 sRGB color matching functions at lambda (i.e.
      the product of the XYZ to sRGB matrix with the CIE xyz color matching
      functions).
  </pre>
  <p>However, this be would waste GPU memory. Instead, we can avoid allocating
  temporary irradiance textures, by merging the two above loops:
  <pre>
    for each wavelength lambda in the n wavelengths:
      accumulate in the final illuminance texture (or, for the first
      iteration, set this texture to) the product of the precomputed
      irradiance at lambda (computed on the fly) with the value of the 3
      sRGB color matching functions at lambda.
  </pre>
  <p>This is the method we use below, with 3 wavelengths per iteration instead
  of 1, using <code>Precompute</code> to compute 3 irradiances values per
  iteration, and <code>luminance_from_radiance</code> to multiply 3 irradiances
  with the values of the 3 sRGB color matching functions at 3 different
  wavelengths (yielding a 3x3 matrix).</li>
</ul>

<p>This yields the following implementation:
*/
void Model::Init(unsigned int num_scattering_orders) {
    // The precomputations require temporary textures, in particular to store the
    // contribution of one scattering order, which is needed to compute the next
    // order of scattering (the final precomputed textures store the sum of all
    // the scattering orders). We allocate them here, and destroy them at the end
    // of this method.
    GLuint delta_irradiance_texture 
        = NewTexture2d( IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
    GLuint delta_rayleigh_scattering_texture
        = NewTexture3d( SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA);
    GLuint delta_mie_scattering_texture
        = NewTexture3d( SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA);
    GLuint delta_scattering_density_texture
        = NewTexture3d( SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, GL_RGBA);
    // delta_multiple_scattering_texture is only needed to compute scattering
    // order 3 or more, while delta_rayleigh_scattering_texture and
    // delta_mie_scattering_texture are only needed to compute double scattering.
    // Therefore, to save memory, we can store delta_rayleigh_scattering_texture
    // and delta_multiple_scattering_texture in the same GPU texture.
    GLuint delta_multiple_scattering_texture = delta_rayleigh_scattering_texture;

    // The precomputations also require a temporary framebuffer object, created
    // here (and destroyed at the end of this method).
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // The actual precomputations depend on whether we want to store precomputed
    // irradiance or illuminance values.
    if (_Myprofile.num_precomputed_wavelengths <= 3) {
        vec3 lambdas{ physics::visiblelight_red<double>, physics::visiblelight_green<double>, physics::visiblelight_blue<double> };
        mat3 luminance_from_radiance{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
        Precompute(fbo, 
            delta_irradiance_texture, 
            delta_rayleigh_scattering_texture, 
            delta_mie_scattering_texture, 
            delta_scattering_density_texture,
            delta_multiple_scattering_texture,
            lambdas, luminance_from_radiance,
            false /* blend */, 
            num_scattering_orders);
    } else {
        constexpr double kLambdaMin = 360.0;
        constexpr double kLambdaMax = 830.0;
        int num_iterations = (_Myprofile.num_precomputed_wavelengths + 2) / 3;
        double dlambda = (kLambdaMax - kLambdaMin) / (3 * num_iterations);
        for (int i = 0; i < num_iterations; ++i) {
            vec3 lambdas = vec3{ kLambdaMin + (3 * i + 0.5) * dlambda,
                                 kLambdaMin + (3 * i + 1.5) * dlambda,
                                 kLambdaMin + (3 * i + 2.5) * dlambda };

            auto coeff = [dlambda](double lambda, int component) {
                // Note that we don't include MAX_LUMINOUS_EFFICACY here, to avoid
                // artefacts due to too large values when using half precision on GPU.
                // We add this term back in kAtmosphereShader, via
                // SKY_SPECTRAL_RADIANCE_TO_LUMINANCE (see also the comments in the
                // Model constructor).
                double x = physics::CieColorMatchingFunctionTableValue(lambda, 1);
                double y = physics::CieColorMatchingFunctionTableValue(lambda, 2);
                double z = physics::CieColorMatchingFunctionTableValue(lambda, 3);
                return static_cast<float>((
                    physics::XYZ_TO_SRGB[component * 3] * x +
                    physics::XYZ_TO_SRGB[component * 3 + 1] * y +
                    physics::XYZ_TO_SRGB[component * 3 + 2] * z) * dlambda);
            };

            mat3 luminance_from_radiance = mat3{ coeff(lambdas[0], 0), coeff(lambdas[1], 0), coeff(lambdas[2], 0),
                                                 coeff(lambdas[0], 1), coeff(lambdas[1], 1), coeff(lambdas[2], 1),
                                                 coeff(lambdas[0], 2), coeff(lambdas[1], 2), coeff(lambdas[2], 2) };
            
            Precompute(fbo, 
                delta_irradiance_texture, 
                delta_rayleigh_scattering_texture, 
                delta_mie_scattering_texture, 
                delta_scattering_density_texture, 
                delta_multiple_scattering_texture,
                lambdas, luminance_from_radiance, 
                i > 0 /* blend */,
                num_scattering_orders);
        }

        // After the above iterations, the transmittance texture contains the
        // transmittance for the 3 wavelengths used at the last iteration. But we
        // want the transmittance at kLambdaR, kLambdaG, kLambdaB instead, so we
        // must recompute it here for these 3 wavelengths:
        GLprogram compute_transmittance(GLvertshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.vert"))), 
                                        GLfragshader(GLshadersource(std::filesystem::path("atmosphere/ComputeTransmittance.frag.h"), this->shader_macros_)) );
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, transmittance_texture_, 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glViewport(0, 0, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
        glUseProgram(compute_transmittance);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    // Delete the temporary resources allocated at the begining of this method.
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &delta_scattering_density_texture);
    glDeleteTextures(1, &delta_mie_scattering_texture);
    glDeleteTextures(1, &delta_rayleigh_scattering_texture);
    glDeleteTextures(1, &delta_irradiance_texture);
    assert( glGetError() == GL_NO_ERROR );
}

/*
<p>Finally, we provide the actual implementation of the precomputation algorithm
described in Algorithm 4.1 of
<a href="https://hal.inria.fr/inria-00288758/en">our paper</a>. Each step is
explained by the inline comments below.
*/
void Model::Precompute(
    GLuint fbo,
    GLuint delta_irradiance_texture,
    GLuint delta_rayleigh_scattering_texture,
    GLuint delta_mie_scattering_texture,
    GLuint delta_scattering_density_texture,
    GLuint delta_multiple_scattering_texture,
    const vec3& lambdas,
    const mat3& luminance_from_radiance,
    bool blend,
    unsigned int num_scattering_orders) 
{
    /* 1. Create precompute-shaders */
    GLprogram compute_transmittance(GLvertshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.vert"))), 
                                    GLfragshader(GLshadersource(std::filesystem::path("atmosphere/ComputeTransmittance.frag.h"), this->shader_macros_)) );

    GLprogram compute_direct_irradiance(GLvertshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.vert"))),
                                        GLfragshader(GLshadersource(std::filesystem::path("atmosphere/ComputeDirectIrradiance.frag.h"), this->shader_macros_)) );
    
    GLprogram compute_single_scattering(GLvertshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.vert"))),
                                        GLgeomshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.geom"))),
                                        GLfragshader(GLshadersource(std::filesystem::path("atmosphere/ComputeSingleScattering.frag.h"), this->shader_macros_)) );
    
    GLprogram compute_scattering_density(GLvertshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.vert"))),
                                         GLgeomshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.geom"))),
                                         GLfragshader(GLshadersource(std::filesystem::path("atmosphere/ComputeScatteringDensity.frag.h"), this->shader_macros_)) );
    
    GLprogram compute_indirect_irradiance(GLvertshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.vert"))), 
                                          GLfragshader(GLshadersource(std::filesystem::path("atmosphere/ComputeIndirectIrradiance.frag.h"), this->shader_macros_)) );
    
    GLprogram compute_multiple_scattering(GLvertshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.vert"))),
                                          GLgeomshader(GLshadersource(std::filesystem::path("atmosphere/Compute_.geom"))),
                                          GLfragshader(GLshadersource(std::filesystem::path("atmosphere/ComputeMultipleScattering.frag.h"), this->shader_macros_)) );
    {
        glUseProgram(compute_transmittance);
        glUniformAtmosphere(glGetUniformLocationAtmosphere(compute_transmittance, "ATMOSPHERE"), lambdas, this->_Myprofile);

        glUseProgram(compute_direct_irradiance);
        glUniformAtmosphere(glGetUniformLocationAtmosphere(compute_direct_irradiance, "ATMOSPHERE"), lambdas, this->_Myprofile);

        glUseProgram(compute_single_scattering);
        glUniformAtmosphere(glGetUniformLocationAtmosphere(compute_single_scattering, "ATMOSPHERE"), lambdas, this->_Myprofile);

        glUseProgram(compute_scattering_density);
        glUniformAtmosphere(glGetUniformLocationAtmosphere(compute_scattering_density, "ATMOSPHERE"), lambdas, this->_Myprofile);

        glUseProgram(compute_indirect_irradiance);
        glUniformAtmosphere(glGetUniformLocationAtmosphere(compute_indirect_irradiance, "ATMOSPHERE"), lambdas, this->_Myprofile);

        glUseProgram(compute_multiple_scattering);
        glUniformAtmosphere(glGetUniformLocationAtmosphere(compute_multiple_scattering, "ATMOSPHERE"), lambdas, this->_Myprofile);
    }

    const GLuint kDrawBuffers[4] = {
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2,
        GL_COLOR_ATTACHMENT3
    };
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);

    /* 1. Compute transmittance into transmittance_texture_ */
    {
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, this->transmittance_texture_, 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glViewport(0, 0, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
        glUseProgram(compute_transmittance);
        glDisablei(GL_BLEND, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);// similar GL_QUADS
    }
    assert(glGetError() == GL_NO_ERROR);

    // Compute the direct irradiance, store it in delta_irradiance_texture and,
    // depending on 'blend', either initialize irradiance_texture_ with zeros or
    // leave it unchanged (we don't want the direct irradiance in
    // irradiance_texture_, but only the irradiance from the sky).
    /* 2. Compute direct-irradiance */
    {
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_irradiance_texture, 0);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, this->irradiance_texture_, 0);
        glDrawBuffers(2, kDrawBuffers);
        glViewport(0, 0, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

        glUseProgram(compute_direct_irradiance);
        glUniformTexture(glGetUniformLocation(compute_direct_irradiance, "transmittance_texture"), 
            GL_TEXTURE0, GL_TEXTURE_2D, transmittance_texture_);
        glDisablei(GL_BLEND, 0);
        if (blend) { glEnablei(GL_BLEND, 1); } else { glDisablei(GL_BLEND, 1); }
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    assert(glGetError() == GL_NO_ERROR);

    // Compute the rayleigh and mie single scattering, store them in
    // delta_rayleigh_scattering_texture and delta_mie_scattering_texture, and
    // either store them or accumulate them in scattering_texture_ and
    // optional_single_mie_scattering_texture_.
    /* 3. Compute single-scattering(Rayleigh and Mie) */
    {
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_rayleigh_scattering_texture, 0);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, delta_mie_scattering_texture, 0);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, this->scattering_texture_, 0);
        if (glIsTexture(this->optional_single_mie_scattering_texture_)) {
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, this->optional_single_mie_scattering_texture_, 0);
            glDrawBuffers(4, kDrawBuffers);
        } else {
            glDrawBuffers(3, kDrawBuffers);
        }
        glViewport(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);
        
        GLint luminance_from_radiance_location = glGetUniformLocation(compute_single_scattering, "luminance_from_radiance");
        GLint transmittance_texture_location   = glGetUniformLocation(compute_single_scattering, "transmittance_texture");
        GLint layer_location                   = glGetUniformLocation(compute_single_scattering, "layer");

        glUseProgram(compute_single_scattering);
        glUniformMatrix3fv(luminance_from_radiance_location, 1, true, luminance_from_radiance.data());
        glUniformTexture(transmittance_texture_location, GL_TEXTURE0, GL_TEXTURE_2D, this->transmittance_texture_);
       
        for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
            glUniform1i(layer_location, layer);
            glDisablei(GL_BLEND, 0);
            glDisablei(GL_BLEND, 1);
            if (blend) { glEnablei(GL_BLEND, 2); } else { glDisablei(GL_BLEND, 2); }
            if (blend) { glEnablei(GL_BLEND, 3); } else { glDisablei(GL_BLEND, 3); }
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
    }
    assert(glGetError() == GL_NO_ERROR);

    /* 4. Compute high-order-scattering(2nd, 3rd, 4th, ...) */
    for (unsigned int scattering_order = 2; scattering_order <= num_scattering_orders; ++scattering_order) {
        // Compute the scattering density, and store it in
        // delta_scattering_density_texture.
        {
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_scattering_density_texture, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
            glViewport(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);

            GLint transmittance_texture_location              = glGetUniformLocation(compute_scattering_density, "transmittance_texture");
            GLint single_rayleigh_scattering_texture_location = glGetUniformLocation(compute_scattering_density, "single_rayleigh_scattering_texture");
            GLint single_mie_scattering_texture_location   = glGetUniformLocation(compute_scattering_density, "single_mie_scattering_texture");
            GLint multiple_scattering_texture_location  = glGetUniformLocation(compute_scattering_density, "multiple_scattering_texture");
            GLint irradiance_texture_location      = glGetUniformLocation(compute_scattering_density, "irradiance_texture");
            GLint scattering_order_location = glGetUniformLocation(compute_scattering_density, "scattering_order");
            GLint layer_location  = glGetUniformLocation(compute_scattering_density, "layer");

            glUseProgram(compute_scattering_density);
            glUniformTexture(transmittance_texture_location,              GL_TEXTURE0, GL_TEXTURE_2D, this->transmittance_texture_);
            glUniformTexture(single_rayleigh_scattering_texture_location, GL_TEXTURE1, GL_TEXTURE_3D, delta_rayleigh_scattering_texture);
            glUniformTexture(single_mie_scattering_texture_location,   GL_TEXTURE2, GL_TEXTURE_3D, delta_mie_scattering_texture);
            glUniformTexture(multiple_scattering_texture_location, GL_TEXTURE3, GL_TEXTURE_3D, delta_multiple_scattering_texture);
            glUniformTexture(irradiance_texture_location, GL_TEXTURE4, GL_TEXTURE_2D, delta_irradiance_texture);
            glUniform1i(scattering_order_location, scattering_order);
            
            for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
                glUniform1i(layer_location, layer);
                glDisablei(GL_BLEND, 0);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }
        }
        assert(glGetError() == GL_NO_ERROR);

        // Compute the indirect irradiance, store it in delta_irradiance_texture and
        // accumulate it in irradiance_texture_.
        {
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_irradiance_texture, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, this->irradiance_texture_, 0);
            glDrawBuffers(2, kDrawBuffers);
            glViewport(0, 0, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

            GLint luminance_from_radiance_location            = glGetUniformLocation(compute_indirect_irradiance, "luminance_from_radiance");
            GLint single_rayleigh_scattering_texture_location = glGetUniformLocation(compute_indirect_irradiance, "single_rayleigh_scattering_texture");
            GLint single_mie_scattering_texture_location      = glGetUniformLocation(compute_indirect_irradiance, "single_mie_scattering_texture");
            GLint multiple_scattering_texture_location        = glGetUniformLocation(compute_indirect_irradiance, "multiple_scattering_texture");
            GLint scattering_order_location                   = glGetUniformLocation(compute_indirect_irradiance, "scattering_order");

            glUseProgram(compute_indirect_irradiance);
            glUniformMatrix3fv(luminance_from_radiance_location, 1, true, luminance_from_radiance.data());
            glUniformTexture(single_rayleigh_scattering_texture_location, GL_TEXTURE0, GL_TEXTURE_3D, delta_rayleigh_scattering_texture);
            glUniformTexture(single_mie_scattering_texture_location,  GL_TEXTURE1, GL_TEXTURE_3D, delta_mie_scattering_texture);
            glUniformTexture(multiple_scattering_texture_location, GL_TEXTURE2, GL_TEXTURE_3D, delta_multiple_scattering_texture);
            glUniform1i(scattering_order_location, scattering_order - 1);

            glDisablei(GL_BLEND, 0);
            glEnablei(GL_BLEND, 1);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        assert(glGetError() == GL_NO_ERROR);

        // Compute the multiple scattering, store it in
        // delta_multiple_scattering_texture, and accumulate it in
        // scattering_texture_.
        {
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, delta_multiple_scattering_texture, 0);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, this->scattering_texture_, 0);
            glDrawBuffers(2, kDrawBuffers);
            glViewport(0, 0, SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT);

            GLint luminance_from_radiance_location    = glGetUniformLocation(compute_multiple_scattering, "luminance_from_radiance");
            GLint transmittance_texture_location      = glGetUniformLocation(compute_multiple_scattering, "transmittance_texture");
            GLint scattering_density_texture_location = glGetUniformLocation(compute_multiple_scattering, "scattering_density_texture");
            GLint layer_location                      = glGetUniformLocation(compute_multiple_scattering, "layer");

            glUseProgram(compute_multiple_scattering);
            glUniformMatrix3fv(luminance_from_radiance_location, 1, true, luminance_from_radiance.data());
            glUniformTexture(transmittance_texture_location, GL_TEXTURE0, GL_TEXTURE_2D, this->transmittance_texture_);
            glUniformTexture(scattering_density_texture_location, GL_TEXTURE1, GL_TEXTURE_3D, delta_scattering_density_texture);
            for (unsigned int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer) {
                glUniform1i(layer_location, layer);
                glDisablei(GL_BLEND, 0);
                glEnablei(GL_BLEND, 1);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }
        }
        assert(glGetError() == GL_NO_ERROR);
    }
   

    glDisablei(GL_BLEND, 0);
    glDisablei(GL_BLEND, 1);
    glDisablei(GL_BLEND, 2);
    glDisablei(GL_BLEND, 3);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, 0, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, 0, 0);
}

}  // namespace atmosphere
