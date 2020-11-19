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

/*<h2>atmosphere/constants.h</h2>

<p>This file defines the size of the precomputed texures used in our atmosphere
model. It also provides tabulated values of the <a href=
"https://en.wikipedia.org/wiki/CIE_1931_color_space#Color_matching_functions"
>CIE color matching functions</a> and the conversion matrix from the <a href=
"https://en.wikipedia.org/wiki/CIE_1931_color_space">XYZ</a> to the
<a href="https://en.wikipedia.org/wiki/SRGB">sRGB</a> color spaces (which are
needed to convert the spectral radiance samples computed by our algorithm to
sRGB luminance values).
*/

#ifndef ATMOSPHERE_CONSTANTS_H_
#define ATMOSPHERE_CONSTANTS_H_

namespace atmosphere {

    constexpr int TRANSMITTANCE_TEXTURE_WIDTH  = 256;
    constexpr int TRANSMITTANCE_TEXTURE_HEIGHT = 64;

    constexpr int SCATTERING_TEXTURE_R_SIZE    = 32;
    constexpr int SCATTERING_TEXTURE_MU_SIZE   = 128;
    constexpr int SCATTERING_TEXTURE_MU_S_SIZE = 32;
    constexpr int SCATTERING_TEXTURE_NU_SIZE   = 8;
    constexpr int SCATTERING_TEXTURE_WIDTH     = SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
    constexpr int SCATTERING_TEXTURE_HEIGHT    = SCATTERING_TEXTURE_MU_SIZE;
    constexpr int SCATTERING_TEXTURE_DEPTH     = SCATTERING_TEXTURE_R_SIZE;

    constexpr int IRRADIANCE_TEXTURE_WIDTH     = 64;
    constexpr int IRRADIANCE_TEXTURE_HEIGHT    = 16;

}  // namespace atmosphere

#endif  // ATMOSPHERE_CONSTANTS_H_
