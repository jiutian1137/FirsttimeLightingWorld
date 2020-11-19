#include "glsl.h"

#define IN(x) const in x
#define OUT(x) out x
#define TEMPLATE(x)
#define TEMPLATE_ARGUMENT(x)
#define assert(x)

#define TRANSMITTANCE_TEXTURE_WIDTH  256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define SCATTERING_TEXTURE_R_SIZE    32
#define SCATTERING_TEXTURE_MU_SIZE   128
#define SCATTERING_TEXTURE_MU_S_SIZE 32
#define SCATTERING_TEXTURE_NU_SIZE   8
#define IRRADIANCE_TEXTURE_WIDTH     64
#define IRRADIANCE_TEXTURE_HEIGHT    16

#define COMBINED_SCATTERING_TEXTURES 1

#include "definitions.glsl.h"
#include "functions.glsl.h"

uniform AtmosphereParameters ATMOSPHERE = AtmosphereParameters(
    vec3(1.474000, 1.850400, 1.911980),
    0.004675,
    6360000.000000,
    6420000.000000,
    DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000, 0.000000, 0.000000, 0.000000, 0.000000), 
                                            DensityProfileLayer(0.000000, 1.000000, -0.000125, 0.000000, 0.000000))),
    vec3(0.000006, 0.000014, 0.000033),
    DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000, 0.000000, 0.000000, 0.000000, 0.000000), 
                                            DensityProfileLayer(0.000000, 1.000000, -0.000833, 0.000000, 0.000000))),
    vec3(0.000004, 0.000004, 0.000004),
    vec3(0.000004, 0.000004, 0.000004),
    0.800000,
    DensityProfile(DensityProfileLayer[2](DensityProfileLayer(25000.000000, 0.000000, 0.000000, 0.000067, -0.666667), 
                                            DensityProfileLayer(0.000000, 0.000000, 0.000000, -0.000067, 2.666667))),
    vec3(0.000001, 0.000002, 0.000000),
    vec3(0.100000, 0.100000, 0.100000),
    -0.500000 );
#define SKY_SPECTRAL_RADIANCE_TO_LUMINANCE vec3(0.0)
#define SUN_SPECTRAL_RADIANCE_TO_LUMINANCE vec3(0.0)