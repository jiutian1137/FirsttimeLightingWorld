#include "Atmosphere.glsl.h"
layout(location = 0) out vec3 transmittance;
void main() {
    transmittance = ComputeTransmittanceToTopAtmosphereBoundaryTexture(
        ATMOSPHERE, gl_FragCoord.xy);
}