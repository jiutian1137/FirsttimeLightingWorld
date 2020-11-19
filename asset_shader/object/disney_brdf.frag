#version 330 core
uniform vec3  base_color     = vec3(0.7f);
uniform float metallic       = 0.0f;
uniform vec3  emission       =  vec3(0.0f);
uniform float subsurface     = 0.0f;
uniform float specular       = 0.5f;
uniform float roughness      = 0.5f;
uniform float specular_tint  = 0.0f;
uniform float anisotropic    = 0.0f;
uniform float sheen          = 0.0f;
uniform float sheen_tint     = 0.5f;
uniform float clearcoat      = 0.0f;
uniform float clearcoat_gloss= 1.0f;
uniform bool use_base_color_texture = false;
uniform bool use_metallic_texture   = false;
uniform bool use_emission_texture   = false;
uniform bool use_roughness_texture  = false;
uniform sampler2D base_color_texture;
uniform sampler2D metallic_texture;
uniform sampler2D emission_texture;
uniform sampler2D roughness_texture;
//
uniform bool use_normal_texture = false;
uniform sampler2D normal_texture;
uniform sampler2D occlusion_texture;
// light
uniform vec3      camera;
uniform vec3      sun_direction = vec3(0, 1, 0);
//uniform sampler2D sky_irradiance_texture;

in vec3 vout_position;
in vec3 vout_normal;
in vec2 vout_texcoord;
in vec3 vout_tangent;
in vec3 vout_bitangent;
in float vout_depth;
layout(location = 0) out vec3 radiance_rate;
layout(location = 1) out vec3 fout_normal;


const float PI = 3.14159265358979323846;

float sqr(float x) { return x*x; }

float SchlickFresnel(float u) {
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

float GTR1(float NdotH, float a) {
    if (a >= 1) return 1/PI;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (PI*log(a2)*t);
}

float GTR2(float NdotH, float a) {
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return a2 / (PI * t*t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay) {
    return 1 / (PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ));
}

float smithG_GGX(float NdotV, float alphaG) {
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay) {
    return 1 / (NdotV + sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ));
}

vec3 mon2lin(vec3 x) {
    return vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
}


vec3 disney_brdf(vec3 position, vec3 normal, vec3 tangent, vec3 bitangent, vec3 in_vector, vec3 out_vector, 
                 vec3 base_color, float metallic, float subsurface, float specular, float roughness, float specular_tint, float anisotropic,
                 float sheen, float sheen_tint, float clearcoat, float clearcoat_gloss){
    float NdotL = dot(normal, in_vector);
    float NdotV = dot(normal, out_vector);
    if (NdotL < 0 || NdotV < 0) return vec3(0);

    vec3  H = normalize(in_vector + out_vector);
    float NdotH = dot(normal, H);
    float LdotH = dot(in_vector, H);

    vec3  Cdlin = mon2lin( base_color );
    float Cdlum = 0.3*Cdlin[0] + 0.6*Cdlin[1]  + 0.1*Cdlin[2]; // luminance approx.

    vec3 Ctint  = Cdlum > 0 ? Cdlin/Cdlum : vec3(1.0); // normalize lum. to isolate hue+sat
    vec3 Cspec0 = mix(specular * 0.08 * mix(vec3(1.0), Ctint, specular_tint), Cdlin, metallic);
    vec3 Csheen = mix(vec3(1.0), Ctint, sheen_tint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL   = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5 + 2 * LdotH*LdotH * roughness;
    float Fd   = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH*LdotH*roughness;
    float Fss   = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
    float ss    = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);

    // specular
    float aspect = sqrt(1 - anisotropic*0.9);
    float ax = max(.001, sqr(roughness)/aspect);
    float ay = max(.001, sqr(roughness)*aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, tangent), dot(H, bitangent), ax, ay);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0, vec3(1.0), FH);
    float Gs = 1.0;
    Gs  = smithG_GGX_aniso(NdotL, dot(in_vector, tangent), dot(in_vector, bitangent), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(out_vector, tangent), dot(out_vector, bitangent), ax, ay);

    // sheen
    vec3 Fsheen = FH * sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, mix(0.1, 0.001, clearcoat_gloss));
    float Fr = mix(0.04, 1.0, FH);
    float Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25);

    return ((1.0/PI) * mix(Fd, ss, subsurface)*Cdlin + Fsheen)
        * (1.0 - metallic)
        + Gs*Fs*Ds + 0.25*clearcoat*Gr*Fr*Dr;
}

void main() {
    vec3 normal = use_normal_texture
        ? (mat3(vout_tangent, vout_bitangent, vout_normal) * normalize(texture(normal_texture, vout_texcoord).xyz * 2.0 - 1.0f))
        : vout_normal;

    radiance_rate = disney_brdf(vout_position, normal, vout_tangent, vout_bitangent, sun_direction, normalize(camera - vout_position),
                use_base_color_texture ? texture(base_color_texture, vout_texcoord).rgb : base_color, 
                (use_metallic_texture   ? texture(metallic_texture, vout_texcoord).r : metallic),
                subsurface,
                specular,
                use_roughness_texture  ? texture(roughness_texture, vout_texcoord).r : roughness,
                specular_tint,
                anisotropic,
                sheen,
                sheen_tint,
                clearcoat,
                clearcoat_gloss)
                 * (use_normal_texture ? texture(occlusion_texture, vout_texcoord).r : 1.0)
                 + (use_emission_texture ? texture(emission_texture, vout_texcoord).rgb : emission);
      fout_normal = normal * 0.5f + 0.5f;

      gl_FragDepth = vout_depth;
//    gl_FragColor.rgb = (use_base_color_texture ? texture(base_color_texture, vout_texcoord).rgb : base_color);
//    gl_FragColor.rgb = use_normal_texture ? normalize(texture(normal_texture, vout_texcoord).xyz  * 2.0 - 1.0f) : vout_normal;
}
