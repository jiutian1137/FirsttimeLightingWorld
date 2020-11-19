#include "../../atmosphere/Atmosphere.glsl.h"
// sun-lighting
uniform sampler2D transmittance_texture;
uniform sampler2D irradiance_texture;
uniform vec3 sun_vector;
uniform vec3 camera;
// scene-shadow
uniform sampler2D scene_shadow_texture;
uniform sampler2D volume_transmittance_texture;
uniform mat4 world_to_sunproj;
uniform mat4 world_to_sunview;
uniform mat4 sunview_to_sunproj;
// scene
uniform sampler2D albedo_texture;

in vec3 fposition;
in vec2 ftexcoord;
in vec3 fnormal;
in vec3 ftangent;
in vec3 fbitangent;
in float fdepth;


void main(){
	vec3 Pss = texture(albedo_texture, ftexcoord).rgb;
	vec3 F0  = Pss * 0.2;
	float metallic = 0.05f;
	vec3 diffuse  = Pss / PI;
	vec3 specular = F0 + (1.0 - F0) * pow(1.0 - dot(normalize(camera - fposition), fnormal), 5.0);
	vec3 BRDF     = mix(diffuse, specular, metallic);

	vec3 earth_center = vec3(0.0, -ATMOSPHERE.bottom_radius, 0.0);
	vec3 sky_irradiance;
	vec3 sun_irradiance = GetSunAndSkyIrradiance(ATMOSPHERE, transmittance_texture, irradiance_texture, fposition - earth_center, fnormal, sun_vector, sky_irradiance);
	
	vec3 position_inlightview = (world_to_sunview * vec4(fposition, 1)).xyz;
	vec3 shadowUV_dFdx = (mat3(sunview_to_sunproj) * dFdx(position_inlightview)) * vec3(0.5, 0.5, 1);
	vec3 shadowUV_dFdy = (mat3(sunview_to_sunproj) * dFdy(position_inlightview)) * vec3(0.5, 0.5, 1);
	vec2 shadowTexsize = textureSize(scene_shadow_texture, 0);
	vec2 bias = vec2( shadowUV_dFdy.y * shadowUV_dFdx.z - shadowUV_dFdx.y * shadowUV_dFdy.z,
					 -shadowUV_dFdy.x * shadowUV_dFdx.z + shadowUV_dFdx.x * shadowUV_dFdy.z);
	float bias_det = shadowUV_dFdx.x * shadowUV_dFdy.y - shadowUV_dFdx.y * shadowUV_dFdy.x;
	bias /= sign(bias_det) * max(abs(bias_det), 1e-8);

	float error = dot(vec2(1,1), abs(bias / shadowTexsize));
	vec4  position_inlight = world_to_sunproj * vec4(fposition, 1);
	vec2  texcoord_inlight = (position_inlight.xy / position_inlight.w) * 0.5 + 0.5;
	float depth_inlight    = position_inlight.z / position_inlight.w - (20.0 / 25000.0);
	float dx = 1.0 / shadowTexsize.x;
	float light_visibility = /*depth_inlight <= texture(scene_shadow_texture, texcoord_inlight).r*/true 
								? texture(volume_transmittance_texture, texcoord_inlight).r : 0.0;

		  light_visibility += /*depth_inlight <= texture(scene_shadow_texture, texcoord_inlight + vec2(-dx, -dx)).r*/true
								? texture(volume_transmittance_texture, texcoord_inlight + vec2(-dx, -dx)).r : 0.0;
		  
		  light_visibility += /*depth_inlight <= texture(scene_shadow_texture, texcoord_inlight + vec2(+dx, -dx)).r */true
								? texture(volume_transmittance_texture, texcoord_inlight + vec2(+dx, -dx)).r : 0.0;

		  light_visibility += /*depth_inlight <= texture(scene_shadow_texture, texcoord_inlight + vec2(-dx, +dx)).r*/true
								? texture(volume_transmittance_texture, texcoord_inlight + vec2(-dx, +dx)).r : 0.0;

		  light_visibility += /*depth_inlight <= texture(scene_shadow_texture, texcoord_inlight + vec2(+dx, +dx)).r*/true
								? texture(volume_transmittance_texture, texcoord_inlight + vec2(+dx, +dx)).r : 0.0;
		  light_visibility /= 5.0;

	gl_FragColor.rgb = (sky_irradiance + sun_irradiance * clamp(light_visibility, 0, 1)) * (1.0 / PI) * PI * BRDF;

	gl_FragDepth = fdepth;
}