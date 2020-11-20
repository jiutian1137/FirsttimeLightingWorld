#include "../atmosphere/Atmosphere.glsl.h"
uniform ivec2 resolution;

uniform sampler2D transmittance_texture;
uniform sampler3D scattering_texture;
uniform sampler3D XXXXX;
uniform vec3 sun_vector;

uniform sampler2D scene_radiance_texture;
uniform sampler2D scene_depth_texture;
uniform sampler2D volume_radiance_and_transmittance_texture;
uniform sampler2D volume_depth_texture;
uniform mat4 proj_to_world;

uniform sampler2D scene_shadow_texture;
uniform sampler2D volume_shadow_texture;
uniform sampler2D volume_shadow_transmittance_texture;
uniform mat4 world_to_sunproj;

//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilm(vec3 x) {
	float a = 2.51;
	float b = 0.03;
	float c = 2.43;
	float d = 0.59;
	float e = 0.14;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}


#define texelOffset vec2(1.75 / resolution.xy)

const float kernel[9] = float[] (
	.0625, .125, .0625,
    .125,  .25,  .125,
    .0625, .125, .0625  
);

vec4 gaussianBlur(sampler2D buffer, vec2 uv)
{
    vec4 col = vec4(0.);
    
 	vec2 offsets[9] = vec2[](
        vec2(-texelOffset.x,  texelOffset.y),  // top-left
        vec2( 			0.,   texelOffset.y),  // top-center
        vec2( texelOffset.x,  texelOffset.y),  // top-right
        vec2(-texelOffset.x,  			 0.),  // center-left
        vec2( 			0.,			 	 0.),  // center-center
        vec2( texelOffset.x,  	 		 0.),  // center-right
        vec2(-texelOffset.x,  -texelOffset.y), // bottom-left
        vec2( 			0.,   -texelOffset.y), // bottom-center
        vec2( texelOffset.x,  -texelOffset.y)  // bottom-right    
    );
    
    for(int i = 0; i < 9; i++) {
        col += textureLod(buffer, uv + offsets[i], 0.) * kernel[i];
    }
    
    return col;
}


/*  volume_depth <= scene_depth <= background_depth(constant:1.0)

	shadow_length: from near to far
		if (intersect_volume) {
			ray_start to ray_volume to lerp(ray_end, ray_scene, intersect_scene)
		} else {
			ray_start to lerp(ray_end, ray_scene, intersect_scene)
		}
*/

Length IntegrateShadowLength(vec3 ray_start, vec3 ray_end) {
	// 1. Setup constants
	vec4 shadow_ray_start    = world_to_sunproj * vec4(ray_start, 1.0);
	     shadow_ray_start   /= shadow_ray_start.w;
	     shadow_ray_start.xy = shadow_ray_start.xy * 0.5 + 0.5;
	vec4 shadow_ray_end      = world_to_sunproj * vec4(ray_end, 1.0);
	     shadow_ray_end     /= shadow_ray_end.w;// { x:[-1,+1], y:[-1,+1], z:[0,1] }
	     shadow_ray_end.xy   = shadow_ray_end.xy * 0.5 + 0.5;// { x:[0,1], y:[0,1], z:[0,1] }

	Direction Ds = shadow_ray_end.xyz - shadow_ray_start.xyz;
	Length S = length(Ds);
	Ds /= S;

	ivec2 texture_size = (textureSize(scene_shadow_texture, 0) + textureSize(volume_shadow_texture, 0)) / 2;
	float texcoord_range = 1.0;
	vec2   dxy = texcoord_range / vec2(texture_size);
	Length ds  = length(dxy);

	// 2. Integrate shadow_length in TextureSpace
	int step = clamp(int(round(S / ds)), 0, 128);
	    ds   = S / float(step);
	Length shadow_length = 0.0;
	for (int i = 0; i != step; ++i) {
		vec3 point    = shadow_ray_start.xyz + Ds * (i * ds);
		bool scene_shadowed  = (point.z - 3e-5) > texture(scene_shadow_texture, point.xy).r;
		bool volume_shadowed = (point.z - 3e-5) > texture(volume_shadow_texture, point.xy).r;

		shadow_length += (scene_shadowed ? 
			float(scene_shadowed) : 
			(volume_shadowed ? 
				1.0 - texture(volume_shadow_transmittance_texture, point.xy).r :
				0.0)
			);
	}

	return shadow_length * (length(ray_end - ray_start) / step);
}

void main(){
	Position earth_center = vec3(0, -ATMOSPHERE.bottom_radius, 0);
	vec2   texcoord    = gl_FragCoord.xy / vec2(resolution);
	Number scene_depth = texture(scene_depth_texture, texcoord).r;
	Number volume_depth = texture(volume_depth_texture, texcoord).r;
	bool intersect_scene = (scene_depth != 1.0);
	bool intersect_volume = (volume_depth != 1.0);
	
	vec4 ray_start  = proj_to_world * vec4((texcoord - 0.5) * 2.0, 0.0, 1.0);
	vec4 ray_end    = proj_to_world * vec4((texcoord - 0.5) * 2.0, 1.0, 1.0);
	vec4 ray_scene  = proj_to_world * vec4((texcoord - 0.5) * 2.0, scene_depth, 1.0);
	vec4 ray_volume = proj_to_world * vec4((texcoord - 0.5) * 2.0, volume_depth, 1.0);
	ray_start  /= ray_start.w;
	ray_end    /= ray_end.w;
	ray_scene  /= ray_scene.w;
	ray_volume /= ray_volume.w;
	vec3 ray_direction = normalize(ray_end.xyz - ray_start.xyz);


	RadianceSpectrum radiance = vec3(0.0);
	DimensionlessSpectrum transmittance = vec3(1.0);

	if (intersect_volume) {

		Length start_to_volume_shadowlength = IntegrateShadowLength(ray_start.xyz, ray_volume.xyz);

		DimensionlessSpectrum start_to_volume_skyscatter_transmittance;
		RadianceSpectrum start_to_volume_skyscatter_radiance = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, XXXXX,
			ray_start.xyz - earth_center, ray_volume.xyz - earth_center, start_to_volume_shadowlength, sun_vector, start_to_volume_skyscatter_transmittance);
		radiance      += transmittance * start_to_volume_skyscatter_radiance;
		transmittance *= start_to_volume_skyscatter_transmittance;
		
		RadianceSpectrum atmosphere_transmittanced_volume_radiance = texture(volume_radiance_and_transmittance_texture, texcoord).rgb;
		//RadianceSpectrum atmosphere_transmittanced_volume_radiance = gaussianBlur(volume_radiance_and_transmittance_texture, texcoord).rgb;
		Number volume_transmittance = texture(volume_radiance_and_transmittance_texture, texcoord).a;
		radiance      += atmosphere_transmittanced_volume_radiance;
		transmittance *= volume_transmittance;

		if (intersect_scene) {
			Length volume_to_scene_shadowlength = IntegrateShadowLength(ray_volume.xyz, ray_scene.xyz);

			DimensionlessSpectrum volume_to_scene_skyscatter_transmittance;
			RadianceSpectrum volume_to_scene_skyscatter_radiance = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, XXXXX,
				ray_volume.xyz-earth_center, ray_scene.xyz-earth_center, volume_to_scene_shadowlength, sun_vector, volume_to_scene_skyscatter_transmittance);
			radiance      += transmittance * volume_to_scene_skyscatter_radiance;
			transmittance *= volume_to_scene_skyscatter_transmittance;
			radiance      += transmittance * texture(scene_radiance_texture, texcoord).rgb;
			transmittance = vec3(0.0);
		} else {
			Length volume_to_background_shadowlength = IntegrateShadowLength(ray_volume.xyz, ray_end.xyz);
			
			DimensionlessSpectrum volume_to_background_skyscatter_transmittance;
			RadianceSpectrum volume_to_background_skyscatter_radiance = GetSkyRadiance(ATMOSPHERE, transmittance_texture, scattering_texture, XXXXX,
				ray_volume.xyz - earth_center, ray_direction, volume_to_background_shadowlength, sun_vector, volume_to_background_skyscatter_transmittance);
			radiance      += transmittance * volume_to_background_skyscatter_radiance;
			transmittance *= volume_to_background_skyscatter_transmittance;
		}

	} else if (intersect_scene) {

		Length start_to_scene_shadowlength = IntegrateShadowLength(ray_start.xyz, ray_scene.xyz);

		DimensionlessSpectrum start_to_scene_skyscatter_transmittance;
		RadianceSpectrum start_to_scene_skyscatter_radiance = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, XXXXX,
			ray_start.xyz-earth_center, ray_scene.xyz-earth_center, start_to_scene_shadowlength, sun_vector, start_to_scene_skyscatter_transmittance);
		radiance      += transmittance * start_to_scene_skyscatter_radiance;
		transmittance *= start_to_scene_skyscatter_transmittance;
		radiance      += transmittance * texture(scene_radiance_texture, texcoord).rgb;
		transmittance = vec3(0.0);

	} else {

		Length start_to_background_shadowlength = IntegrateShadowLength(ray_start.xyz, ray_end.xyz);

		DimensionlessSpectrum start_to_background_skyscatter_transmittance;
		RadianceSpectrum start_to_background_skyscatter_radiance = GetSkyRadiance(ATMOSPHERE, transmittance_texture, scattering_texture, XXXXX,
			ray_start.xyz-earth_center, ray_direction, start_to_background_shadowlength, sun_vector, start_to_background_skyscatter_transmittance);
		radiance      += transmittance * start_to_background_skyscatter_radiance;
		transmittance *= start_to_background_skyscatter_transmittance;

	}

	
	gl_FragColor.rgb = 1.0 - exp(-radiance * 10.0f);
	//gl_FragColor.rgb = ACESFilm(radiance * 10.0f);
	gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(1.0 / 2.2));
	
	//gl_FragColor.rgb = texture(volume_shadow_texture, texcoord).rgb;
	//gl_FragColor = texture(scene_depth_texture, texcoord);
	//gl_FragColor = texture(transmittance_texture, texcoord);
}