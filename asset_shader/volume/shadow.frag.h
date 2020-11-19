#include "volumeX.glsl.h"

// system property
uniform ivec2 resolution;
uniform float time; // [h]
uniform float time_per_day;// [h/d]

// cloud texture
uniform sampler3D base_cloud_texture; uniform float base_cloud_texture_scale = 0.0016;
uniform sampler3D cloud_detail_texture; uniform float cloud_detail_texture_scale = 0.0016;
uniform sampler2D cloud_weather_texture_first;
uniform sampler2D cloud_weather_texture_second; uniform float cloud_weahter_texture_scale = 50.0;
uniform sampler2D cloud_height_texture;
uniform sampler2D cirrus_weather_texture;
uniform sampler2D cloud_curl_texture;
// cloud property
uniform float transmit_determinant = 0.01;
uniform float cloud_forward_scattering = 0.80f;
uniform float cloud_backward_scattering = 0.45f;
uniform CloudProfile CLOUDSPHERE0 = CloudProfile(1000.0, 4000.0, 0.1);
// sky property
uniform vec3 sun_vector;
// scene property
uniform sampler2D scene_shadow_texture;
uniform mat4 sunproj_to_world;
uniform mat4 world_to_proj;

out float out_transmittance;
// out float gl_FragDepth


Number GetVolumesDensity(Position point, Position earth_center) {
	Number percipitation1;
	Number density = GetCloudDensity(CLOUDSPHERE0, base_cloud_texture, cloud_detail_texture, base_cloud_texture_scale, 8.0, 0.2, 2.0, vec3(1, 0, 0) * time, 2,
		cloud_weather_texture_first, cloud_weahter_texture_scale, cloud_height_texture, point, earth_center, percipitation1);

	Number percipitation2;
	density += GetCloudDensity(CLOUDSPHERE0, base_cloud_texture, cloud_detail_texture, base_cloud_texture_scale, 8.0, 0.2, 2.0, vec3(1, 0, 0) * time, 2,
		cloud_weather_texture_second, cloud_weahter_texture_scale, cloud_height_texture, point, earth_center, percipitation2);

	return density;
}

void main() {
	// 1. Constants
	vec3  earth_center = vec3(0, -ATMOSPHERE.bottom_radius, 0);
	vec2  texcoord    = gl_FragCoord.xy / vec2(resolution);
	float scene_depth = texture(scene_shadow_texture, texcoord).r;

	vec4 ray_start = sunproj_to_world * vec4((texcoord - 0.5) * 2.0, 0.0, 1.0);
	vec4 ray_end = sunproj_to_world * vec4((texcoord - 0.5) * 2.0, scene_depth, 1.0);
	ray_start /= ray_start.w;
	ray_end /= ray_end.w;

	int step = 64;
	Length    ds = length(ray_end.xyz - ray_start.xyz) / float(step);
	Direction Ds = normalize(ray_end.xyz - ray_start.xyz);

	// 2. Integrate (cloud_sphere_top to scene)
	Number transmittance = 1.0;
	bool    intersect    = false;
	Position intersect_point;
	for (int i = 0; i != step; ++i) {
		Position point   = ray_start.xyz + Ds * (i * ds);
		Number   density = GetVolumesDensity(point, earth_center);
		transmittance   *= exp(-density * ds);
		
		if (!intersect && transmittance < 0.9) {
			intersect_point = point;
			intersect = true;
		}
	}

	// 3. Output result
	out_transmittance = transmittance;// [1.0, 1e-38]
	if (intersect) {
		vec4 temp = world_to_proj * vec4(intersect_point, 1);
		gl_FragDepth = temp.z / temp.w;
	} else {
		gl_FragDepth = 1.0;
	}
}