#include "glsl.h"
#include "../EricBruneton/functions.glsl"
#include "collision.h"


// atmosphere
uniform sampler2D transmittance_texture;
uniform sampler3D scattering_texture;
uniform sampler2D irradiance_texture;
uniform vec3 sun_vector;
uniform vec2 sun_size;
// cloud
uniform sampler2D cloud_height_texture;
uniform sampler2D cloud_weather_texture; uniform float cloud_weather_texture_scale = 500.0f;// 500 chunk per earth
uniform sampler3D base_cloud_texture;    uniform float base_cloud_texture_scale    = 0.00008f;
uniform sampler3D cloud_detail_texture;  uniform float cloud_detail_texture_scale  = 0.0004f;
uniform sampler2D cloud_curl_texture;
uniform sampler2D cirrus_cloud_texture;  uniform float cirrus_cloud_texture_scale = 500.0f;
uniform float cloud_transmittance_lowbound = 0.01f;// near to _ZERO
uniform float cloud_density_scale       = 1.0f;
uniform float cloud_forward_scattering  = 0.80f;
uniform float cloud_backward_scattering = 0.45f;
uniform float cloud_precipitation_scale = 1.0f;
uniform vec3  wind_direction = vec3(1.0f, 0.0f, 1.0f);
uniform float wind_intensity = 100.0f;
// scene
uniform sampler2D scene_brdf_texture;
uniform sampler2D scene_normal_texture;
uniform sampler2D scene_depth_texture;
uniform sampler2D scene_shadow_texture;

uniform float exposure = 8.0F;
uniform vec3  white_point = vec3(1.08241f, 0.967556f, 0.95003f);
uniform vec3  earth_center = vec3(0, -6360000, 0);
uniform mat4  proj_to_world;
uniform mat4  world_to_Lproj;
uniform float time = 0.0f;

uniform sampler3D single_mie_scattering_texture;


in vec2 texcoord;

//uniform vec3 base_noise_ratios  = vec3(0.625f, 0.25f, 0.125f);
//uniform vec3 detail_noise_ratio = vec3(0.625f, 0.25f, 0.125f);

//uniform vec3 sun_tint = vec3(1.0f);
//uniform float sun_gain = 0.25f;
//uniform vec3 atmosphere_bottom_tint = vec3(0.55f, 0.775f, 1.0f);
//uniform vec3 atmosphere_top_tint = vec3(0.45f, 0.675f, 1.0f);

RadianceSpectrum GetSolarRadiance() {                        
    return ATMOSPHERE.solar_irradiance /
        (PI * ATMOSPHERE.sun_angular_radius * ATMOSPHERE.sun_angular_radius);
}

float remap(float old_val, float old_min, float old_max, float new_min, float new_max){
	return clamp( (old_val - old_min)/(old_max-old_min)*(new_max-new_min) + new_min, new_min, new_max );// clamp can slightly avoid float-error
}

float rescale(float vMin, float vMax, float v){
	return clamp((v - vMin) / (vMin - vMax), vMin, vMax);
}

mat2 rotation(float a) {
	return mat2(cos(a), -sin(a),
				sin(a),  cos(a));
}

const float ground_height       = 0.0;
const float cloud_bottom_height = 1500.0;
const float cloud_top_height    = 6000.0;
const float cloud_thickness     = 4500.0;
const float atmosphere_top_height = 60000.0f;

float HenyeyGreenstein(float dot_angle, float scattering_value) {
	float squared_scattering_value = pow(scattering_value, 2.0f);

	return (1.0 - squared_scattering_value) / (4.0f * PI * pow(squared_scattering_value - (2.0f * scattering_value * dot_angle) + 1.0f, 1.5f));
}

float DoblobeHenyeyGreenstein(float dot_angle, float scattering_value0, float scattering_value1, float t) {
	return lerp(HenyeyGreenstein(dot_angle, scattering_value0), HenyeyGreenstein(dot_angle, scattering_value1), t);
}

float exponential_transmittance(float optical_depth) {
	return exp(-optical_depth);
}

float powdersugar_transmittance(float optical_depth) {
	return 1.0 - exp(-(optical_depth * 2.0));
}

float compute_height_gradient(float relativeHeight, float cloudType) {
    float cumulus = max(0.0f, remap(relativeHeight, 0.01, 0.3, 0.0, 1.0) * remap(relativeHeight, 0.6, 0.95, 1.0, 0.0));
	float stratocumulus = max(0.0f, remap(relativeHeight, 0.0, 0.25, 0.0, 1.0) * remap(relativeHeight,  0.3, 0.65, 1.0, 0.0)); 
	float stratus = max(0.0f, remap(relativeHeight, 0, 0.1, 0.0, 1.0) * remap(relativeHeight, 0.2, 0.3, 1.0, 0.0)); 
	//return cumulus;

	float a = lerp(stratus, stratocumulus, clamp(cloudType * 2.0f, 0.0f, 1.0f));

    float b = lerp(stratocumulus, stratus, clamp((cloudType - 0.5) * 2.0, 0.0, 1.0));
    return smoothstep(a, b, cloudType);
}

float compute_altitude(vec3 pos){
	return length(pos - earth_center) - 6360000.f;
}

float compute_height_ratio(vec3 pos) {// not saturate, because out of cloud layer
	return (compute_altitude(pos) - cloud_bottom_height) / cloud_thickness;
}

vec3 compute_cloud_coordinate(vec3 world_position) {
	float r       = length(world_position);
	float azimuth = atan2(world_position.y, world_position.x); if(azimuth < 0.0f) { azimuth += 2.0f*PI; }
	float zenith  = acos( clamp(world_position.z / r, -1.0f, 1.0f) );
	float height_ratio = (r - 6360000.f - cloud_bottom_height) / cloud_thickness;
	//return vec3(azimuth/PI*0.5+0.5, (zenith/PI)*0.5, height_ratio);// { [0,1], [0,1], [0,1] }
	return vec3(azimuth/PI, (zenith/PI), height_ratio);// { [0,1], [0,1], [0,1] }
}

float sample_cloud(vec3 ray_position, out float precipitation) {
	///*mat2 T = rotation(t * (3.14f / 180.0f) * ((length(ray_position.xz)/100000.0f) * 10.0f));
	//ray_position.xz = T * ray_position.xz;*/

	 //1. wind
	ray_position += time * wind_direction * wind_intensity;
	ray_position += compute_height_ratio(ray_position) * wind_direction * wind_intensity;

	// 2. sample weather
	vec3 cloud_position = compute_cloud_coordinate(ray_position - earth_center);
	vec3 weather_data   = texture(cloud_weather_texture, cloud_position.xy * cloud_weather_texture_scale + 0.5f).xyz;
	float height_ratio = cloud_position.z;
	//float coverage     = weather_data.r;
	//float type         = weather_data.g;
	float coverage = (weather_data.r != 0.0 && weather_data.g != 0.0) ? (max(weather_data.r, weather_data.g)) : (weather_data.r != 0.0 ? weather_data.r : weather_data.g);
	float type     = (weather_data.r != 0.0 && weather_data.g != 0.0) ? (6.0 / 32.0) : (weather_data.r != 0.0 ? weather_data.r : (18.0/32.0));
	precipitation  = weather_data.b;
	float height_gradient = texture(cloud_height_texture, vec2(type, height_ratio)).r;
	
	if (0.0f <= height_ratio && height_ratio <= 1.0f && coverage != 0.0f && height_gradient != 0.0f) {
		// 3. sample base cloud
		float base_noise = texture(base_cloud_texture, ray_position * base_cloud_texture_scale).r;

		// 4. sample height multiplier
		float base_erosion = remap(base_noise, 1.0f - coverage, 1.0f, 0.0f, 1.0f) * coverage * height_gradient;
	 
		if (base_erosion > 0.01f) {
			// 5. sample cloud_details
			//ray_position.xz += texture(cloud_curl_texture, ray_position.xz).xz;
			//ray_position.xy += texture(cloud_curl_texture, ray_position.xy).xy;
			float detail_noise = texture(cloud_detail_texture, ray_position * cloud_detail_texture_scale).r;
			detail_noise = lerp(detail_noise, 1.0f - detail_noise, saturate(height_ratio*10.0f));

			return remap(base_erosion, detail_noise * 0.4f, 1.0, 0.0, 1.0) * max(height_ratio, 0.001f) * cloud_density_scale;
		}
	}

	return 0.0f;
}

float sample_light_transmittance(vec3 r_origin, vec3 r_direction, float step_length, out float optical_depth) {
	optical_depth = 0.0f;
	float transmittance = 1.0f;

	for (int i = 0; i != 6; ++i) {
		vec3 x = r_origin + r_direction * (float(i + 1) * step_length);
		float sampled_precipitation;
		float sampled_density       = sample_cloud(x, sampled_precipitation);
		float sampled_optical_depth = sampled_density * step_length;
		optical_depth += sampled_optical_depth;
		transmittance *= exponential_transmittance(sampled_optical_depth * sampled_precipitation * cloud_precipitation_scale);
	}

	// ignore cirrus
	return transmittance;
}

vec3 raymarch_cloud(vec3 r_origin, vec3 r_direction, float r_start, float r_end, out float transmittance){
	if (r_start > 100000.0) {
		transmittance = 1.0f;
		return vec3(0.0f, 0.0f, 0.0f);
	}

	int   num_step           = 64;
	float step_length        = 300.0f;
	float detail_step_length = 50.0f;

	// 2. integrate
	float cos_angle              = dot(-r_direction, sun_vector);
	float light_scattering_phase = clamp(DoblobeHenyeyGreenstein(cos_angle, cloud_forward_scattering, cloud_backward_scattering, 0.5), 1.0f, 2.5f);
	transmittance = 1.0f;
	vec3 radiance = vec3(0.0f, 0.0f, 0.0f);

	float s = step_length; float test = 0.0; int zero_density_count = 0;
	for (int i = 0; i != num_step; ++i) {
		vec3  x  = r_origin + r_direction * (r_start + s);
		float ds = (test != 0.0) ? detail_step_length : step_length;
		s += ds;

		if (test != 0.0) {
			float precipitation;
			float sampled_cloud = sample_cloud(x, precipitation);

			if (sampled_cloud != 0.0) { 
				zero_density_count = 0;

				float sampled_optical_depth = sampled_cloud * ds;
				float sampled_transmittance = exponential_transmittance(sampled_optical_depth);
				float sampled_sun_optical_depth;
				float sampled_sun_transmittance = sample_light_transmittance(x, sun_vector, (cloud_top_height - compute_altitude(x)) / 6.0f, sampled_sun_optical_depth);

				vec3 point = x - earth_center;
				Length r    = length(point);
				Number mu_s = dot(point, sun_vector) / r;

				// Indirect
				vec3 sky_radiance       = GetIrradiance(ATMOSPHERE, irradiance_texture, r, mu_s) * (1.0 / PI);
				// Direct
				vec3  sun_radiance      = ATMOSPHERE.solar_irradiance * GetTransmittanceToSun(ATMOSPHERE, transmittance_texture, r, mu_s) * (1.0 / PI);
				float sun_transmittance = sampled_sun_transmittance * powdersugar_transmittance(sampled_sun_optical_depth);
				float sun_visibility    = 1.0f;

				// <approximate>
				//vec3 sky_radiance = lerp(atmosphere_bottom_tint, atmosphere_top_tint, get_height_ratio(x)) * sun_gain * sun_angle_multiplier;
				//vec3 sun_radiance = sun_tint * sun_gain * sun_angle_multiplier;
				// </approximate>
				vec3 scattered_radiance = (sky_radiance + sun_radiance * sun_transmittance * sun_visibility) * light_scattering_phase;
				//vec3 scattered_radiance = ((sky_radiance + sun_radiance) * sun_transmittance * sun_visibility) * light_scattering_phase;
				radiance      += scattered_radiance * sampled_cloud * transmittance * ds;
				transmittance *= sampled_transmittance;
			
				if (transmittance < cloud_transmittance_lowbound) {
					transmittance = 0.0001f;
					return radiance;
				}

			} else {
				++zero_density_count;
				if (zero_density_count == 6) {
					test = 0.0;
					zero_density_count = 0;
				}
			}
		} else {
			float temp;
			test = sample_cloud(x, temp);
			if (test != 0.0) {
				s -= ds;
			}
		}
	}

	return radiance;
}

vec3 raymarch_cirrus_cloud(vec3 r_origin, vec3 r_direction, vec3 cld_center, float cld_radius, int cld_index, out float transmittance) {
	float t;
	if (intersect_ray_sphsurf(r_origin, r_direction, cld_center, cld_radius, t)) {
		// 1. setup some constants
		float cos_angle = dot(-r_direction, sun_vector);
		float light_scattering_phase  = clamp(lerp(HenyeyGreenstein(cos_angle, cloud_forward_scattering), 
												   HenyeyGreenstein(cos_angle, -1.0 * cloud_backward_scattering),
												   0.5),
											  1.0f, 2.5f);
		float ds = 2.0f;
		vec3 point       = r_origin + r_direction * t - cld_center;
		vec3 cloud_point = compute_cloud_coordinate(point);
		float density = texture(cirrus_cloud_texture, cloud_point.xy * cirrus_cloud_texture_scale + 0.5f)[cld_index];

		Length r    = length(point);
		Number mu_s = dot(point, sun_vector) / r;

		// Indirect
		vec3 sky_radiance = GetIrradiance(ATMOSPHERE, irradiance_texture, r, mu_s) * (1.0 / PI);

		// Direct
		vec3  sun_radiance      = ATMOSPHERE.solar_irradiance * GetTransmittanceToSun(ATMOSPHERE, transmittance_texture, r, mu_s) * (1.0 / PI);
		float sun_transmittance = exponential_transmittance(density * ds) * powdersugar_transmittance(density * ds);// Cloud only scatter
		float sun_visibility    = 1.0f;

		vec3 scattered_radiance = (sky_radiance + sun_radiance * sun_transmittance * sun_visibility) * light_scattering_phase;
		transmittance = exponential_transmittance(density * ds);
		return scattered_radiance * density * ds;
	}

	transmittance = 1.0f;
	return vec3(0.0, 0.0, 0.0);
}

vec3 render_target(vec3 P, vec3 N, vec3 brdf) {
    vec3 sky_irradiance;
    vec3 sun_irradiance = GetSunAndSkyIrradiance(ATMOSPHERE, transmittance_texture, irradiance_texture,
                                                    P-earth_center, vec3(0,1,0), sun_vector, sky_irradiance);
	vec3 sky_radiance = sky_irradiance / (PI * sr);
	vec3 sun_radiance = sun_irradiance / (PI * sr);
	float sun_visibility = 0.0f;

	vec4 Pshadow = world_to_Lproj * vec4(P,1);
	     Pshadow.xyz /= Pshadow.www;

	float dx = 1.0 / 2048.0f;
	vec2 offsets[9] = vec2[9](
		vec2(-dx,  -dx), vec2(0.0f,  -dx), vec2(dx,  -dx),
		vec2(-dx, 0.0f), vec2(0.0f, 0.0f), vec2(dx, 0.0f),
		vec2(-dx,  +dx), vec2(0.0f,  +dx), vec2(dx,  +dx)
	);

	for (int i = 0; i != 9; ++i) {
		sun_visibility += float(Pshadow.z <= texture(scene_shadow_texture, Pshadow.xy * 0.5 + 0.5 + offsets[i]).r  ? 1.0f : 0.0f);
	}
	sun_visibility /= 9.0;

	//sun_visibility = float(Pshadow.z <= (texture(scene_shadow_texture, Pshadow.xy * 0.5 + 0.5).r) ? 1.0f : 0.0f);
	return brdf * (sky_radiance + sun_radiance * sun_visibility);
}

void main() {
    float depth = texture(scene_depth_texture, texcoord).r;
	vec4  temp    = proj_to_world * vec4(texcoord * 2 - 1, 0.0f, 1.0);
	vec3  Pcamera = temp.xyz / temp.w;
          temp    = proj_to_world * vec4(texcoord * 2 - 1, depth, 1.0);
	vec3  Ptarget = temp.xyz / temp.w;

	bool  hit_scene       = (depth != 1.0f);
	float target_distance = length(Pcamera - Ptarget);
    vec3  view_direction  = (Ptarget - Pcamera) / target_distance;
	      target_distance = hit_scene ? target_distance : 99999999.0f;


	float altitude = compute_altitude(Pcamera);
	float ground_distance       = -1.0f;
	float cloud_bottom_distance = -1.0f;
	float cloud_top_distance    = -1.0f;
	float atmosphere_distance   = -1.0f;
	bool hit_ground       = intersect_ray_sphsurf(Pcamera, view_direction, earth_center, 6360000.0f + ground_height, ground_distance);
	bool hit_cloud_bottom = intersect_ray_sphsurf(Pcamera, view_direction, earth_center, 6360000.0f + cloud_bottom_height, cloud_bottom_distance);
	bool hit_cloud_top    = intersect_ray_sphsurf(Pcamera, view_direction, earth_center, 6360000.0f + cloud_top_height, cloud_top_distance);
	bool hit_atmosphere   = intersect_ray_sphsurf(Pcamera, view_direction, earth_center, 6360000.0f + atmosphere_top_height, atmosphere_distance);
	bool above_cloud = altitude > cloud_top_height;
	bool below_cloud = altitude < cloud_bottom_height;
	bool in_cloud    = !(above_cloud || below_cloud);

	// atmosphere0_scattering_and_transmittance: Pcamera to cloud_bottom
	// cloud_scattering_and_transmittance: cloud_bottom to cloud_top 
	// atmosphere1_scattering_and_transmittance: cloud_top to atmosphere 
	// no Cloud: atmosphere0_radiance + atmosphere1_radiance * atmosphere0_transmittance
	// has Cloud: atmosphere0_radiance + cloud_radiance * atmosphere0_transmittance + atmosphere1_radiance * atmosphere0_transmittance * cloud_transmittance
    vec3 radiance      = vec3(0.0f, 0.0f, 0.0f);
	vec3 transmittance = vec3(1.0f, 1.0f, 1.0f);

	if (below_cloud) {
		radiance = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
										 Pcamera - earth_center,
										 (Pcamera + view_direction * min(cloud_bottom_distance, target_distance)) - earth_center, 0.0f, sun_vector, transmittance);

		if ((!hit_scene) || target_distance > cloud_bottom_distance) {
			float cloud_transmittance;
			vec3  cloud_radiance = raymarch_cloud(Pcamera, view_direction, cloud_bottom_distance, min(cloud_top_distance, target_distance), cloud_transmittance);
			//radiance      += transmittance * cloud_radiance;
			radiance += transmittance * lerp(cloud_radiance, radiance, saturate(cloud_bottom_distance / 20000.0));
			transmittance *= cloud_transmittance;

			float cloud4_transmittance;
			vec3  cloud4_scattering = raymarch_cirrus_cloud(Pcamera, view_direction, earth_center, 6360000.0 + 8000.0f, 2, cloud4_transmittance);
			radiance      += transmittance * cloud4_scattering;
			transmittance *= cloud4_transmittance;

			if ((!hit_scene) || target_distance > cloud_top_distance) {
				vec3 atmosphere1_transmittance;
				vec3 atmosphere1_radiance = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
																  (Pcamera + view_direction * cloud_bottom_distance) - earth_center,
																  (Pcamera + view_direction * min(atmosphere_distance, target_distance)) - earth_center, 0.0f, sun_vector, atmosphere1_transmittance);
				radiance      += transmittance * atmosphere1_radiance;
				transmittance *= atmosphere1_transmittance;
			}

		}

	} else if (in_cloud) {
		float cloud_tatget_distance = (hit_cloud_bottom && cloud_top_distance > cloud_bottom_distance) ? cloud_bottom_distance : cloud_top_distance;

		float cloud_transmittance = 1.0f;
		vec3  cloud_scattering    = raymarch_cloud(Pcamera, view_direction, 0.0f, min(cloud_tatget_distance, target_distance), cloud_transmittance);
		radiance      += transmittance * cloud_scattering;
		transmittance *= cloud_transmittance;

		if (target_distance > cloud_tatget_distance) {
			vec3 atmosphere1_transmittance;
			vec3 atmosphere1_sacttering = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
																(Pcamera + view_direction * cloud_tatget_distance) - earth_center,
																(Pcamera + view_direction * min(atmosphere_distance, target_distance)) - earth_center, 0.0f, sun_vector, atmosphere1_transmittance);
			radiance      += transmittance * atmosphere1_sacttering;
			transmittance *= atmosphere1_transmittance;
		}

	}

	// accumulate scene radiance
	if (depth != 1.0f) {
		vec3 albedo = texture(scene_brdf_texture, texcoord).xyz;
		vec3 normal = normalize(texture(scene_normal_texture, texcoord).xyz * 2.0 - 1.0f);;
		radiance += transmittance * render_target(Ptarget, normal, albedo);
	}
	
	// accumulate solar radiance
	if (dot(view_direction, sun_vector) > sun_size.y) {
		radiance = radiance + transmittance * GetSolarRadiance();
	}

    gl_FragColor.xyz = pow(vec3(1.0f) - exp(-radiance / white_point * exposure), vec3(1.0f / 2.2f));
	gl_FragColor.a   = 1.0f;
	//gl_FragColor = vec4(texture(scattering_texture, vec3(texcoord, 0)).xyz, 1.f);
	//gl_FragColor = vec4(texture(scene_shadow_texture, texcoord).xyz, 1.f);
}
