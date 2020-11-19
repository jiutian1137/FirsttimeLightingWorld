#include "glsl.h"
#include "../EricBruneton/functions.glsl"

in vec2 texcoord;
// scene
uniform sampler2D albedo_texture;
uniform sampler2D normal_texture;
uniform sampler2D depth_texture;
uniform mat4 proj2world;
uniform vec3 earth_center = vec3(0, -6360000, 0);
uniform vec3 sun_direction;
uniform vec2 sun_size;
uniform vec3 camera;
// atmosphere
uniform sampler2D transmittance_texture;
uniform sampler3D scattering_texture;
uniform sampler2D irradiance_texture;
uniform sampler3D single_mie_scattering_texture;
// visual
uniform float exposure   = 10.0F;
uniform vec3 white_point = vec3(1.0, 1.0, 1.0);
uniform float t = 0.0f;
// cloud
uniform sampler3D base_cloud_texture;
uniform sampler3D cloud_details_texture;
uniform sampler2D weather_texture;
uniform sampler2D curl_texture;
uniform sampler2D height_gradient_texture;


//const float PI = 3.14159265;
const vec3 kSphereCenter = vec3(0.0, 0.0, 1000.0);
const float kSphereRadius = 1000.0;
const vec3 kSphereAlbedo = vec3(0.8);
const vec3 kGroundAlbedo = vec3(0.0, 0.0, 0.04);
#ifdef USE_LUMINANCE
#define GetSolarRadiance GetSolarLuminance
#define GetSkyRadiance GetSkyLuminance
#define GetSkyRadianceToPoint GetSkyLuminanceToPoint
#define GetSunAndSkyIrradiance GetSunAndSkyIlluminance
#endif

RadianceSpectrum GetSolarRadiance() {
    return ATMOSPHERE.solar_irradiance /
        (PI * ATMOSPHERE.sun_angular_radius * ATMOSPHERE.sun_angular_radius);
}

const vec3 earthO = vec3(0.0f, -6360000.f, 0.0f);
const float ground_height       = 0.0;
const float cloud_bottom_height = 1500.0f;
const float cloud_top_height    = 4000.0f;
const float cloud_thickness     = 2500.0f;
const float atmosphere_top_height = 60000.0f;

struct segment_t{
	vec3 origin;
	vec3 direction;
	float min_t;
	float max_t;
};

struct sphere_t{
	vec3 center;
	float radius;
};

bool quadratic(float A, float B, float C, out float t0, out float t1) {// return false if(B^2-4AC < 0)
	float discrim = B*B - 4*A*C;
	if (discrim < 0.0) {
		return false;
	} else {
		discrim = sqrt(discrim);
		t0 = (-B - discrim) * 0.5F / A;
		t1 = (-B + discrim) * 0.5F / A;
		return true;
	}
}

bool sphere_equation(vec3 origin, vec3 direction, vec3 center, float radius, out float t0, out float t1){
	vec3 Co = (origin - center);
	float  A  = dot(direction, direction);
	float  B  = dot(direction, Co)*2;
	float  C  = dot(Co,Co) - radius*radius;
	return quadratic(A, B, C, t0, t1);
}

bool test_segment_sphere(segment_t r, sphere_t sph){
	vec3  Co = (r.origin - sph.center);
	float A  = dot(r.direction, r.direction);
	float B  = dot(r.direction, Co)*2;
	float C  = dot(Co,Co) - sph.radius*sph.radius;
	float t0, t1;
	if ( quadratic(A, B, C, t0, t1) ) {
		if (r.min_t <= t0 && t0 <= r.max_t || r.min_t <= t1 && t1 <= r.max_t) {
			return true;
		}
	}
	return false;
}

bool intersect_segment_sphere(segment_t r, sphere_t sph, out float t){
	// 1. Solve sphere equation
	vec3  Co = (r.origin - sph.center);
	float A  = dot(r.direction, r.direction);
	float B  = dot(r.direction, Co)*2;
	float C  = dot(Co,Co) - sph.radius*sph.radius;
	float t0, t1;
	if ( quadratic(A, B, C, t0, t1) ) {
	// 2. Get luminance.t from t0 and t1
		if (r.min_t <= t0 && t0 <= r.max_t) {
			t = t0; return true;
		} else if (r.min_t <= t1 && t1 <= r.max_t) {
			t = t1; return true;
		}
	}

	return false;
}

float remap(float old_val, float old_min, float old_max, float new_min, float new_max){
	return (old_val - old_min)/(old_max-old_min)*(new_max-new_min) + new_min;
}

float rescale(float vMin, float vMax, float v){
	return saturate((v - vMin) / (vMin - vMax));
}

float Beer_law(float optical_depth) {
	return exp(-optical_depth);
}

float powder_sugar_effect(float optical_depth) {
	return 1.0f - exp( - (optical_depth * 2.0f) );
}

float HGphase(float mu, float g){
	float pi = 3.141592f;
	float gg = g*g;
	float a  = 1 - gg;
	float b  = pow(1.0f + gg - 2*g*mu, 3.0f/2.0f);
	return (a/b) / 4.0f * pi;
}

vec3 ambientLight(float heightFrac) {
	return lerp(
		vec3(0.5f, 0.67f, 0.82f),
		vec3(1.0f, 1.0f, 1.0f),
		heightFrac);
	//	return vec3(0,0,1);
}


float compute_height_gradient(float relativeHeight, float cloudType) {
	return texture(height_gradient_texture, vec2(relativeHeight, cloudType)).r;

   /* float cumulus = max(0.0f, remap(relativeHeight, 0.01, 0.3, 0.0, 1.0) * remap(relativeHeight, 0.6, 0.95, 1.0, 0.0));
	float stratocumulus = max(0.0f, remap(relativeHeight, 0.0, 0.25, 0.0, 1.0) * remap(relativeHeight,  0.3, 0.65, 1.0, 0.0)); 
    float stratus = max(0.0f, remap(relativeHeight, 0, 0.1, 0.0, 1.0) * remap(relativeHeight, 0.2, 0.3, 1.0, 0.0)); 
	return stratus;

	float a = lerp(stratus, stratocumulus, clamp(cloudType * 2.0f, 0.0f, 1.0f));

    float b = lerp(stratocumulus, stratus, clamp((cloudType - 0.5) * 2.0, 0.0, 1.0));
    return smoothstep(a, b, cloudType);*/
}

float compute_altitude(vec3 pos){
	return length(pos - earthO) - 6360000.f;
}

float compute_relative_height(vec3 pos) {
	return saturate((compute_altitude(pos) - cloud_bottom_height) / cloud_thickness);
	//return (compute_altitude(pos) - cloud_bottom_height) / cloud_thickness;
}

vec3 sample_weather_data(vec2 sample_pos){
	return texture(weather_texture, (sample_pos) / 30000.0f + 0.5f).xyz;
}

float sample_base_cloud(vec3 sample_pos, float relative_height, vec3 weather_data){
	sample_pos.xz -= vec2(1.0f, 0.0f) * relative_height * 250.0f;
	sample_pos -= vec3(1.0f, 0.0f, 0.0f) * 3.0f * t * 500.0f;
	sample_pos /= 10000.0f;

	float base_cloud = texture(base_cloud_texture, sample_pos).r;

	float type     = weather_data.g;
	float height_gradient = compute_height_gradient(relative_height, type);
	base_cloud *= height_gradient;

	float coverage = weather_data.r;
	float base_cloud_with_coverage = remap(base_cloud, 0.6f, 1.0f, 0.0f, 1.0f) * 0.6f;

	//return remap(base_cloud, 1.0f - coverage, 1.0f, 0.0f, 1.0f) * weather_data.r * height_gradient;
	return base_cloud_with_coverage;
}

float erose_cloud(vec3 sample_pos, float relative_height, float base_cloud_with_coverage){
	sample_pos.xz -= vec2(1.0f, 0.0f) * relative_height * 250.0f;
	sample_pos -= vec3(1.0f, 0.0f, 0.0f) * 3.0f * t * 500.0f;
	sample_pos /= 10000.0f;

	vec2 curl = texture(curl_texture, sample_pos.xy).xy;

	//sample_pos.xy += curl * relative_height;

	float high_freq_fbm = texture(cloud_details_texture, sample_pos).r;
	float high_freq_noise_modifier = lerp(high_freq_fbm, 1.0f - high_freq_fbm, saturate(relative_height*4.0f));

	float final_cloud = remap(base_cloud_with_coverage, high_freq_noise_modifier * 0.5f, 1.0f, 0.0f, 1.0f);
	return final_cloud;
//	return base_cloud_with_coverage - high_freq_fbm*curl;
}

float map(in float input_value, in float input_start, in float input_end, in float output_start, in float output_end) {
	float slope = (output_end - output_start) / (input_end - input_start);

	return clamp((slope * (input_value - input_start)) + output_start, min(output_start, output_end), max(output_start, output_end));
}

float henyey_greenstein(in float dot_angle, in float scattering_value) {
	float squared_scattering_value = pow(scattering_value, 2.0f);

	return (1.0 - squared_scattering_value) / (4.0f * PI * pow(squared_scattering_value - (2.0f * scattering_value * dot_angle) + 1.0f, 1.5f));
}

float get_height_ratio(in vec3 ray_position) {
	return compute_relative_height(ray_position);
}

// cirrus

uniform vec3 wind_offset = vec3(250.0f, 0.0f, 250.0f);
uniform float base_noise_scale = 0.00004f;
uniform vec3 base_noise_ratios = vec3(0.625f, 0.25f, 0.125f);

uniform float cloud_map_scale = 0.000005f;
uniform float cloud_coverage = 0.85f;

uniform float detail_noise_scale = 0.0004f;
uniform vec3 detail_noise_ratio = vec3(0.625f, 0.25f, 0.125f);

uniform float cloud_density = 0.4f;
uniform float fade_start_distance = 3000.0f;
uniform float fade_end_distance = 3000.0f;

uniform vec3 sun_tint = vec3(1.0f);
uniform float sun_gain = 0.2f;

uniform vec3 ambient_tint = vec3(1.0f);
uniform float ambient_gain = 0.8f;

uniform float forward_mie_scattering = 0.85f;
uniform float backward_mie_scattering = 0.45f;

uniform vec3 atmosphere_bottom_tint = vec3(0.55f, 0.775f, 1.0f);
uniform vec3 atmosphere_top_tint = vec3(0.45f, 0.675f, 1.0f);

uniform float atmospheric_blending = 0.65f;

uniform float blue_noise_scale = 0.01f;

float sample_clouds(in vec3 ray_position, vec3 ray_start_position) {
	vec4 base_noise_sample = texture(base_cloud_texture, ray_position * base_noise_scale);
	float base_noise = map(base_noise_sample.x, dot(base_noise_sample.yzw, base_noise_ratios), 1.0, 0.0, 1.0);

	vec2 cloud_map_sample = texture(weather_texture, (ray_position.xz + wind_offset.xz * 2 * t) * cloud_map_scale).xy;

	//if (cloud_types[cloud_layer_index] == 1) cloud_map_sample.y = 1.0; else
	cloud_map_sample.y = map(cloud_map_sample.y, 0.0, 1.0, 0.625, 1.0);

	float height_ratio = get_height_ratio(ray_position);
	//float height_multiplier = min(map(height_ratio, 0.0, 0.125, 0.0, 1.0), map(height_ratio, 0.625 * cloud_map_sample.y, cloud_map_sample.y, 1.0, 0.0));
	float height_multiplier = compute_height_gradient(height_ratio, cloud_map_sample.y);
	 
	//float base_erosion = map(base_noise * height_multiplier, 1.0f - max(cloud_map_sample.x, cloud_coverage), 1.0, 0.0, 1.0);
	float base_erosion = map(base_noise * height_multiplier, 1.0f - cloud_map_sample.x, 1.0, 0.0, 1.0);

	if (base_erosion > 0.01) {
		vec3 detail_noise_sample = texture(cloud_details_texture, ray_position * detail_noise_scale).xyz;
		float detail_noise = dot(detail_noise_sample, detail_noise_ratio);
		detail_noise = lerp(detail_noise, 1.0f - detail_noise, height_ratio);

		return map(base_erosion, 0.625 * detail_noise, 1.0, 0.0, 1.0)
			* cloud_density
			* cloud_map_sample.x * height_ratio;
			//* map(length(ray_position - ray_start_position), fade_start_distance, fade_end_distance, 1.0, 0.0);
	}
	else return 0.0;
}

float sun_ray_march(in float input_transmittance, in vec3 ray_position, vec3 ray_start_position) {
	int SUN_STEP_COUNT = 6;

	float output_optical_depth = 0.0f;
	float output_transmittance = input_transmittance;

	float step_size = min((cloud_top_height - ray_position.y) / SUN_STEP_COUNT, 50.0f);

	vec3 current_ray_position = ray_position;

	for (int step_index = 0; step_index < SUN_STEP_COUNT; step_index++) {
		float cloud_sample = sample_clouds(current_ray_position, ray_start_position);
		output_optical_depth += cloud_sample * step_size;

	/*	output_transmittance *= exp(-1.0 * cloud_sample * step_size);
		if (output_transmittance < 0.01f) break;*/
		current_ray_position += sun_direction * step_size;
	}

	current_ray_position += step_size * 8 * sun_direction;
	if (compute_relative_height(current_ray_position) < 0.99f) {
		output_optical_depth += sample_clouds(current_ray_position, ray_start_position) * step_size;
	}

	//return exp(-1.0f * output_optical_depth);
	return 2.0f * Beer_law(output_optical_depth) * powder_sugar_effect(output_optical_depth);
}


vec3 raymarch(vec3 o, vec3 d, inout float transmittance){
	segment_t r           = segment_t(o, d, 1.0, 10000000.0);
	sphere_t cloud_bottom = sphere_t(earthO, cloud_bottom_height + 6360000.0f);
	sphere_t cloud_top    = sphere_t(earthO, cloud_top_height + 6360000.0f);
	sphere_t ground       = sphere_t(earthO, 6360000.0f);

	bool is_raymarch = false;
	float ray_begin;
	float ray_end;

	if( !test_segment_sphere(r, ground) ){
		intersect_segment_sphere(r, cloud_bottom, ray_begin);
		intersect_segment_sphere(r, cloud_top, ray_end);
		if(ray_end < 100000.0f){
			is_raymarch = true;
		}
	}

	float mu = dot(sun_direction, -d);

	if(is_raymarch){
		int   nstep   = 64;
		//float total_s = ray_end - ray_begin;

		//float ds = total_s / float(nstep);
		//vec3  dx = d * ds;
		//vec3  x  = o + d * (ray_begin + ds*0.5f);

		//vec3  luminance = vec3(0.0f);

		//for(int i = 0; transmittance > 0.1f && i != nstep; ++i, x += dx) {
		//	vec3  weather_data    = sample_weather_data(x.xz);
		//	float relative_height = compute_relative_height(x);

		//	//x.xz += vec2(250.0f, 250.0f) * relative_height;
		//	float base_cloud      = sample_base_cloud(x, relative_height, weather_data);

		//	if(base_cloud > 0.002f) {
		//		float sampled_density       = erose_cloud(x, relative_height, base_cloud) * relative_height;
		//		float sampled_optical_depth = sampled_density * ds;
		//		float sampled_transmittance = Beer_law( sampled_optical_depth );
		//		transmittance *= sampled_transmittance;// exp(-(sigma_e1 * s1)) * exp(-(sigma_e2 * s2)) = exp( - (sigma_e1 * s1 + sigma_e2 * s2) )

		//		/*float density_along_light = raymarch_light(x, sun_direction, 11.0f);
		//		float light_energy        = 32.0f * Beer_law(density_along_light * lerp(1.8f, 3.6f, smoothstep(0.15f, 0.4f, sun_direction.z))) * powder_sugar_effect(density_along_light);
		//		vec3  luminance           = ambientLight(relative_height) * vec3(1.0f, 0.3f, 0.3f) * HGphase(mu, 0.2f) * light_energy * sampled_density;
		//		vec3  inscatter           = (luminance - luminance * sampled_transmittance) / sampled_density;
		//		luminance += inscatter * transmittance;*/
		//		float light_luminance = raymarch_light(x, sun_direction, ds);
		//		luminance += (light_luminance * ambientLight(relative_height)) * transmittance;
		//	}
		//}

		//return luminance * HGphase(mu, 0.2f);

		float MAXIMUM_SAMPLE_STEP_SIZE = 100.0f;

		vec4 output_color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
		float accumulate_optical_depth = 0.0f;

		vec3 ray_start_position = o;
		vec3 ray_direction = d;

		float ray_march_distance = ray_end - ray_begin;

		vec3  current_ray_position = ray_start_position + (ray_direction * ray_begin);
		float current_ray_distance = 0.0;

		float step_size = min(ray_end / nstep, MAXIMUM_SAMPLE_STEP_SIZE);

		float sun_angle_multiplier = map(sun_direction.y, 0.05, -0.125, 1.0, 0.125);

		float sun_dot_angle = dot(ray_direction, sun_direction);
		float mie_scattering_gain = clamp(lerp(henyey_greenstein(sun_dot_angle, forward_mie_scattering), henyey_greenstein(sun_dot_angle, -1.0 * backward_mie_scattering), 0.5), 1.0, 2.5);

		while (current_ray_distance <= ray_march_distance)
		{
			//float current_step_size = step_size * map(texture(curl_texture, current_ray_position.xz * blue_noise_scale).x, 0.0, 1.0, 0.75, 1.0);
			float current_step_size = step_size;

			float distance_multiplier;

			/*if (current_ray_distance < 40000.0) distance_multiplier = map(current_ray_distance, 0.0, 40000.0, 1.0, 4.0);
			else distance_multiplier = map(current_ray_distance, 40000.0, ray_march_distance, 4.0, 16.0);

			current_step_size *= distance_multiplier;*/

			float cloud_sample = sample_clouds(current_ray_position, ray_start_position);


			if (cloud_sample != 0.0) {
				float light_attenuation = 1.0;
				//for (int current_layer_index = cloud_layer_index; current_layer_index < CLOUD_LAYER_COUNT; current_layer_index++) 
				light_attenuation = sun_ray_march(light_attenuation, current_ray_position, ray_start_position) * mie_scattering_gain;

				vec3 sun_color = vec3(1.0f) * sun_gain * sun_angle_multiplier;
				vec3 transmittance2;
				vec3 ambient_color = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
															camera - earth_center, current_ray_position - earth_center, 0.0F, sun_direction,
															transmittance2);
				//vec3 ambient_color = lerp(ambient_tint, lerp(atmosphere_bottom_tint, atmosphere_top_tint, get_height_ratio(current_ray_position)) * dot(ambient_tint, vec3(0.21, 0.72, 0.07)), atmospheric_blending) * ambient_gain * sun_angle_multiplier;

				vec3 sample_color = (sun_color + ambient_color) * light_attenuation * cloud_sample * current_step_size;

				float sampled_optical_depth = cloud_sample * current_step_size;

				output_color.xyz += sample_color * output_color.w;
				accumulate_optical_depth += sampled_optical_depth;
				output_color.w = Beer_law(accumulate_optical_depth);

				if (output_color.w < 0.01) {
					output_color.w = 0.0f;
					break;
				}
			}

			current_ray_position += ray_direction * current_step_size;
			current_ray_distance += current_step_size;
		}

		transmittance = output_color.w;
		return output_color.xyz;
	}

	return vec3(0.0f);
}


void main() {
  // Normalized view direction vector.
    float depth = texture(depth_texture, texcoord).r;
    vec4  posP  = vec4(texcoord * 2 - 1, depth, 1.0);
    vec4  temp  = (proj2world * posP);

    vec3  the_point  = temp.xyz / temp.w;
    vec3  view_direction = normalize(the_point - camera);

    vec3 radiance = vec3(0);

    if (depth != 1.0) {
        vec3 albedo = texture(albedo_texture, texcoord).xyz;
        vec3 normal = texture(normal_texture, texcoord).xyz;

        vec3 sky_irradiance;
        vec3 sun_irradiance = GetSunAndSkyIrradiance(ATMOSPHERE, transmittance_texture, irradiance_texture,
                                                     the_point-earth_center, normal, sun_direction, sky_irradiance);
        vec3 scene_radiance = albedo * (1.0/PI) * (sky_irradiance + sun_irradiance);

        vec3 transmittance;
        vec3 in_scatter = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
                                                camera - earth_center, the_point - earth_center, 0.0F, sun_direction,
                                                transmittance);
        radiance = scene_radiance * transmittance + in_scatter;
    } else {
		segment_t r                = segment_t(camera, view_direction, 1.0, 10000000.0f);
		sphere_t atmosphere_top    = sphere_t(earthO, atmosphere_top_height + 6360000.0f);
		sphere_t atmosphere_bottom = sphere_t(earthO, 6360000.0f);
		float t = 0.0f;
		float atmosphere_bottom_t;
		float atmosphere_top_t;
		if( intersect_segment_sphere(r, atmosphere_bottom, atmosphere_bottom_t) ) {
			intersect_segment_sphere(r, atmosphere_top, atmosphere_top_t);
			t = min(atmosphere_bottom_t, atmosphere_top_t);
		} else {
			intersect_segment_sphere(r, atmosphere_top, atmosphere_top_t);
			t = atmosphere_top_t;
		}

       /* vec3 transmittance;
        radiance = GetSkyRadiance(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture, camera - earth_center, view_direction, 0.0, sun_direction, transmittance);*/
        
		vec3 transmittance1;
		vec3 radiance1 = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
										camera - earth_center, (camera + view_direction * t*0.5f) - earth_center, 0.0F, sun_direction,
										transmittance1);
		vec3 transmittance2;
		vec3 radiance2 = GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture, scattering_texture, single_mie_scattering_texture,
										(camera + view_direction * t*0.5f) - earth_center, (camera + view_direction * t) - earth_center, 0.0F, sun_direction,
										transmittance2);
		vec3 transmittance = transmittance1 * transmittance2;
		radiance = radiance1 + radiance2 * transmittance1;
		
		// If the view ray intersects the Sun, add the Sun radiance.
        if (dot(view_direction, sun_direction) > sun_size.y) {
            radiance = radiance + transmittance * GetSolarRadiance();
        }

		float cloud_transmittance = 1.0f;
		vec3 cloud = raymarch(camera, view_direction, cloud_transmittance);
		radiance = cloud + radiance * cloud_transmittance;
    }

    gl_FragColor.xyz = pow(vec3(1.0) - exp(-radiance / white_point * exposure), vec3(1.0 / 2.2));
	gl_FragColor.a   = 1.0f;
	//gl_FragColor = vec4(texture(base_cloud_texture, vec3(texcoord, 0)).a, 0.f, 0.f, 1.f);
}
