#include "../../atmosphere/Atmosphere.glsl.h"

struct CloudProfile {
	Length bottom_height;
	Length top_height;
	Number density_scale;
};


// { remap(value, lower, upper, 0.0, 1.0) }
Number rescale(float value, float lower, float upper) {
	return (value - lower) / (upper - lower);
}

Number saturate(Number value) {
	return clamp(value, 0.0, 1.0);
}

float HenyeyGreenstein(float dot_angle, float scattering_value) {
	float squared_scattering_value = pow(scattering_value, 2.0f);
	return (1.0 - squared_scattering_value) / 
		(4.0f * PI * pow(squared_scattering_value - (2.0f * scattering_value * dot_angle) + 1.0f, 1.5f));
}

float DoblobeHenyeyGreenstein(float dot_angle, float scattering_value0, float scattering_value1, float t) {
	return mix(HenyeyGreenstein(dot_angle, scattering_value0), HenyeyGreenstein(dot_angle, scattering_value1), t);
}

Number exponential_transmittance(Length opticalLength) {
	return exp(-opticalLength);
}

Number powdersugar_transmittance(Length opticalLength) {
	return 1.0 - exp(-(opticalLength * 2.0));
}


#define CloudPostion vec3 // { [0, 1], [0, 1], [0, 1] }

Length Altitude(
	Position point,
	Position earth_center) {
	return length(point - earth_center) - ATMOSPHERE.bottom_radius;
}

Number CloudHeightRatio(
	CloudProfile profile, 
	Position point,
	Position earth_center) {
	// no saturate
	return rescale(Altitude(point, earth_center), profile.bottom_height, profile.top_height);
}

CloudPostion World2CloudCoordinate(
	CloudProfile profile, 
	Position point, Position earth_center) {

	point -= earth_center;
	Length r       = length(point);
	Angle  azimuth = atan(point.y, point.x); if(azimuth < 0.0) { azimuth += 2.0*pi; }
	Angle  zenith  = acos( clamp( point.z / r, -1.0f, 1.0f ) );
	Number height_ratio = rescale(r - ATMOSPHERE.bottom_radius, profile.bottom_height, profile.top_height);
	return CloudPostion(azimuth, zenith * 2.0, height_ratio);// { [0,2Pi], [0,Pi], [0,1] }
}

Number GetCloudDensity(
	CloudProfile profile, 
	sampler3D traits, sampler3D detail, float base_scale, float detail_frequency, float detail_weight, float detail_speed, vec3 detail_curl_vector, int detail_octive,
	sampler2D weather_map, float weather_map_scale, sampler2D height_gradient_map, 
	Position point, 
	Position earth_center,
	out float percipitation) {
	
	// 1. Get CloudPosition
	CloudPostion cloud_point = World2CloudCoordinate(profile, point, earth_center);
	Number height_ratio = cloud_point[2];
	// 2. Get Weather using CloudPosition
	//vec3   weather_data = texture(weather_map, point.xz * (1.0 / 50000.0) + 0.5).xyz;
	vec3   weather_data = texture(weather_map, cloud_point.xy * weather_map_scale + 0.5).xyz;
	Number coverage = weather_data[0];
	Number type    = weather_data[1];
	percipitation = weather_data[2];
	Number coverage_height = texture(height_gradient_map, vec2(type, height_ratio)).r;

	// 3. Check height and coverage
	if (0.0 <= height_ratio && height_ratio <= 1.0 && coverage != 0.0 && coverage_height != 0.0) {
		
		// 4. Get BaseCloud using height and coverage
		float density = rescale((coverage + coverage_height)*0.5, 1.0 - 0.5 * texture(traits, point * base_scale).r, 1.0);
		      /*density = saturate(rescale(density, 1.0 - coverage, 1.0));
		      density = saturate(rescale(density, 1.0 - coverage_height, 1.0));*/

		if (density > 0.0) {
			/*vec3 direction = vec3(0.0); 
			direction.xz += texture(curl_texture, point.xz * base_scale).xz;
			direction.xy += texture(curl_texture, point.xy * base_scale).xy;*/
			//direction = direction * 2.0 - 1.0;
			// 5. Get Details(erosion) using more-and-more-High-frequency-noise
			float detail_scale  = base_scale * detail_frequency;
			float detail_speed  = detail_speed;
			for (int i = 0; i != detail_octive; ++i) {
				density = saturate(rescale(density, 
					detail_weight * (1.0 - texture(detail, (point + detail_curl_vector * detail_speed) * detail_scale).r), 1.0));
				detail_scale  *= detail_frequency;
				detail_weight *= 0.5;
				detail_speed  *= detail_frequency;
			}

			// 6. Control with artist and physics(density increment with height)
			return density * profile.density_scale * height_ratio;
		}
	}

	percipitation = 0.0;
	return 0.0;
}