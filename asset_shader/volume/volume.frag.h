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
uniform CloudProfile CLOUDSPHERE0 = CloudProfile(3000.0, 5000.0, 0.1);
// sky property
uniform sampler2D transmittance_texture;
uniform sampler2D irradiance_texture;
uniform vec3 sun_vector;
uniform float sky_transmittance = 1.0;
// scene property
uniform sampler2D scene_depth_texture;
uniform mat4 proj_to_world;
uniform mat4 world_to_proj;

out vec4 radiance_and_transmittance;
// out float gl_FragDepth

//uniform sampler2D scene_shadow_texture;
//uniform mat4 world_to_sunproj;


DimensionlessSpectrum GetTransmittanceToPoint(
	IN(AtmosphereParameters) atmosphere,
	IN(TransmittanceTexture) transmittance_texture, 
	Position camera, IN(Position) point, IN(Direction) sun_direction) {
  Direction view_ray = normalize(point - camera);
  Length r = length(camera);
  Length rmu = dot(camera, view_ray);
  Length distance_to_top_atmosphere_boundary = -rmu -
      sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
  // If the viewer is in space and the view ray intersects the atmosphere, move
  // the viewer to the top atmosphere boundary (along the view ray):
  if (distance_to_top_atmosphere_boundary > 0.0 * m) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r = atmosphere.top_radius;
    rmu += distance_to_top_atmosphere_boundary;
  }

  // Compute the r, mu, mu_s and nu parameters for the first texture lookup.
  Number mu = rmu / r;
  Number mu_s = dot(camera, sun_direction) / r;
  Number nu = dot(view_ray, sun_direction);
  Length d = length(point - camera);
  bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

  return GetTransmittance(atmosphere, transmittance_texture,
      r, mu, d, ray_r_mu_intersects_ground);
}

Number GetVolumesDensity(Position point, Position earth_center, out Number percipitation) {
	vec3 curl = vec3(0.0);
	curl.xy += texture(cloud_curl_texture, point.xy * base_cloud_texture_scale).xy;
	curl.xz += texture(cloud_curl_texture, point.xz * base_cloud_texture_scale).xz;
	curl = curl * 2.0 - 1.0;

	Number percipitation1;
	Number density = GetCloudDensity(CLOUDSPHERE0, base_cloud_texture, cloud_detail_texture, base_cloud_texture_scale, 4.0, 0.1, 40.0, vec3(1, 0, 0) * time, 1,
		cloud_weather_texture_first, cloud_weahter_texture_scale, cloud_height_texture, point, earth_center, percipitation1);

	Number percipitation2;
	density += GetCloudDensity(CloudProfile(1000, 4000, 0.1), base_cloud_texture, cloud_detail_texture, base_cloud_texture_scale, 4.0, 0.1, 20.0, vec3(1, 0, 0) * time, 1,
		cloud_weather_texture_second, cloud_weahter_texture_scale, cloud_height_texture, point, earth_center, percipitation2);

	percipitation = percipitation1 + percipitation2;
	return density;
}

Number GetLightingVolumeDensity(Position point, Position earth_center, out Number percipitated_density) {
	vec3 curl = vec3(0.0);
	curl.xy += texture(cloud_curl_texture, point.xy * base_cloud_texture_scale).xy;
	curl.xz = texture(cloud_curl_texture, point.xz * base_cloud_texture_scale).xz;
	curl = curl * 2.0 - 1.0;

	Number percipitation1;
	Number density1 = GetCloudDensity(CLOUDSPHERE0, base_cloud_texture, cloud_detail_texture, base_cloud_texture_scale, 4.0, 0.1, 40.0, vec3(1, 0, 0) * time, 1,
		cloud_weather_texture_first, cloud_weahter_texture_scale, cloud_height_texture, point, earth_center, percipitation1);

	Number percipitation2;
	Number density2 = GetCloudDensity(CloudProfile(1000, 4000, 0.1), base_cloud_texture, cloud_detail_texture, base_cloud_texture_scale, 4.0, 0.1, 20.0, vec3(1, 0, 0) * time, 1,
		cloud_weather_texture_second, cloud_weahter_texture_scale, cloud_height_texture, point, earth_center, percipitation2);

	percipitated_density = percipitation1 * density1 + percipitation2 * density2;
	return density1 + density2;
}

const vec3 noiseKernel[6u] = vec3[] (
	vec3( .38051305,  .92453449, -.02111345),
	vec3(-.50625799, -.03590792, -.86163418),
	vec3(-.32509218, -.94557439,  .01428793),
	vec3( .09026238, -.27376545,  .95755165),
	vec3( .28128598,  .42443639, -.86065785),
	vec3(-.16852403,  .14748697,  .97460106)
);

void main() {
	// 1. Constants
	vec3 earth_center = vec3(0, -ATMOSPHERE.bottom_radius, 0);
	vec2  texcoord    = gl_FragCoord.xy / vec2(resolution);
	float scene_depth = texture(scene_depth_texture, texcoord).r;

	vec4 ray_start = proj_to_world * vec4((texcoord - 0.5) * 2.0, 0.0, 1.0);
	vec4 ray_end = proj_to_world * vec4((texcoord - 0.5) * 2.0, scene_depth, 1.0);
	ray_start /= ray_start.w;
	ray_end /= ray_end.w;

	Direction ray_direction = normalize(ray_end.xyz - ray_start.xyz);
	Number cos_angle = dot(-ray_direction, sun_vector);
	Number light_scattering_phase = clamp(DoblobeHenyeyGreenstein(cos_angle, cloud_forward_scattering, -cloud_backward_scattering, 0.5), 1.0f, 2.5f);

	int lighting_step = 12;

	Length unit_length = 100.0;
	Length S           = length(ray_end.xyz - ray_start.xyz);
	int    volume_step = int(round(clamp(S / unit_length, 0.0, 256.0)));
	       unit_length = min(S / float(volume_step), 500*m);
	Length detail_length = min((unit_length / 6.0f), 50*m);
	/*Length unit_length = 60000.0 / volume_step;
	Length detail_length = unit_length * 0.2;*/
	
	// setup sampling - compute intersection of ray with 2 sets of planes
	/*vec4 t, dt;
	vec4 wt;
	SetupSampling(t, dt, wt, ray_start.xyz, ray_direction.xyz);*/

	// 2. Integrate (camera to scene)
	RadianceSpectrum radiance = vec3(0.0, 0.0, 0.0);
	Number   transmittance = 1.0;
	bool     intersect = false;
	Position intersect_point;
	Length s = 0.0;
	Number test = 0.0;
	int zero_density_count = 0;
	for (int i = 0; i != volume_step; ++i) {
		/*vec3 point;
        float ds;
        
        vec4 dataT = vec4(float(t.x <= t.y),
                          float(t.y < t.x),
                          float(t.z <= t.w),
                          float(t.w < t.z));

        dataT = vec4(dataT.xy * float(min(t.z, t.w) >= min(t.x, t.y)),
                     dataT.zw * float(min(t.z, t.w) < min(t.x, t.y)));

		point = ray_start.xyz + length(t * dataT) * ray_direction.xyz;
		ds = length(wt * dataT * dt);
        
        // fade samples at far extend
        w *= smoothstep( endFade, startFade, length(t * dataT) );

        t += dataT * dt;*/   

		Length   ds    = test != 0.0 ? detail_length : unit_length;
		s += ds;
		Position point = ray_start.xyz + ray_direction * s;
		//ds *= smoothstep(25000.0, 100.0, s);

		/*float wind_speed = 100.0;
		vec3  wind_direction = vec3(1, 0, 0);
		point += wind_direction * wind_speed * 5.0 * (1.0 - saturate(CloudHeightRatio(CLOUDSPHERE0, point, earth_center)));
		point += wind_direction * wind_speed * time;*/

		if (test != 0.0) {
			Number precipitation;
			Number sampled_cloud = GetVolumesDensity(point, earth_center, precipitation);

			if (sampled_cloud > 0.0) { 
				zero_density_count = 0;
				
				Length sampled_optical_depth = sampled_cloud * ds;
				Number sampled_transmittance = exponential_transmittance(sampled_optical_depth);
				
				Length lighting_unit_length = (CLOUDSPHERE0.top_height - Altitude(point, earth_center)) / float(lighting_step);// transmit thick clouds
				//Length lighting_unit_length = 20.0;// transmit thin clouds
				//float coneSpread = length(sun_vector);
				Number sampled_lighting_density              = sampled_cloud;
				Number sampled_precipitated_lighting_density = sampled_cloud * precipitation;
				for (int j = 1; j < lighting_step; ++j) {
					Position x = point + sun_vector * (j * lighting_unit_length)/* + coneSpread * noiseKernel[i] * float(i)*/;
					float ppppp;
					sampled_lighting_density += /*mix(1.0, 0.75, cos_angle) * */GetLightingVolumeDensity(x, earth_center, ppppp);
					sampled_precipitated_lighting_density += /*mix(1.0, 0.75, cos_angle) * */ppppp;
				}
				Length sampled_lighting_opticallength     = sampled_lighting_density * lighting_unit_length;
				Length sampled_precipitated_opticallength = sampled_precipitated_lighting_density * lighting_unit_length;
				//Position x = point + sun_vector * lighting_unit_length * 20.0;
				//float ppppp;
				//sampled_lighting_opticallength += mix(1.0, 0.75, cos_angle) * GetLightingVolumeDensity(x, earth_center, ppppp) * lighting_unit_length;
				//sampled_precipitated_lighting_density += /*mix(1.0, 0.75, cos_angle) * */ppppp * lighting_unit_length ;

				Position point_in_earth = point - earth_center;
				Length   r    = length(point_in_earth);
				Number   mu_s = dot(point_in_earth, sun_vector) / r;
				
				// Indirect
				RadianceSpectrum sky_radiance = GetIrradiance(ATMOSPHERE, irradiance_texture, r, mu_s) * (1.0 / PI);
				// Direct
				RadianceSpectrum sun_radiance = ATMOSPHERE.solar_irradiance * GetTransmittanceToSun(ATMOSPHERE, transmittance_texture, r, mu_s) * (1.0 / PI);
				Number sun_transmittance = /*mix(exponential_transmittance(sampled_precipitated_opticallength),*/
					2.0 * exponential_transmittance(sampled_precipitated_opticallength) * powdersugar_transmittance(sampled_lighting_opticallength)/*,
					cos_angle)*/;
				Number sun_visibility = 1.0f;// no used(scene occluded cloud)
				//
				DimensionlessSpectrum view_transmittance = GetTransmittanceToPoint(ATMOSPHERE, transmittance_texture, ray_start.xyz, point_in_earth, sun_vector);

				// <approximate>
				//vec3 sky_radiance = lerp(atmosphere_bottom_tint, atmosphere_top_tint, get_height_ratio(x)) * sun_gain * sun_angle_multiplier;
				//vec3 sun_radiance = sun_tint * sun_gain * sun_angle_multiplier;
				// </approximate>
				RadianceSpectrum scattered_radiance = (sky_radiance * sky_transmittance + sun_radiance * sun_transmittance * sun_visibility) * light_scattering_phase;
				//RadianceSpectrum scattered_radiance = ((sky_radiance + sun_radiance) * sun_transmittance * sun_visibility) * light_scattering_phase;
				radiance      += scattered_radiance * sampled_cloud * (transmittance * view_transmittance) * ds;
				//radiance      += (scattered_radiance - scattered_radiance * sampled_transmittance) / sampled_cloud * transmittance * ds;
				transmittance *= sampled_transmittance;
				
				if (!intersect) {
					intersect_point = point;
					intersect = true;
				}

				if (transmittance < transmit_determinant) {
					transmittance = 0.0001f;
					break;
				}

			} else {
				++zero_density_count;
				if (zero_density_count == 6) {
					test = 0.0;
					zero_density_count = 0;
				}
			}

		} else {
			float precipitation;
			test = GetVolumesDensity(point, earth_center, precipitation);
			if (test != 0.0) {
				s -= ds;
				i -= 1;
			}
		}
	}

	radiance_and_transmittance = vec4(radiance, transmittance);
	if (intersect) {
		vec4 temp = world_to_proj * vec4(intersect_point, 1);
		gl_FragDepth = temp.z / temp.w;
	} else {
		gl_FragDepth = 1.0;
	}
}


//Number CloudHeight(Number height_ratio, Number cloud_type) {
//	float cumulus       = max(0.0f, remap(height_ratio, 0.01, 0.3,  0.0, 1.0) * remap(height_ratio, 0.6, 0.95, 1.0, 0.0));
//	float stratocumulus = max(0.0f, remap(height_ratio, 0.0,  0.25, 0.0, 1.0) * remap(height_ratio, 0.3, 0.65, 1.0, 0.0));
//	float stratus       = max(0.0f, remap(height_ratio, 0.0,  0.1,  0.0, 1.0) * remap(height_ratio, 0.2, 0.3,  1.0, 0.0));
//	//return stratus;
//	float a = mix(stratus, stratocumulus, clamp(cloud_type * 2.0f, 0.0f, 1.0f));
//	float b = mix(stratocumulus, stratus, clamp((cloud_type - 0.5) * 2.0, 0.0, 1.0));
//	return smoothstep(a, b, cloud_type);
//}

//float PERIOD = 20.0;
//
//#define float2 vec2
//#define float3 vec3
//#define fmod mod
//float mymax (float3 x) {return max(x.x, max(x.y, x.z));}
//float mymin (float3 x) {return min(x.x, min(x.y, x.z));}
//
//// compute ray march start offset and ray march step delta and blend weight for the current ray
//vec4 SetupSampling( out vec4 t, out vec4 dt, out vec4 wt, in float3 ro, in float3 rd )
//{    
//    //Every possible direction under each normal type are orthogonal   
//    
//    //Axis planes
//    vec3 n0 = vec3(rd.x * float(abs(rd.x) > abs(rd.y)),
//              rd.y * float(abs(rd.y) > abs(rd.x)),
//              rd.z);
//    n0 = vec3(n0.xy * float(abs(n0.z) < max(abs(n0.x), abs(n0.y))),
//              n0.z  * float(abs(n0.z) > max(abs(n0.x),abs(n0.y))));
//    n0 = normalize(n0);
//    
//    vec3 n1 = float3(sign(rd.x), 0., sign( rd.z )); // XZ diagonals
//    vec3 n2 = float3(sign(rd.x), sign( rd.y ), 0.); //XY diagonals
//    vec3 n3 = float3(0., sign( rd.y ), sign( rd.z )); //YZ diagonals
//        
//    // normal lengths
//    vec4 ln = vec4(length( n0 ), length( n1 ), length( n2 ), length(n3));
//    n0 = normalize(n0);
//    n1 = normalize(n1);
//    n2 = normalize(n2);    
//    n3 = normalize(n3);
//
//    // some useful DPs
//    vec4 ndotro = vec4(dot( ro, n0 ), dot( ro, n1 ), dot(ro, n2), dot(ro, n3));
//    vec4 ndotrd = vec4(dot( rd, n0 ), dot( rd, n1 ), dot(rd, n2), dot(rd, n3));
//
//    // step size
//    // Gets smaller for planes that are orthogonal to the ray
//    // As we always take step until the next closest sample, sample size is independent for each plane type
//    vec4 period = ln * PERIOD;
//    dt = period / abs( ndotrd );
//   
//    // raymarch start offset - skips leftover bit to get from ro to first strata lines
//    t = -sign( ndotrd ) * fmod( ndotro, period ) / abs(ndotrd);
//
//    if( ndotrd.x > 0. ) t.x += dt.x;
//    if( ndotrd.y > 0. ) t.y += dt.y;
//	if( ndotrd.z > 0. ) t.z += dt.z;
//    if( ndotrd.w > 0. ) t.w += dt.w;
//    
//    // sample weights
//    float minperiod = PERIOD;
//    
//    //This is related to maximum length between parallel lines, sqrt2 for diagonals
//    float maxperiod = sqrt(2.) * PERIOD;
//    wt = smoothstep( maxperiod, minperiod, dt / ln );
//
//    wt /= (wt.x + wt.y + wt.z + wt.w);
//    
//    return vec4(wt);
//}