#version 400 core

struct ray_t 
{
	vec3 origin;
	vec3 direction;
};

vec3 sun_direction(float t){ return ( normalize( vec3(0, sin(t), cos(t)) ) ); }

/*
@_Example:
	g_shader.uniform_set("g_sun_dir", vec3(0.f, -0.707f, -0.707f));
	g_shader.uniform_set("g_sun_color", vec3(1.0f, 0.7f, 0.55f));
	g_shader.uniform_set("g_sky_top_color", vec3(0.3f, 0.6f, 0.8f));
	g_shader.uniform_set("g_sky_down_color", vec3(0.0f, 0.1f, 0.4f));
*/

// < sky >
uniform vec3 g_sun_dir = vec3(0, -0.707, -0.707); //@sun_direction
uniform vec3 g_sun_color = vec3(1.0, 0.7, 0.55);
uniform vec3 g_sky_top_color = vec3(0.3, 0.6, 0.8);
uniform vec3 g_sky_down_color = vec3(0.0, 0.1, 0.4);
// </ sky >

// < sky >
vec3 render_sky(ray_t _Eye)
	{
	vec3 _Rd = _Eye.direction;
	float _Sunamount = max(dot(_Rd, g_sun_dir), 0.0);

	vec3  sky = mix(g_sky_down_color, g_sky_top_color, 1.0 - _Rd.y);
	sky = sky + g_sun_color * min(pow(_Sunamount, 1500.0) * 5.0, 1.0);
	sky = sky + g_sun_color * min(pow(_Sunamount, 10.0) * 0.6, 1.0);

	return (sky);
	}
// </ sky >