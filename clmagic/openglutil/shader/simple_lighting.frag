#version 400 core

/*
@_Example:
	g_shader.uniform_set("g_light_pos",   _Light.position);
	g_shader.uniform_set("g_light_color", _Light.color);
	g_shader.uniform_set("g_eye_pos",     g_camera.eyepos());
*/

// < simple_lighting >
uniform vec3 g_light_pos = vec3(100.f);
uniform vec3 g_light_color = vec3(1.f);
uniform vec3 g_eye_pos;
// </ simple_lighting >

// < simple_lighting >
vec3 simple_lighting(vec3 _Color, vec3 _Position, vec3 _Normal)
	{
	vec3 _Tolight = normalize(g_light_pos - _Position);
	vec3 _Toeye = normalize(g_eye_pos - _Position);
	vec3 _Halfvec = normalize(_Tolight + _Toeye);

	float _Ambient = 0.2;
	float _Diffuse = max( 0, dot(_Tolight, _Normal) );
	float _Specular = pow( clamp(dot(_Halfvec, _Normal), 0, 1), 32.0);

	return ( _Color * (_Ambient + _Diffuse) + _Specular );
	}
// </ simple_lighting >
