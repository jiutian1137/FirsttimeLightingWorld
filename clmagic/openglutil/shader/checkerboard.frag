#version 400 core

/*
@_Example:
	g_shader.uniform_set("g_color0", vec3(1.0f, 0.0f, 0.0f));
	g_shader.uniform_set("g_color1", vec3(0.0f, 0.0f, 1.0f));
	g_shader.uniform_set("g_avg_color", vec3(1.0f, 0.5f, 1.0f));
	g_shader.uniform_set("g_frequency", 100.0f);
*/

// < checker_board >
uniform vec3 g_color0 = vec3(0.0);
uniform vec3 g_color1 = vec3(1.0);
uniform vec3 g_avg_color = vec3(1.0, 0.5, 1.0);
uniform float g_frequency = 1.0f;// [-¡Þ, ¡Þ]
// </ checker_board >

// < checker_board >
vec3 checker_board(vec2 _Texcoord)
	{
	vec2 _Fw = fwidth(_Texcoord);
	vec2 _Fuzz = _Fw * g_frequency * 2.0;
	float _Fuzzmax = max(_Fuzz.x, _Fuzz.y);

	vec2 _Checkpos = fract(_Texcoord * g_frequency);

	vec3 _Result = vec3(0.0);
	if(_Fuzzmax < 0.5)
		{
		vec2 _P = smoothstep( vec2(0.5), _Fuzz + vec2(0.5), _Checkpos )
				+ (1.0 - smoothstep( vec2(0.0), _Fuzz, _Checkpos) );
		_Result = mix(g_color0, g_color1, _P.x * _P.y + (1.0 - _P.x) * (1.0 - _P.y));
		_Result = mix(_Result, g_avg_color, smoothstep(0.125, 0.5, _Fuzzmax));
		}
	else
		{
		_Result = g_avg_color;
		}

	return (_Result);
	}
// </ checker_board >