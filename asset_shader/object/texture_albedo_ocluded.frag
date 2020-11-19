#version 330 core
uniform vec3 albedo    = vec3(0.7, 0.7, 0.7);
uniform vec3 fresnelF0 = vec3(0.1, 0.1, 0.1);
uniform sampler2D albedo_texture;

uniform mat4 world_to_Lproj;
uniform sampler2D scene_shadow_texture;

in vec3 vout_position;
in vec3 vout_normal;
in vec2 vout_texcoord;
in vec3 vout_tangent;
in vec3 vout_bitangent;
in float vout_depth;
layout(location = 0) out vec4 fout_albedo_and_visibility;
layout(location = 1) out vec4 fout_normal;

void main(){
	vec4 Lproj_position = world_to_Lproj * vec4(vout_position, 1.0);
	Lproj_position /= Lproj_position.wwww;

	fout_albedo_and_visibility.rgb = texture(albedo_texture, vout_texcoord).rgb;
//	fout_albedo_and_visibility.a   = 
	fout_normal = vec4(vout_normal * 0.5f + 0.5f, 0);
	gl_FragDepth = vout_depth;
}